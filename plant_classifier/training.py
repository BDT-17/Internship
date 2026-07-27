"""Training and evaluation orchestration."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

from plant_classifier.config import TrainConfig
from plant_classifier.data import build_dataloaders, discover_samples, stratified_split
from plant_classifier.losses import build_criterion
from plant_classifier.mixup import MixupCutmix, mixup_criterion
from plant_classifier.models import build_model
from plant_classifier.plots import save_confusion_matrix, save_history_plot, save_model_comparison


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, config: TrainConfig):
    if config.lr_scheduler == "none":
        return None
    if config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.lr_scheduler == "cosine_warmup":
        warmup_epochs = max(1, config.warmup_epochs)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs - warmup_epochs)
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs]
        )
    raise ValueError(f"Unknown lr_scheduler: {config.lr_scheduler}")


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
    optimizer=None,
    scaler=None,
    mixup_fn: MixupCutmix | None = None,
) -> tuple[dict[str, float], list[int], list[int]]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        use_mixup = is_train and mixup_fn is not None and mixup_fn.enabled

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                if use_mixup:
                    mixed, target_a, target_b, lam = mixup_fn(images, labels)
                    logits = model(mixed)
                    loss = mixup_criterion(criterion, logits, target_a, target_b, lam)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    return metrics, all_labels, all_preds


def train_one_model(
    model_name: str,
    model: nn.Module,
    learning_rate: float,
    config: TrainConfig,
    device: torch.device,
    amp_enabled: bool,
    class_names: list[str],
    train_samples,
    val_samples,
    test_samples,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    mixup_fn,
) -> tuple[dict, pd.DataFrame]:
    model_dir = config.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    trainable = filter(lambda param: param.requires_grad, model.parameters())
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    history: list[dict[str, float]] = []
    # Select the best checkpoint by the configured validation metric. For this
    # long-tailed task macro-F1 is the honest choice (val accuracy just tracks
    # the largest class); the column names below map to the history row keys.
    selection_key = config.model_selection_metric  # "val_f1_macro" | "val_accuracy"
    best_selection_score = -1.0
    best_val_acc = -1.0
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            amp_enabled,
            optimizer=optimizer,
            scaler=scaler,
            mixup_fn=mixup_fn,
        )
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, device, amp_enabled)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)
        print(
            f"[{model_name}] Epoch {epoch:02d} | "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_acc={row['val_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f}"
        )

        if config.save_every_epoch:
            torch.save(model.state_dict(), model_dir / f"epoch_{epoch:02d}.pth")

        if row[selection_key] > best_selection_score:
            best_selection_score = row[selection_key]
            best_val_acc = row["val_accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            print(f"[{model_name}] Early stopping triggered.")
            break

    elapsed = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics, test_labels, test_preds = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        amp_enabled,
    )
    history_df = pd.DataFrame(history)
    history_df.to_csv(model_dir / "history.csv", index=False)
    save_history_plot(history_df, model_dir / "history.png", model_name)
    save_confusion_matrix(
        test_labels,
        test_preds,
        model_dir / "confusion_matrix.png",
        f"{model_name} Test Confusion Matrix",
    )
    pd.DataFrame(
        {
            "true_index": test_labels,
            "true_class": [class_names[i] for i in test_labels],
            "predicted_index": test_preds,
            "predicted_class": [class_names[i] for i in test_preds],
            "correct": [int(t == p) for t, p in zip(test_labels, test_preds)],
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)

    summary = {
        "model_name": model_name,
        "epochs_completed": len(history),
        "model_selection_metric": selection_key,
        "best_selection_score": best_selection_score,
        "best_val_accuracy": best_val_acc,
        "test_metrics": test_metrics,
        "runtime_seconds": elapsed,
        "num_classes": len(class_names),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
    }
    with open(model_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary, history_df


def run_training(config: TrainConfig) -> pd.DataFrame:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = config.use_amp and device.type == "cuda"
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    class_names, samples, class_counts = discover_samples(config.dataset_dir)
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        config.train_ratio,
        config.val_ratio,
        config.seed,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        config.image_size,
        config.batch_size,
        config.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Dataset:", config.dataset_dir)
    print("Total classes:", len(class_names))
    print("Total images:", len(samples))
    print("Min class size:", min(class_counts.values()))
    print("Max class size:", max(class_counts.values()))
    print("Train:", len(train_samples), "Val:", len(val_samples), "Test:", len(test_samples))

    with open(config.output_dir / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    pd.Series(class_counts).sort_index().to_csv(config.output_dir / "class_counts.csv")

    train_class_counts = [0] * len(class_names)
    for _, label in train_samples:
        train_class_counts[label] += 1
    criterion = build_criterion(
        config.loss,
        label_smoothing=config.label_smoothing,
        class_counts=train_class_counts,
        focal_gamma=config.focal_gamma,
        cb_beta=config.cb_beta,
        device=device,
    )
    mixup_fn = MixupCutmix(
        mixup_alpha=config.mixup_alpha,
        cutmix_alpha=config.cutmix_alpha,
        prob=config.mixup_prob,
    )
    print(
        f"Loss: {config.loss} | optimizer: {config.optimizer} | "
        f"scheduler: {config.lr_scheduler} | label_smoothing: {config.label_smoothing} | "
        f"mixup: {config.mixup_alpha} | cutmix: {config.cutmix_alpha}"
    )

    learning_rates = {
        "custom_cnn": config.learning_rate_cnn,
        "custom_cnn_v2": config.learning_rate_cnn_v2,
        "resnet50_feature_extraction": config.learning_rate_feature_extractor,
        "resnet50_fine_tuning": config.learning_rate_finetune,
        "efficientnet_v2_s": config.learning_rate_efficientnet_v2_s,
        "convnext_tiny": config.learning_rate_convnext_tiny,
    }
    all_summaries = []

    for model_name in config.models:
        print(f"\n===== Training {model_name} =====")
        model = build_model(
            model_name,
            len(class_names),
            use_pretrained_weights=config.use_pretrained_weights,
        )
        summary, _ = train_one_model(
            model_name,
            model,
            learning_rates[model_name],
            config,
            device,
            amp_enabled,
            class_names,
            train_samples,
            val_samples,
            test_samples,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            mixup_fn,
        )
        all_summaries.append(summary)

    summary_df = pd.DataFrame(
        [
            {
                "model_name": item["model_name"],
                "epochs_completed": item["epochs_completed"],
                "best_val_accuracy": item["best_val_accuracy"],
                "test_accuracy": item["test_metrics"]["accuracy"],
                "test_f1_macro": item["test_metrics"]["f1_macro"],
                "test_precision_macro": item["test_metrics"]["precision_macro"],
                "test_recall_macro": item["test_metrics"]["recall_macro"],
                "runtime_minutes": item["runtime_seconds"] / 60.0,
            }
            for item in all_summaries
        ]
    )
    summary_df.to_csv(config.output_dir / "all_models_summary.csv", index=False)
    with open(config.output_dir / "all_models_summary.json", "w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=2)
    save_model_comparison(summary_df, config.output_dir / "model_comparison.png")

    return summary_df


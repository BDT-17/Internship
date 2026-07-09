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
from plant_classifier.models import build_model
from plant_classifier.plots import save_confusion_matrix, save_history_plot, save_model_comparison


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
    optimizer=None,
    scaler=None,
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

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
) -> tuple[dict, pd.DataFrame]:
    model_dir = config.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    history: list[dict[str, float]] = []
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
        )
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, device, amp_enabled)

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

        if row["val_accuracy"] > best_val_acc:
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

    learning_rates = {
        "custom_cnn": config.learning_rate_cnn,
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


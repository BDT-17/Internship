"""Deep post-training analysis for the plant classification benchmark.

This module operates on already-trained models. It reconstructs the exact
deterministic test split (same seed and ratios used during training), reruns
inference with each saved ``best_model.pth``, and produces the class-level and
group-level artifacts needed for the internship report:

- ``test_predictions.csv``     raw label/prediction/confidence per test image
- ``per_class_metrics.csv``    precision/recall/F1/support for every class
- ``group_analysis.csv``       crop (PlantVillage) vs tree-genus (Leafsnap)
- ``most_confused_pairs.csv``  ranked off-diagonal confusion pairs
- ``per_class_f1_vs_size.png`` per-class F1 against training set size
- ``analysis_summary.json``    headline numbers for the whole run

Because the split is deterministic, this reproduces the true held-out test set
without retraining. It can also be called from ``training.py`` right after a
model finishes, in which case the in-memory predictions are reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from plant_classifier.config import TrainConfig
from plant_classifier.data import build_transforms, discover_samples, stratified_split
from plant_classifier.models import build_model

# PlantVillage crop-level classes. Folder names are matched case-insensitively
# by prefix, so "Pepper", "Pepper__bell", "Potato", "Tomato" all count as crops.
DEFAULT_CROP_PREFIXES = ("pepper", "potato", "tomato")


@dataclass(slots=True)
class GroupLabels:
    """Assigns each class index to the crop or tree-genus group."""

    class_names: list[str]
    crop_prefixes: tuple[str, ...] = DEFAULT_CROP_PREFIXES
    crop_indices: set[int] = field(default_factory=set)
    genus_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        for idx, name in enumerate(self.class_names):
            lowered = name.lower()
            if any(lowered.startswith(prefix) for prefix in self.crop_prefixes):
                self.crop_indices.add(idx)
            else:
                self.genus_indices.add(idx)

    def group_of(self, idx: int) -> str:
        return "crop" if idx in self.crop_indices else "tree_genus"


def _resolve_test_split(
    config: TrainConfig,
) -> tuple[list[str], list[tuple[Path, int]], dict[str, int]]:
    """Rebuild the deterministic held-out test split used during training."""
    class_names, samples, class_counts = discover_samples(config.dataset_dir)
    train_samples, _, test_samples = stratified_split(
        samples,
        config.train_ratio,
        config.val_ratio,
        config.seed,
    )
    train_counts: dict[str, int] = {name: 0 for name in class_names}
    for _, label in train_samples:
        train_counts[class_names[label]] += 1
    return class_names, test_samples, train_counts


def collect_test_predictions(
    model_name: str,
    weights_path: Path,
    config: TrainConfig,
    device: torch.device | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Run inference over the reconstructed test split for one trained model.

    Returns ``(class_names, labels, preds, confidences, train_counts)``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names, test_samples, train_counts = _resolve_test_split(config)
    _, eval_transform = build_transforms(config.image_size)

    model = build_model(model_name, len(class_names), use_pretrained_weights=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    labels: list[int] = []
    preds: list[int] = []
    confidences: list[float] = []

    from plant_classifier.data import ImageListDataset
    from torch.utils.data import DataLoader

    dataset = ImageListDataset(test_samples, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            batch_conf, batch_pred = torch.max(probs, dim=1)
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(batch_pred.cpu().numpy().tolist())
            confidences.extend(batch_conf.cpu().numpy().tolist())

    return (
        class_names,
        np.asarray(labels),
        np.asarray(preds),
        np.asarray(confidences),
        train_counts,
    )


def per_class_metrics_frame(
    class_names: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    train_counts: dict[str, int],
    groups: GroupLabels,
) -> pd.DataFrame:
    indices = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=indices,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_name": class_names,
            "group": [groups.group_of(idx) for idx in indices],
            "train_images": [train_counts.get(name, 0) for name in class_names],
            "test_support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ).sort_values("f1", ascending=True, ignore_index=True)


def group_analysis_frame(
    per_class: pd.DataFrame,
    labels: np.ndarray,
    preds: np.ndarray,
    groups: GroupLabels,
) -> pd.DataFrame:
    rows = []
    label_group = np.array(["crop" if g in groups.crop_indices else "tree_genus" for g in labels])
    for group_name in ("crop", "tree_genus"):
        mask = label_group == group_name
        n = int(mask.sum())
        if n == 0:
            continue
        group_classes = per_class[per_class["group"] == group_name]
        rows.append(
            {
                "group": group_name,
                "num_classes": int(len(group_classes)),
                "test_images": n,
                "accuracy": float(accuracy_score(labels[mask], preds[mask])),
                "macro_f1": float(
                    f1_score(labels[mask], preds[mask], average="macro", zero_division=0)
                ),
                "mean_train_images": float(group_classes["train_images"].mean()),
            }
        )
    return pd.DataFrame(rows)


def most_confused_pairs(
    class_names: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    top_n: int = 25,
) -> pd.DataFrame:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    rows = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            support = int(cm[true_idx].sum())
            rows.append(
                {
                    "true_class": class_names[true_idx],
                    "predicted_class": class_names[pred_idx],
                    "count": count,
                    "true_support": support,
                    "error_rate": count / support if support else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("count", ascending=False, ignore_index=True).head(top_n)


def save_per_class_f1_vs_size(per_class: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))
    for group_name, color in (("crop", "#d1495b"), ("tree_genus", "#2e7d9a")):
        subset = per_class[per_class["group"] == group_name]
        if subset.empty:
            continue
        plt.scatter(
            subset["train_images"],
            subset["f1"],
            label=group_name,
            color=color,
            alpha=0.75,
            edgecolors="none",
        )
    plt.xscale("log")
    plt.xlabel("Training images per class (log scale)")
    plt.ylabel("Per-class test F1")
    plt.title("Per-class F1 vs class size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_convergence_comparison(
    output_dir: Path,
    model_names: tuple[str, ...],
    output_path: Path,
) -> pd.DataFrame:
    """Overlay validation curves of every model from their ``history.csv``.

    Uses only training logs, so it needs no weights or dataset. Returns a small
    per-model convergence table (best epoch, best val accuracy, overfit gap).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = []
    for model_name in model_names:
        history_path = output_dir / model_name / "history.csv"
        if not history_path.exists():
            print(f"Skipping convergence for {model_name}: no {history_path}")
            continue
        history = pd.read_csv(history_path)
        axes[0].plot(history["epoch"], history["val_loss"], label=model_name)
        axes[1].plot(history["epoch"], history["val_accuracy"], label=model_name)

        best_row = history.loc[history["val_accuracy"].idxmax()]
        final_row = history.iloc[-1]
        rows.append(
            {
                "model_name": model_name,
                "epochs_completed": int(len(history)),
                "best_epoch": int(best_row["epoch"]),
                "best_val_accuracy": float(best_row["val_accuracy"]),
                "final_train_accuracy": float(final_row["train_accuracy"]),
                "final_val_accuracy": float(final_row["val_accuracy"]),
                "overfit_gap": float(final_row["train_accuracy"] - final_row["val_accuracy"]),
            }
        )

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "convergence_summary.csv", index=False)
    return table


def analyze_model(
    model_name: str,
    model_dir: Path,
    config: TrainConfig,
    labels: np.ndarray | None = None,
    preds: np.ndarray | None = None,
    class_names: list[str] | None = None,
    train_counts: dict[str, int] | None = None,
    crop_prefixes: tuple[str, ...] = DEFAULT_CROP_PREFIXES,
    top_confused: int = 25,
) -> dict:
    """Produce all analysis artifacts for a single model directory.

    If ``labels``/``preds`` are supplied (e.g. reused from training), inference
    is skipped; otherwise ``best_model.pth`` is loaded and rerun.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    confidences: np.ndarray | None = None

    if labels is None or preds is None or class_names is None or train_counts is None:
        weights_path = model_dir / "best_model.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights for analysis: {weights_path}")
        class_names, labels, preds, confidences, train_counts = collect_test_predictions(
            model_name, weights_path, config
        )

    groups = GroupLabels(class_names=class_names, crop_prefixes=crop_prefixes)

    predictions_frame = pd.DataFrame(
        {
            "true_index": labels,
            "true_class": [class_names[i] for i in labels],
            "predicted_index": preds,
            "predicted_class": [class_names[i] for i in preds],
            "correct": (labels == preds).astype(int),
        }
    )
    if confidences is not None:
        predictions_frame["confidence"] = confidences
    predictions_frame.to_csv(model_dir / "test_predictions.csv", index=False)

    per_class = per_class_metrics_frame(class_names, labels, preds, train_counts, groups)
    per_class.to_csv(model_dir / "per_class_metrics.csv", index=False)

    group_frame = group_analysis_frame(per_class, labels, preds, groups)
    group_frame.to_csv(model_dir / "group_analysis.csv", index=False)

    confused = most_confused_pairs(class_names, labels, preds, top_n=top_confused)
    confused.to_csv(model_dir / "most_confused_pairs.csv", index=False)

    save_per_class_f1_vs_size(per_class, model_dir / "per_class_f1_vs_size.png")

    weakest = per_class.head(10)[["class_name", "group", "train_images", "f1"]]
    summary = {
        "model_name": model_name,
        "test_accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "num_classes": len(class_names),
        "test_images": int(len(labels)),
        "group_analysis": group_frame.to_dict(orient="records"),
        "weakest_classes": weakest.to_dict(orient="records"),
    }
    with open(model_dir / "analysis_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    return summary


def analyze_all_models(
    config: TrainConfig,
    model_names: tuple[str, ...] | None = None,
    crop_prefixes: tuple[str, ...] = DEFAULT_CROP_PREFIXES,
) -> pd.DataFrame:
    """Analyze every trained model under ``config.output_dir`` and compare them."""
    names = model_names or config.models
    summaries = []
    for model_name in names:
        model_dir = config.output_dir / model_name
        if not (model_dir / "best_model.pth").exists():
            print(f"Skipping {model_name}: no best_model.pth in {model_dir}")
            continue
        print(f"\n===== Analyzing {model_name} =====")
        summary = analyze_model(model_name, model_dir, config, crop_prefixes=crop_prefixes)
        summaries.append(summary)
        print(
            f"  test_acc={summary['test_accuracy']:.4f} "
            f"macro_f1={summary['macro_f1']:.4f} "
            f"weighted_f1={summary['weighted_f1']:.4f}"
        )

    if not summaries:
        raise RuntimeError("No trained models found to analyze.")

    comparison_rows = []
    for summary in summaries:
        row = {
            "model_name": summary["model_name"],
            "test_accuracy": summary["test_accuracy"],
            "macro_f1": summary["macro_f1"],
            "weighted_f1": summary["weighted_f1"],
        }
        for group in summary["group_analysis"]:
            row[f"{group['group']}_accuracy"] = group["accuracy"]
            row[f"{group['group']}_macro_f1"] = group["macro_f1"]
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(config.output_dir / "analysis_comparison.csv", index=False)
    with open(config.output_dir / "analysis_comparison.json", "w", encoding="utf-8") as file:
        json.dump(summaries, file, indent=2)
    print("\nWrote analysis_comparison.csv")
    return comparison

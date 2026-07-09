"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_history_plot(history_df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val")
    axes[0].set_title(f"{model_name} Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="train")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val")
    axes[1].set_title(f"{model_name} Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_model_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    summary_df.plot(
        x="model_name",
        y=["test_accuracy", "test_f1_macro"],
        kind="bar",
        figsize=(10, 5),
        rot=20,
    )
    plt.title("Model Comparison on Test Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


"""Path helpers for local and Kaggle runs."""

from __future__ import annotations

from pathlib import Path


def find_dataset_dir(
    input_root: Path = Path("/kaggle/input"),
    dataset_name: str = "dataset_plant_classification",
) -> Path:
    direct_candidates = [
        input_root / "plant_classification" / dataset_name,
        input_root / dataset_name,
        Path(dataset_name),
    ]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    if input_root.exists():
        for candidate in sorted(input_root.rglob(dataset_name)):
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(f"Could not find {dataset_name} under {input_root}.")


def find_model_root(
    input_root: Path = Path("/kaggle/input"),
    output_dir_name: str = "plant_training_outputs",
) -> Path:
    direct_candidates = [
        input_root / "best-models" / output_dir_name,
        input_root / output_dir_name,
        Path(output_dir_name),
    ]
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    if input_root.exists():
        for candidate in sorted(input_root.rglob(output_dir_name)):
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(f"Could not find {output_dir_name} under {input_root}.")


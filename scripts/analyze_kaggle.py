"""Run deep post-training analysis on already-trained plant models.

This does not retrain. It reconstructs the deterministic held-out test split,
reruns inference with each saved ``best_model.pth``, and writes per-class,
crop-vs-genus, and confusion-pair artifacts next to the existing outputs.

Kaggle example (models and dataset attached as inputs):

    python scripts/analyze_kaggle.py \
        --model-root /kaggle/input/best-models/plant_training_outputs \
        --output-dir /kaggle/working/analysis

Local example:

    python scripts/analyze_kaggle.py \
        --dataset-dir dataset_plant_classification \
        --output-dir plant_training_outputs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plant_classifier.analysis import DEFAULT_CROP_PREFIXES, analyze_all_models
from plant_classifier.config import DEFAULT_MODEL_NAMES, TrainConfig
from plant_classifier.paths import find_dataset_dir, find_model_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODEL_NAMES))
    parser.add_argument(
        "--crop-prefixes",
        nargs="+",
        default=list(DEFAULT_CROP_PREFIXES),
        help="Class-name prefixes treated as PlantVillage crop classes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or find_dataset_dir(args.input_root)
    model_root = args.model_root or find_model_root(args.input_root)

    # Analysis reads best_model.pth from model_root and writes artifacts beside
    # them. On Kaggle the input tree is read-only, so mirror it into output_dir.
    output_dir = args.output_dir or model_root
    if output_dir != model_root:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Mirror the weights (for inference) and history.csv (for the training
        # curves) so the analysis output_dir is a self-contained bundle.
        for model_name in args.models:
            dst = output_dir / model_name
            for filename in ("best_model.pth", "history.csv"):
                src = model_root / model_name / filename
                if src.exists():
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst / filename)

    config = TrainConfig(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        models=tuple(args.models),
    )
    config.validate()

    print("Dataset:", dataset_dir)
    print("Model root:", model_root)
    print("Output:", output_dir)

    comparison = analyze_all_models(
        config,
        model_names=tuple(args.models),
        crop_prefixes=tuple(args.crop_prefixes),
    )
    print("\n=== Analysis comparison ===")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

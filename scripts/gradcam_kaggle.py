"""Render Grad-CAM lab-versus-field panels for already-trained plant models.

This does not retrain and does not re-run the full test set. It reconstructs the
deterministic split, finds genera where a reference model classifies a Leafsnap
*lab* image correctly but a Leafsnap *field* image of the same genus wrongly,
and renders Grad-CAM for every model on those same pairs.

Because the genus is held fixed across each pair, the panels test whether the
lab-to-field accuracy gap comes with a corresponding shift in where the model
looks -- the mechanism proposed in the report's domain-shift section.

Only a few dozen images are processed, so this runs comfortably on CPU.

Kaggle example (models and dataset attached as inputs):

    python scripts/gradcam_kaggle.py \
        --model-root /kaggle/working/model_root \
        --output-dir /kaggle/working/analysis

Local example:

    python scripts/gradcam_kaggle.py \
        --dataset-dir dataset_plant_classification \
        --model-root plant_training_outputs \
        --output-dir plant_training_outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plant_classifier.config import DEFAULT_MODEL_NAMES, TrainConfig
from plant_classifier.gradcam import run_gradcam_comparison
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
        "--reference-model",
        default="custom_cnn",
        help=(
            "Model used to choose the lab-correct / field-wrong pairs. Defaults "
            "to the weakest model, which yields the clearest failure cases."
        ),
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=6,
        help="Number of genera to visualize (one panel each).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or find_dataset_dir(args.input_root)
    model_root = args.model_root or find_model_root(args.input_root)
    output_dir = args.output_dir or model_root
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print("Output:", output_dir / "gradcam")

    summary = run_gradcam_comparison(
        config,
        model_names=tuple(args.models),
        model_root=model_root,
        output_dir=output_dir,
        reference_model=args.reference_model,
        max_pairs=args.max_pairs,
    )
    print("\n=== Grad-CAM summary (centre attention = fraction of CAM mass in image centre) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

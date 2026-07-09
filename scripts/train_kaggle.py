"""Train plant classification models from Kaggle or a local terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plant_classifier.config import DEFAULT_MODEL_NAMES, TrainConfig
from plant_classifier.paths import find_dataset_dir
from plant_classifier.training import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/plant_training_outputs"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-save-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or find_dataset_dir(args.input_root)
    config = TrainConfig(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        seed=args.seed,
        use_amp=not args.no_amp,
        save_every_epoch=not args.no_save_every_epoch,
        use_pretrained_weights=not args.no_pretrained,
        models=tuple(args.models) if args.models else DEFAULT_MODEL_NAMES,
    )
    summary_df = run_training(config)
    print(summary_df)


if __name__ == "__main__":
    main()

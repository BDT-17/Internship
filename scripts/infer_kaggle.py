"""Run showcase inference for a trained plant classification model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plant_classifier.inference import load_class_names, predict_image
from plant_classifier.paths import find_dataset_dir, find_model_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model-name", default="resnet50_fine_tuning")
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-path", type=Path, default=Path("/kaggle/working/prediction.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir or find_dataset_dir(args.input_root)
    model_root = args.model_root or find_model_root(args.input_root)
    image_path = args.image_path or (dataset_dir / "Tomato")
    weights_path = args.weights_path or (model_root / args.model_name / "best_model.pth")

    class_names = load_class_names(dataset_dir=dataset_dir, model_root=model_root)
    rows = predict_image(
        model_name=args.model_name,
        weights_path=weights_path,
        image_path=image_path,
        class_names=class_names,
        image_size=args.image_size,
        top_k=args.top_k,
        output_path=args.output_path,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

"""Inference helpers for trained plant classification models."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from plant_classifier.data import VALID_EXTENSIONS, build_transforms, discover_samples
from plant_classifier.models import build_model


def load_class_names(dataset_dir: Path | None = None, model_root: Path | None = None) -> list[str]:
    if model_root is not None:
        class_names_path = model_root / "class_names.json"
        if class_names_path.exists():
            return json.loads(class_names_path.read_text(encoding="utf-8"))

    if dataset_dir is None:
        raise ValueError("dataset_dir is required when class_names.json is not available.")
    class_names, _, _ = discover_samples(dataset_dir)
    return class_names


def choose_image(image_path: Path) -> Path:
    if image_path.is_dir():
        candidates = sorted(
            path
            for path in image_path.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(f"No images found in folder: {image_path}")
        return random.choice(candidates)
    return image_path


def predict_image(
    model_name: str,
    weights_path: Path,
    image_path: Path,
    class_names: list[str],
    image_size: int = 224,
    top_k: int = 5,
    output_path: Path | None = None,
) -> list[dict[str, float | str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_transform = build_transforms(image_size)

    model = build_model(model_name, len(class_names), use_pretrained_weights=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    selected_image = choose_image(image_path)
    image = Image.open(selected_image).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))

    rows: list[dict[str, float | str]] = []
    for idx, prob in zip(top_indices.detach().cpu().numpy(), top_probs.detach().cpu().numpy()):
        rows.append(
            {
                "class_name": class_names[int(idx)],
                "probability": float(prob),
                "probability_percent": float(prob) * 100,
            }
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predicted = rows[0]
        plt.figure(figsize=(7, 7))
        plt.imshow(image)
        plt.title(
            f"Prediction: {predicted['class_name']} "
            f"({predicted['probability_percent']:.2f}%)"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    print("Using image:", selected_image)
    print("Predicted class:", rows[0]["class_name"])
    print(f"Confidence: {rows[0]['probability_percent']:.2f}%")
    return rows


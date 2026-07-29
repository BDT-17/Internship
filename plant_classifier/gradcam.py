"""Grad-CAM saliency for the lab-versus-field generalization question.

Grad-CAM :cite:`selvaraju2017` weights the last convolutional feature maps by
the gradient of a class score flowing into them, giving a class-discriminative
heatmap over the input.

The point of running it here is not to produce pretty pictures but to test a
specific, falsifiable claim behind the source-conditioned accuracy gap: if a
model relies on capture conditions rather than leaf morphology, its attention
should stay on the leaf for ``leafsnap_lab`` images it classifies correctly and
drift onto background for ``leafsnap_field`` images of the *same genus* that it
gets wrong. :func:`select_lab_field_pairs` builds exactly those pairs, holding
the genus fixed so capture condition is the only variable.

Artifacts written under ``<output_dir>/gradcam``:

- ``<genus>_panel.png``   one row per model, lab-correct beside field-wrong
- ``gradcam_pairs.csv``   the chosen pairs and each model's prediction
- ``gradcam_summary.csv`` per-model mean leaf-region attention mass
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from plant_classifier.analysis import _resolve_test_split, source_of_path
from plant_classifier.config import TrainConfig
from plant_classifier.data import build_transforms
from plant_classifier.models import build_model

# ImageNet statistics used by build_transforms; needed to undo normalization
# when overlaying a heatmap on the original image.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Return the last convolutional block, where Grad-CAM hooks attach.

    The last conv stage is the standard choice: it still carries spatial layout
    (7x7 or 14x14 at 224px input) while encoding high-level semantics. Each
    architecture in this project exposes it under a different attribute, so the
    lookup is explicit rather than guessed.
    """
    if model_name == "custom_cnn_v2":
        return model.stage4
    if model_name == "custom_cnn":
        # CustomCNN's `features` ends with AdaptiveAvgPool2d((1,1)), which has no
        # spatial extent -- hooking it would give a 1x1 "heatmap". Take the last
        # Conv2d instead, whose output is still spatial.
        conv_indices = [
            i for i, layer in enumerate(model.features)
            if isinstance(layer, torch.nn.Conv2d)
        ]
        if not conv_indices:
            raise ValueError("custom_cnn has no Conv2d layer to hook")
        return model.features[conv_indices[-1]]
    if model_name.startswith("resnet50"):
        return model.layer4
    if model_name.startswith("efficientnet"):
        return model.features
    if model_name.startswith("convnext"):
        return model.features
    raise ValueError(f"No Grad-CAM target layer registered for {model_name!r}")


class GradCAM:
    """Grad-CAM for a single model, holding forward/backward hooks.

    Use as a context manager so the hooks are always removed:

        with GradCAM(model, layer) as cam:
            heatmap, pred, conf = cam(input_tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> "GradCAM":
        self._handles.append(
            self.target_layer.register_forward_hook(self._save_activation)
        )
        # full_backward_hook fires with the gradient w.r.t. the layer output.
        self._handles.append(
            self.target_layer.register_full_backward_hook(self._save_gradient)
        )
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(
        self, image: torch.Tensor, class_index: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        """Return ``(heatmap, predicted_index, confidence)`` for one image.

        ``image`` is a normalized ``(1, 3, H, W)`` tensor. ``class_index``
        defaults to the model's own prediction -- the right target when asking
        "why did it say that?". The heatmap is upsampled to the input size and
        scaled to ``[0, 1]``.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probabilities = F.softmax(logits, dim=1)
        predicted = int(logits.argmax(dim=1).item())
        target = predicted if class_index is None else class_index
        confidence = float(probabilities[0, target].item())

        logits[0, target].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks captured nothing; check the target layer.")

        # alpha_k = global-average-pooled gradient for channel k.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # keep only evidence *for* the class
        cam = F.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        heatmap = cam[0, 0].cpu().numpy()
        peak = float(heatmap.max())
        if peak > 0:
            heatmap = heatmap / peak
        return heatmap, predicted, confidence


@dataclass(slots=True)
class LabFieldPair:
    """One genus with a lab image the model got right and a field image it got wrong."""

    class_name: str
    class_index: int
    lab_path: Path
    field_path: Path


def select_lab_field_pairs(
    config: TrainConfig,
    reference_model: str,
    weights_path: Path,
    max_pairs: int = 6,
    device: torch.device | None = None,
) -> list[LabFieldPair]:
    """Pick genera where ``reference_model`` is right on lab and wrong on field.

    Holding the genus fixed across the two images is what makes the comparison
    interpretable: the label is identical, so any difference in where the model
    looks is attributable to capture conditions alone.

    The reference model is normally the weakest one (largest lab-to-field gap),
    since it supplies the clearest failure cases; all models are then visualized
    on the same pairs so the panels are comparable.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names, test_samples, _ = _resolve_test_split(config)
    _, eval_transform = build_transforms(config.image_size)

    model = build_model(reference_model, len(class_names), use_pretrained_weights=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device).eval()

    # Bucket test images by (class, source) so candidates are cheap to look up.
    lab_by_class: dict[int, list[Path]] = {}
    field_by_class: dict[int, list[Path]] = {}
    for path, label in test_samples:
        source = source_of_path(path)
        if source == "leafsnap_lab":
            lab_by_class.setdefault(label, []).append(path)
        elif source == "leafsnap_field":
            field_by_class.setdefault(label, []).append(path)

    shared = sorted(set(lab_by_class) & set(field_by_class))

    def predict(path: Path) -> int:
        image = Image.open(path).convert("RGB")
        tensor = eval_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            return int(model(tensor).argmax(dim=1).item())

    pairs: list[LabFieldPair] = []
    for label in shared:
        lab_hit = next((p for p in lab_by_class[label][:12] if predict(p) == label), None)
        if lab_hit is None:
            continue
        field_miss = next(
            (p for p in field_by_class[label][:12] if predict(p) != label), None
        )
        if field_miss is None:
            continue
        pairs.append(
            LabFieldPair(
                class_name=class_names[label],
                class_index=label,
                lab_path=lab_hit,
                field_path=field_miss,
            )
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display."""
    array = tensor[0].cpu().numpy().transpose(1, 2, 0)
    return np.clip(array * _STD + _MEAN, 0.0, 1.0)


def _overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a jet-coloured heatmap over the image."""
    import matplotlib.cm as cm

    colored = cm.jet(heatmap)[..., :3]
    return np.clip((1 - alpha) * image + alpha * colored, 0.0, 1.0)


def _attention_mass_center(heatmap: np.ndarray, border_fraction: float = 0.25) -> float:
    """Fraction of Grad-CAM mass inside the central region of the image.

    A crude but model-agnostic proxy for "is the model looking at the leaf?".
    Leaves are roughly centred in both Leafsnap domains, so attention drifting
    to the border is evidence of reliance on background. Reported alongside the
    panels rather than as a standalone claim.
    """
    height, width = heatmap.shape
    top, bottom = int(height * border_fraction), int(height * (1 - border_fraction))
    left, right = int(width * border_fraction), int(width * (1 - border_fraction))
    total = float(heatmap.sum())
    if total <= 0:
        return float("nan")
    return float(heatmap[top:bottom, left:right].sum() / total)


def run_gradcam_comparison(
    config: TrainConfig,
    model_names: Sequence[str],
    model_root: Path,
    output_dir: Path,
    reference_model: str | None = None,
    max_pairs: int = 6,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Build lab/field pairs, run Grad-CAM for every model, write panels + CSVs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available = [
        name for name in model_names if (model_root / name / "best_model.pth").exists()
    ]
    if not available:
        raise SystemExit(f"No best_model.pth found under {model_root}")

    # Default to the weakest model as the pair selector: it produces the most
    # field failures, and the same pairs are then shown for every model.
    reference = reference_model or available[0]
    print(f"Selecting lab/field pairs with reference model: {reference}")
    pairs = select_lab_field_pairs(
        config,
        reference,
        model_root / reference / "best_model.pth",
        max_pairs=max_pairs,
        device=device,
    )
    if not pairs:
        raise SystemExit(
            "No genus had both a correct lab image and an incorrect field image; "
            "try a weaker --reference-model or raise --max-pairs."
        )
    print(f"Found {len(pairs)} lab-correct / field-wrong pairs: "
          + ", ".join(p.class_name for p in pairs))

    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    class_names, _, _ = _resolve_test_split(config)
    _, eval_transform = build_transforms(config.image_size)

    rows: list[dict] = []
    # cache: model_name -> (model, target_layer)
    loaded: dict[str, tuple[torch.nn.Module, torch.nn.Module]] = {}
    for name in available:
        model = build_model(name, len(class_names), use_pretrained_weights=False)
        model.load_state_dict(
            torch.load(model_root / name / "best_model.pth", map_location=device)
        )
        model = model.to(device).eval()
        loaded[name] = (model, resolve_target_layer(model, name))

    for pair in pairs:
        n_rows = len(available)
        fig, axes = plt.subplots(
            n_rows, 4, figsize=(11.5, 2.9 * n_rows), squeeze=False
        )
        for row, name in enumerate(available):
            model, target_layer = loaded[name]
            for col_offset, (path, domain) in enumerate(
                ((pair.lab_path, "lab"), (pair.field_path, "field"))
            ):
                image = Image.open(path).convert("RGB")
                tensor = eval_transform(image).unsqueeze(0).to(device)
                tensor.requires_grad_(True)
                with GradCAM(model, target_layer) as cam:
                    heatmap, predicted, confidence = cam(tensor)

                display = _denormalize(tensor.detach())
                centre = _attention_mass_center(heatmap)
                rows.append(
                    {
                        "model_name": name,
                        "class_name": pair.class_name,
                        "domain": domain,
                        "image": path.name,
                        "predicted_class": class_names[predicted],
                        "correct": int(predicted == pair.class_index),
                        "confidence": confidence,
                        "centre_attention": centre,
                    }
                )

                ax_img = axes[row][col_offset * 2]
                ax_cam = axes[row][col_offset * 2 + 1]
                ax_img.imshow(display)
                ax_cam.imshow(_overlay(display, heatmap))
                mark = "OK" if predicted == pair.class_index else "WRONG"
                ax_img.set_title(f"{domain}: {path.name[:18]}", fontsize=8)
                ax_cam.set_title(
                    f"{mark} -> {class_names[predicted]} ({confidence:.2f})\n"
                    f"centre mass {centre:.2f}",
                    fontsize=8,
                )
                for ax in (ax_img, ax_cam):
                    ax.set_xticks([])
                    ax.set_yticks([])
            axes[row][0].set_ylabel(name, fontsize=9)

        fig.suptitle(
            f"Grad-CAM, genus {pair.class_name}: lab (correct) versus field (misclassified)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        panel_path = gradcam_dir / f"{pair.class_name}_panel.png"
        fig.savefig(panel_path, dpi=150)
        plt.close(fig)
        print("  wrote", panel_path.name)

    frame = pd.DataFrame(rows)
    frame.to_csv(gradcam_dir / "gradcam_pairs.csv", index=False)

    summary = (
        frame.groupby(["model_name", "domain"], as_index=False)
        .agg(
            images=("image", "count"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_centre_attention=("centre_attention", "mean"),
        )
        .sort_values(["model_name", "domain"])
    )
    summary.to_csv(gradcam_dir / "gradcam_summary.csv", index=False)
    return summary

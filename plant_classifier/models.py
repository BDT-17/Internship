"""Model definitions used by training and inference."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    ResNet50_Weights,
)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def load_resnet50_backbone(use_pretrained_weights: bool = True) -> nn.Module:
    if not use_pretrained_weights:
        print("USE_PRETRAINED_WEIGHTS=False -> using ResNet50 without pretrained weights")
        return models.resnet50(weights=None)

    try:
        return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    except Exception as exc:
        print(f"Warning: could not load pretrained ResNet50 weights: {exc}")
        print("Falling back to weights=None so the run can continue offline.")
        return models.resnet50(weights=None)


def build_resnet50_feature_extractor(
    num_classes: int,
    use_pretrained_weights: bool = True,
) -> nn.Module:
    model = load_resnet50_backbone(use_pretrained_weights=use_pretrained_weights)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_resnet50_finetune(
    num_classes: int,
    use_pretrained_weights: bool = True,
) -> nn.Module:
    model = load_resnet50_backbone(use_pretrained_weights=use_pretrained_weights)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_backbone_with_fallback(builder, weights, name: str, use_pretrained_weights: bool):
    """Load a torchvision backbone, falling back to no weights if offline."""
    if not use_pretrained_weights:
        print(f"USE_PRETRAINED_WEIGHTS=False -> using {name} without pretrained weights")
        return builder(weights=None)
    try:
        return builder(weights=weights)
    except Exception as exc:
        print(f"Warning: could not load pretrained {name} weights: {exc}")
        print("Falling back to weights=None so the run can continue offline.")
        return builder(weights=None)


def _unfreeze_last_feature_stages(model: nn.Module, num_stages: int = 2) -> None:
    """Freeze everything, then unfreeze the last ``num_stages`` feature blocks.

    Both EfficientNetV2 and ConvNeXt expose their backbone as ``model.features``
    (an ``nn.Sequential``), so unfreezing the final stages mirrors the ResNet50
    fine-tuning setup where only ``layer4`` and the head are trainable.
    """
    for param in model.parameters():
        param.requires_grad = False
    for stage in list(model.features.children())[-num_stages:]:
        for param in stage.parameters():
            param.requires_grad = True


def build_efficientnet_v2_s_finetune(
    num_classes: int,
    use_pretrained_weights: bool = True,
) -> nn.Module:
    model = _load_backbone_with_fallback(
        models.efficientnet_v2_s,
        EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "EfficientNetV2-S",
        use_pretrained_weights,
    )
    # torchvision head: classifier = Sequential(Dropout, Linear(1280, 1000)).
    in_features = model.classifier[1].in_features
    _unfreeze_last_feature_stages(model, num_stages=2)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_convnext_tiny_finetune(
    num_classes: int,
    use_pretrained_weights: bool = True,
) -> nn.Module:
    model = _load_backbone_with_fallback(
        models.convnext_tiny,
        ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "ConvNeXt-Tiny",
        use_pretrained_weights,
    )
    # torchvision head: classifier = Sequential(LayerNorm2d, Flatten, Linear(768, 1000)).
    # Keep the LayerNorm2d + Flatten and replace only the final Linear.
    in_features = model.classifier[2].in_features
    _unfreeze_last_feature_stages(model, num_stages=2)
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_model(
    model_name: str,
    num_classes: int,
    use_pretrained_weights: bool = True,
) -> nn.Module:
    if model_name == "custom_cnn":
        return CustomCNN(num_classes)
    if model_name == "resnet50_feature_extraction":
        return build_resnet50_feature_extractor(
            num_classes,
            use_pretrained_weights=use_pretrained_weights,
        )
    if model_name == "resnet50_fine_tuning":
        return build_resnet50_finetune(
            num_classes,
            use_pretrained_weights=use_pretrained_weights,
        )
    if model_name == "efficientnet_v2_s":
        return build_efficientnet_v2_s_finetune(
            num_classes,
            use_pretrained_weights=use_pretrained_weights,
        )
    if model_name == "convnext_tiny":
        return build_convnext_tiny_finetune(
            num_classes,
            use_pretrained_weights=use_pretrained_weights,
        )
    raise ValueError(f"Unknown model name: {model_name}")


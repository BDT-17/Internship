"""Print the CustomCNN architecture summary used in the Phase 2 notebook."""

from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    """Custom CNN trained from scratch for 76-class plant classification."""

    def __init__(self, num_classes: int = 76) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def main() -> None:
    model = CustomCNN(num_classes=76)
    rows = [
        ("Input", "Image tensor", "3 x 224 x 224", 0),
        ("Block 1", "Conv2D 3->64 + ReLU + MaxPool", "64 x 112 x 112", 1_792),
        ("Block 2", "Conv2D 64->128 + ReLU + MaxPool", "128 x 56 x 56", 73_856),
        ("Block 3", "Conv2D 128->256 + ReLU + MaxPool", "256 x 28 x 28", 295_168),
        ("Block 4", "Conv2D 256->512 + ReLU + MaxPool", "512 x 14 x 14", 1_180_160),
        ("Pooling", "AdaptiveAvgPool2D", "512 x 1 x 1", 0),
        ("Classifier", "Linear 512->256 + ReLU + Dropout + Linear 256->76", "76", 150_860),
    ]

    print("CustomCNN architecture summary")
    print("-" * 88)
    print(f"{'Stage':<12} {'Operation':<55} {'Output':<16} {'Params':>10}")
    print("-" * 88)
    for stage, operation, output, params in rows:
        print(f"{stage:<12} {operation:<55} {output:<16} {params:>10,}")
    print("-" * 88)
    print(f"{'Total parameters':<85} {count_parameters(model):>10,}")
    print(f"{'Trainable parameters':<85} {count_trainable_parameters(model):>10,}")


if __name__ == "__main__":
    main()

"""Loss functions for imbalanced classification.

Provides focal loss (Lin et al., ICCV 2017) and class-balanced weighting via
the effective-number-of-samples scheme (Cui et al., CVPR 2019), both of which
target macro-F1 on the long-tailed 76-class plant dataset. A ``build_criterion``
factory wires these together with optional label smoothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights and label smoothing."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def class_balanced_weights(class_counts: Sequence[int], beta: float = 0.9999) -> torch.Tensor:
    """Effective-number-of-samples class weights (Cui et al., CVPR 2019)."""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    effective_num = 1.0 - torch.pow(beta, counts)
    weights = (1.0 - beta) / effective_num
    return weights / weights.sum() * len(counts)


def build_criterion(
    loss_name: str,
    label_smoothing: float = 0.0,
    class_counts: Sequence[int] | None = None,
    focal_gamma: float = 2.0,
    cb_beta: float = 0.9999,
    device: torch.device | None = None,
) -> nn.Module:
    """Create a loss module.

    ``loss_name`` is one of ``ce``, ``focal``, ``class_balanced``, ``cb_focal``.
    Class-balanced variants require ``class_counts`` (train-split per-class counts).
    """
    weight = None
    if loss_name in ("class_balanced", "cb_focal"):
        if class_counts is None:
            raise ValueError(f"loss '{loss_name}' requires class_counts")
        weight = class_balanced_weights(class_counts, cb_beta)
        if device is not None:
            weight = weight.to(device)

    if loss_name in ("focal", "cb_focal"):
        return FocalLoss(gamma=focal_gamma, weight=weight, label_smoothing=label_smoothing)
    if loss_name in ("ce", "class_balanced"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss: {loss_name}")

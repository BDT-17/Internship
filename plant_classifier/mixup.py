"""Mixup (Zhang et al., ICLR 2018) and CutMix (Yun et al., ICCV 2019).

Both are strong regularizers for CNNs trained from scratch. ``MixupCutmix`` is
applied to a training batch and returns the (possibly mixed) images together
with the two label sets and the mixing coefficient, so the loss can be computed
as a convex combination via ``mixup_criterion``.
"""

from __future__ import annotations

import numpy as np
import torch


class MixupCutmix:
    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        prob: float = 0.5,
    ) -> None:
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    @property
    def enabled(self) -> bool:
        return self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0

    def __call__(
        self, images: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if not self.enabled or np.random.rand() > self.prob:
            return images, target, target, 1.0

        use_cutmix = self.cutmix_alpha > 0.0 and (
            self.mixup_alpha <= 0.0 or np.random.rand() < 0.5
        )
        index = torch.randperm(images.size(0), device=images.device)
        target_b = target[index]

        if use_cutmix:
            lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
            height, width = images.shape[2], images.shape[3]
            ratio = np.sqrt(1.0 - lam)
            cut_w, cut_h = int(width * ratio), int(height * ratio)
            cx, cy = np.random.randint(width), np.random.randint(height)
            x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, width)
            y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, height)
            images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
        else:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            images = lam * images + (1.0 - lam) * images[index]

        return images, target, target_b, lam


def mixup_criterion(criterion, logits, target_a, target_b, lam: float) -> torch.Tensor:
    return lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)

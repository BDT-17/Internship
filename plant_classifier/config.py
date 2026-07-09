"""Configuration objects for training and inference scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAMES = (
    "custom_cnn",
    "resnet50_feature_extraction",
    "resnet50_fine_tuning",
)


@dataclass(slots=True)
class TrainConfig:
    dataset_dir: Path
    output_dir: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 50
    learning_rate_cnn: float = 1e-3
    learning_rate_cnn_v2: float = 1e-3
    learning_rate_feature_extractor: float = 1e-3
    learning_rate_finetune: float = 1e-4
    learning_rate_efficientnet_v2_s: float = 1e-4
    learning_rate_convnext_tiny: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 7
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    use_amp: bool = True
    save_every_epoch: bool = True
    use_pretrained_weights: bool = True
    models: tuple[str, ...] = DEFAULT_MODEL_NAMES
    # Tier-1 training improvements (defaults keep the original behaviour).
    optimizer: str = "adam"  # adam | adamw
    lr_scheduler: str = "none"  # none | cosine | cosine_warmup
    warmup_epochs: int = 3
    label_smoothing: float = 0.0
    loss: str = "ce"  # ce | focal | class_balanced | cb_focal
    focal_gamma: float = 2.0
    cb_beta: float = 0.9999
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    mixup_prob: float = 0.5

    def validate(self) -> None:
        split_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(split_sum - 1.0) > 1e-8:
            raise ValueError(f"Split ratios must sum to 1.0, got {split_sum}.")
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_dir}")

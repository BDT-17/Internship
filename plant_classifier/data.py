"""Dataset discovery, stratified splitting, and PyTorch data loaders."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageListDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Path, int]], transform=None) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def discover_samples(dataset_dir: Path) -> tuple[list[str], list[tuple[Path, int]], dict[str, int]]:
    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    class_names = [path.name for path in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: list[tuple[Path, int]] = []
    class_counts: dict[str, int] = {}
    for class_dir in class_dirs:
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        )
        class_counts[class_dir.name] = len(images)
        samples.extend((img_path, class_to_idx[class_dir.name]) for img_path in images)

    if not samples:
        raise ValueError(f"No images found in dataset folder: {dataset_dir}")

    return class_names, samples, class_counts


def stratified_split(
    samples: Sequence[tuple[Path, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = {}
    for img_path, label in samples:
        by_class.setdefault(label, []).append((img_path, label))

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []

    for items in by_class.values():
        items = items.copy()
        rng.shuffle(items)
        n_items = len(items)
        train_end = int(n_items * train_ratio)
        val_end = int(n_items * (train_ratio + val_ratio))

        if train_end < 1:
            train_end = 1
        if val_end <= train_end:
            val_end = min(train_end + 1, n_items)
        if val_end >= n_items:
            val_end = n_items - 1

        train_samples.extend(items[:train_end])
        val_samples.extend(items[train_end:val_end])
        test_samples.extend(items[val_end:])

    return train_samples, val_samples, test_samples


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(
    train_samples: Sequence[tuple[Path, int]],
    val_samples: Sequence[tuple[Path, int]],
    test_samples: Sequence[tuple[Path, int]],
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = ImageListDataset(train_samples, transform=train_transform)
    val_dataset = ImageListDataset(val_samples, transform=eval_transform)
    test_dataset = ImageListDataset(test_samples, transform=eval_transform)

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    )


from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENETTE_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def create_transforms(train: bool, size: int = 224) -> transforms.Compose:
    """Create transform pipeline with efficient resize."""
    resize_size = int(size * 1.143)  # Slightly larger for random crops

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENETTE_STATS, inplace=False),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(resize_size, antialias=True),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENETTE_STATS, inplace=False),
            ]
        )


def get_dataloaders(cfg) -> Tuple[DataLoader, DataLoader]:
    """Get dataloaders with optimized settings."""
    # Create transforms
    train_transform = create_transforms(True, cfg.patch_size * 14)  # 16 * 14 = 224
    val_transform = create_transforms(False, cfg.patch_size * 14)

    # Load dataset
    full_dataset = datasets.ImageFolder(root=cfg.data.root, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=cfg.data.root, transform=val_transform)

    # Split datasets
    train_size = int(len(full_dataset) * cfg.data.train_val_split)
    val_size = len(full_dataset) - train_size

    if cfg.seed is not None:
        generator = torch.Generator().manual_seed(cfg.seed)
    else:
        generator = None

    train_dataset, _ = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    _, val_dataset = random_split(
        val_dataset, [train_size, val_size], generator=generator
    )

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=True if cfg.data.num_workers > 0 else False,
        prefetch_factor=2 if cfg.data.num_workers > 0 else None,
    )

    return train_loader, val_loader

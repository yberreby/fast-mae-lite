import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import logging

logger = logging.getLogger(__name__)

IMAGENETTE_STATS = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def create_transform(train: bool, size: int = 224) -> transforms.Compose:
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


class TensorCycler:
    """Provides DataLoader-like iteration over a fixed tensor."""

    def __init__(self, images: torch.Tensor, batch_size: int, device: str = "cuda"):
        self.images = images
        self.batch_size = batch_size
        self.device = device
        self.length = len(images)
        assert (
            self.batch_size <= self.length
        ), f"Batch size ({batch_size}) cannot be larger than dataset size ({self.length})"

    def __iter__(self):
        while True:  # Infinite iteration
            idx = torch.randperm(self.length, device=self.device)[: self.batch_size]
            yield self.images[idx], torch.zeros(self.batch_size, device=self.device)

    def __len__(self):
        return self.length // self.batch_size


def get_preprocessed_tensors(dataset, indices, device="cuda") -> torch.Tensor:
    """Process specific indices through dataset and store results."""
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=len(indices),  # Process all at once
        num_workers=1,
        shuffle=False,
    )
    # Single batch processing
    images, _ = next(iter(loader))
    return images.to(device)


def get_dataloaders(cfg):
    """Get either normal dataloaders or tensor cyclers."""
    if cfg.data.tiny_dataset_size is None:
        # Normal training path
        train_transform = create_transform(True, cfg.patch_size * 14)
        val_transform = create_transform(False, cfg.patch_size * 14)

        train_dataset = datasets.ImageFolder(cfg.data.root, transform=train_transform)
        val_dataset = datasets.ImageFolder(cfg.data.root, transform=val_transform)

        logger.info(
            f"Creating regular dataloaders with {len(train_dataset)} total images"
        )

        return map(
            lambda ds: DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                persistent_workers=cfg.data.num_workers > 0,
                prefetch_factor=2 if cfg.data.num_workers > 0 else None,
            ),
            (train_dataset, val_dataset),
        )

    # Tiny dataset path
    N = cfg.data.tiny_dataset_size
    logger.info(f"Creating tiny dataset with {N} images each for train/val")

    # Create transforms and base datasets
    transforms_list = [
        create_transform(is_train, cfg.patch_size * 14) for is_train in (True, False)
    ]

    datasets_list = list(
        map(lambda t: datasets.ImageFolder(cfg.data.root, transform=t), transforms_list)
    )

    # Get separate indices for train and val
    all_indices = torch.randperm(len(datasets_list[0]))
    indices_list = [all_indices[:N].tolist(), all_indices[N : 2 * N].tolist()]

    # Process images and create cyclers
    return map(
        lambda ds, idx: TensorCycler(
            get_preprocessed_tensors(ds, idx, cfg.device), cfg.batch_size, cfg.device
        ),
        datasets_list,
        indices_list,
    )

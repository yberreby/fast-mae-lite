import torch
from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Optional
from pathlib import Path


@dataclass
class DataConfig:
    root: str = str(Path.home() / "datasets/imagenette")
    tiny_dataset_size: Optional[int] = None  # If set, use N images each for train/val
    train_val_split: float = 0.9
    num_workers: int = field(default_factory=lambda: min(mp.cpu_count(), 4))
    pin_memory: bool = False


@dataclass
class TrainingConfig:
    # Core behavior flags
    peak_performance_test: bool = False  # New flag for peak performance testing
    compile: bool = True
    amp: bool = True
    profiler: bool = False

    # Model architecture
    patch_size: int = 16
    embed_dim: int = 192  # Tiny variant
    decoder_embed_dim: int = 96

    # Training parameters
    batch_size: int = 256
    total_samples: int = 1_000_000
    grad_clip: float = 1.0
    mask_ratio: float = 0.75

    # Optimization parameters
    lr: float = 1.5e-4
    lr_layer_decay: float = 1.0  # layerwise LR decay
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999

    # Random seed
    seed: Optional[int] = None

    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Logging and checkpointing
    log_dir: str = "runs"
    ckpt_dir: str = "checkpoints"
    samples_per_viz: int = 1000
    samples_per_val: int = 10000
    samples_per_ckpt: int = 50000

    # Pretrained model
    pretrained_path: Optional[str] = "ckpt/mae_tiny_400e.pth.tar"

    def __post_init__(self):
        assert self.samples_per_ckpt >= self.samples_per_val >= self.samples_per_viz
        assert 0 < self.mask_ratio < 1

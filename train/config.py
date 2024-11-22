from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Optional

from omegaconf import MISSING


@dataclass
class DataConfig:
    root: str = MISSING
    train_val_split: float = 0.9
    num_workers: int = field(default_factory=lambda: min(mp.cpu_count(), 16))
    pin_memory: bool = True


@dataclass
class BaseTrainConfig:
    # Model
    patch_size: int = 16
    embed_dim: int = 192  # Tiny variant
    decoder_embed_dim: int = 96
    compile: bool = True

    # Training
    batch_size: int = 256
    total_samples: int = 1_000_000  # ~400 epochs on ImageNette
    amp: bool = True
    grad_clip: float = 1.0
    mask_ratio: float = 0.75
    lr: float = 1.5e-4
    seed: Optional[int] = None
    profiler: bool = False

    # Data
    data: DataConfig = field(default_factory=DataConfig)

    # Logging
    log_dir: str = "runs"
    ckpt_dir: str = "checkpoints"
    samples_per_viz: int = 1000
    samples_per_val: int = 10000  # ~4 epochs on ImageNette
    samples_per_ckpt: int = 50000  # ~20 epochs

    pretrained_path: Optional[str] = None

    def __post_init__(self):
        assert self.samples_per_ckpt >= self.samples_per_val >= self.samples_per_viz
        assert 0 < self.mask_ratio < 1

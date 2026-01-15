import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from fml.model import MAEConfig, MAELite
from train.config import TrainingConfig
from train.data import get_dataloaders
from train.training import MAETrainer
from train.opt import create_optimizer_and_scheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="base_schema", node=TrainingConfig)


@dataclass
class TrainingResources:
    """Container for training resources to simplify resource management."""

    model: MAELite
    train_loader: Optional[DataLoader]
    val_loader: Optional[DataLoader]
    writer: SummaryWriter
    profiler: Optional[torch.profiler.profile] = None

    def cleanup(self):
        """Cleanup resources to prevent memory leaks."""
        if self.profiler:
            self.profiler.stop()
        self.writer.close()


def create_model(cfg: TrainingConfig, device: torch.device) -> MAELite:
    """Create and configure the model."""
    t0 = time.perf_counter()

    model_cfg = MAEConfig.tiny(
        norm_pix_loss=False,
        masked_loss=False,
    )
    model = MAELite(model_cfg, device)

    if cfg.pretrained_path:
        path = to_absolute_path(cfg.pretrained_path)
        if Path(path).exists():
            logger.info(f"Loading pretrained weights from {path}")
            model.load_legacy_weights(path)
        else:
            logger.warning(
                f"Pretrained weights path {path} not found, starting from scratch"
            )

    if cfg.compile:
        t1 = time.perf_counter()
        logger.info("Compiling model...")
        model = torch.compile(model)
        logger.info(f"Compilation took {time.perf_counter() - t1:.2f}s")

    logger.info(f"Model creation took {time.perf_counter() - t0:.2f}s")
    return model


def setup_profiler(log_dir: str) -> torch.profiler.profile:
    """Configure the PyTorch profiler."""
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    )


def setup_paths(cfg: TrainingConfig) -> None:
    """Setup and validate logging and checkpoint directories."""
    for path_attr in ["log_dir", "ckpt_dir"]:
        path = to_absolute_path(getattr(cfg, path_attr))
        Path(path).mkdir(parents=True, exist_ok=True)
        setattr(cfg, path_attr, path)


def initialize_training(
    cfg: TrainingConfig,
    device: torch.device,
) -> TrainingResources:
    """Initialize all training resources."""
    t0 = time.perf_counter()

    # Create model
    model = create_model(cfg, device)

    # Setup data loading
    if cfg.peak_performance_test:
        logger.info("🚀 Running in peak performance test mode - bypassing data loading")
        train_loader = val_loader = None
    else:
        logger.info("Setting up data loaders...")
        train_loader, val_loader = get_dataloaders(cfg)
        logger.info(
            f"Dataset sizes - Train: {len(train_loader)}, Val: {len(val_loader)} batches"
        )

    # Setup logging and profiling
    writer = SummaryWriter(cfg.log_dir)
    profiler = setup_profiler(cfg.log_dir) if cfg.profiler else None

    logger.info(f"Initialization took {time.perf_counter() - t0:.2f}s")

    return TrainingResources(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        profiler=profiler,
    )


@hydra.main(version_base="1.2", config_path="config", config_name="base")
def main(cfg: TrainingConfig) -> None:
    """Main training entry point."""
    t_start = time.perf_counter()

    # Set random seeds for reproducibility
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        logger.info(f"Set random seed to {cfg.seed}")

    # Setup directories
    setup_paths(cfg)

    # Select device
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"CUDA capabilities: {torch.cuda.get_device_capability()}")

    # Initialize training resources
    resources = initialize_training(cfg, device)
    logger.info(f"Setup completed in {time.perf_counter() - t_start:.2f}s")

    try:
        # Start profiler if enabled
        if resources.profiler:
            resources.profiler.start()

        # Create optimizer and scheduler
        total_steps = cfg.total_samples // cfg.batch_size
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        logger.info(
            f"Preparing for {total_steps} steps"
            f"{', with ' + str(warmup_steps) + ' warmup steps' if warmup_steps > 0 else ''}"
        )

        optimizer, scheduler = create_optimizer_and_scheduler(
            logger, resources.model, cfg, warmup_steps
        )

        # Create and run trainer
        trainer = MAETrainer(
            model=resources.model,
            train_loader=resources.train_loader,
            val_loader=resources.val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            device=device,
            logger=logger,
            profiler=resources.profiler,
        )

        if cfg.peak_performance_test:
            trainer.train_peak_performance()
        else:
            trainer.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Cleanup
        resources.cleanup()

    logger.info(f"Run completed in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()

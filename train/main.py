import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, cast

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader

from fml.model import MAEConfig, MAELite
from train.config import BaseTrainConfig
from train.data import get_dataloaders
from train.training import MAETrainer  # Using our refactored trainer
from train.opt import create_optimizer_and_scheduler

torch.set_float32_matmul_precision("high")
cs = ConfigStore.instance()
cs.store(name="base_train_config", node=BaseTrainConfig)


@dataclass
class TrainingResources:
    model: MAELite
    train_loader: DataLoader
    val_loader: DataLoader
    writer: SummaryWriter
    profiler: Optional[torch.profiler.profile] = None


def setup_paths(cfg: BaseTrainConfig) -> None:
    for path_attr in ["log_dir", "ckpt_dir"]:
        path = to_absolute_path(getattr(cfg, path_attr.split(".")[-1]))
        Path(path).mkdir(parents=True, exist_ok=True)
        setattr(cfg, path_attr.split(".")[-1], path)


def create_model(
    cfg: BaseTrainConfig, device: torch.device, logger: logging.Logger
) -> MAELite:
    t0 = time.perf_counter()

    model_cfg = MAEConfig(
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        decoder_embed_dim=cfg.decoder_embed_dim,
        norm_pix_loss=False,
        masked_loss=False,
    )
    model = MAELite(model_cfg, device)

    if cfg.pretrained_path:
        model.load_legacy_weights(cfg.pretrained_path)

    if cfg.compile:
        t1 = time.perf_counter()
        model = torch.compile(model)
        logger.info(f"Compilation: {time.perf_counter() - t1:.2f}s")

    logger.info(f"Model creation: {time.perf_counter() - t0:.2f}s")
    return cast(MAELite, model)


def setup_profiler(log_dir: str) -> torch.profiler.profile:
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        profile_memory=True,
        with_stack=True,
        record_shapes=True,
    )


def initialize_training(
    cfg: BaseTrainConfig,
    device: torch.device,
    logger: logging.Logger,
) -> TrainingResources:
    model = create_model(cfg, device, logger)

    t0 = time.perf_counter()
    train_loader, val_loader = get_dataloaders(cfg)
    logger.info(
        f"Dataset sizes (in batches) - Train: {len(train_loader)}, Val: {len(val_loader)}"
    )
    logger.info(f"Dataloader setup: {time.perf_counter() - t0:.2f}s")

    writer = SummaryWriter(cfg.log_dir)
    profiler = setup_profiler(cfg.log_dir) if cfg.profiler else None

    return TrainingResources(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        profiler=profiler,
    )


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: BaseTrainConfig) -> None:
    t_start = time.perf_counter()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Initializing training pipeline...")

    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    setup_paths(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resources = initialize_training(cfg, device, logger)
    logger.info(f"Setup complete in {time.perf_counter() - t_start:.2f}s")

    if resources.profiler:
        resources.profiler.start()

    # Create optimizer and scheduler
    total_steps = cfg.total_samples // cfg.batch_size
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    logger.info(
        f"Preparing to train for {total_steps} steps, warming up over {warmup_steps}"
    )
    optimizer, scheduler = create_optimizer_and_scheduler(
        logger, resources.model, cfg, total_steps
    )

    # Pass scheduler to trainer
    trainer = MAETrainer(
        model=resources.model,
        train_loader=resources.train_loader,
        val_loader=resources.val_loader,
        optimizer=optimizer,
        scheduler=scheduler,  # Add this
        cfg=cfg,
        device=device,
        logger=logger,
        profiler=resources.profiler,
    )

    trainer.train()

    if resources.profiler:
        resources.profiler.stop()
    resources.writer.close()


if __name__ == "__main__":
    main()

import math
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from torch.amp.autocast_mode import autocast

from fml.model import MAELite
from fml.utils import denorm
from .data import TensorCycler


@dataclass
class StepOutput:
    loss: float
    reconstructed: Optional[torch.Tensor]
    mask: Optional[torch.Tensor]
    latent: Optional[torch.Tensor]


class MAETrainer:
    def __init__(
        self,
        model: MAELite,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        cfg,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[logging.Logger] = None,
        profiler=None,
        max_vis_images: Optional[int] = 4,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = cfg
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.profiler = profiler
        self.max_vis_images = max_vis_images

        self.writer = SummaryWriter()
        self.scaler = GradScaler(enabled=cfg.amp)
        self.imgs_seen = 0
        self.epoch = 0
        self.samples_since_val = 0
        self.samples_since_viz = 0
        self.best_val_loss = float("inf")

        self.val_vis_images = None
        self.val_vis_batch = None

    @property
    def step_idx(self) -> int:
        return self.imgs_seen // self.cfg.batch_size

    def save_checkpoint(self, path: Path) -> None:
        ckpt = {
            "imgs_seen": self.imgs_seen,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
        }
        tmp_path = path.with_suffix(".tmp")
        torch.save(ckpt, tmp_path)
        tmp_path.rename(path)

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path)
        self.imgs_seen = ckpt["imgs_seen"]
        self.epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler and "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])

    def training_step(self, images: torch.Tensor) -> StepOutput:
        images = images.to(self.device, non_blocking=True)

        with autocast("cuda", enabled=self.cfg.amp):
            loss, pred, mask, latent = self.model(
                images, mask_ratio=self.cfg.mask_ratio
            )
            reconstructed = self.model.unpatchify(pred)

        self.scaler.scale(loss).backward()

        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler:
            self.scheduler.step()  # Step scheduler after optimizer
        self.optimizer.zero_grad(set_to_none=True)

        return StepOutput(
            loss=loss.item(), reconstructed=reconstructed, mask=mask, latent=latent
        )

    def log_visuals(
        self, prefix: str, originals: torch.Tensor, out: StepOutput
    ) -> None:
        assert out.reconstructed is not None
        assert out.mask is not None
        # print("Logging visuals...", end='', flush=True)
        if self.max_vis_images:
            originals = originals[: self.max_vis_images]
            reconstructed = out.reconstructed[: self.max_vis_images]
            mask = out.mask[: self.max_vis_images]
        else:
            reconstructed = out.reconstructed
            mask = out.mask

        self.writer.add_images(f"{prefix}/original", denorm(originals), self.imgs_seen)
        self.writer.add_images(
            f"{prefix}/reconstructed",
            torch.clamp(denorm(reconstructed), 0, 1),
            self.imgs_seen,
        )

        # Reshape mask from (B, N) patch space to (B, 1, H, W) image space
        B, N = mask.shape
        h = w = int(math.sqrt(N))
        mask_vis = mask.reshape(B, h, w).unsqueeze(1)
        mask_vis = torch.nn.functional.interpolate(
            mask_vis.float(), scale_factor=self.cfg.patch_size, mode="nearest"
        )
        self.writer.add_images(
            f"{prefix}/mask", mask_vis, self.imgs_seen, dataformats="NCHW"
        )
        # print(" done")

    def log_metrics(self, out: StepOutput, prefix: str = "train") -> None:
        metrics = {
            f"{prefix}/loss": out.loss,
            f"{prefix}/epoch": self.epoch,
        }

        # Log learning rates for each parameter group
        for i, group in enumerate(self.optimizer.param_groups):
            metrics[f"{prefix}/lr_group_{i}"] = group["lr"]

        if prefix == "train":
            metrics["train/memory_gb"] = torch.cuda.memory_allocated() / 1e9

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, self.imgs_seen)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        # Get validation iterator
        if isinstance(self.val_loader, TensorCycler):
            num_val_batches = 50
            validation_iter = (
                next(iter(self.val_loader)) for _ in range(num_val_batches)
            )

            # For TensorCycler, reuse same images but allow random masks
            if self.val_vis_images is None:
                # Just take the first N images from our fixed tensor
                self.val_vis_images = self.val_loader.images[
                    : min(4, len(self.val_loader.images))
                ]
        else:
            validation_iter = self.val_loader
            num_val_batches = len(self.val_loader)

            # For regular DataLoader, store first batch we see for consistent visualization
            if self.val_vis_images is None:
                images, _ = next(iter(self.val_loader))
                self.val_vis_images = images[: min(4, len(images))]

        # Run validation
        for batch_idx, (images, _) in enumerate(
            tqdm(validation_iter, desc="Validating", total=num_val_batches, leave=False)
        ):
            out = self.validation_step(images)

            # Log first batch visualization
            if batch_idx == 0:
                # For vis, run a fresh forward pass on our stored images
                # This preserves random masking while keeping images consistent
                vis_out = self.validation_step(self.val_vis_images)
                self.log_visuals("val", self.val_vis_images, vis_out)

            batch_size = images.shape[0]
            total_loss += out.loss * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        self.log_metrics(StepOutput(avg_loss, None, None, None), prefix="val")
        return avg_loss

    @torch.no_grad()
    def validation_step(self, images: torch.Tensor) -> StepOutput:
        """Run validation step with no seed manipulation."""
        self.model.eval()
        images = images.to(self.device, non_blocking=True)

        with autocast("cuda", enabled=self.cfg.amp):
            loss, pred, mask, latent = self.model(
                images, mask_ratio=self.cfg.mask_ratio
            )
            reconstructed = self.model.unpatchify(pred)

        return StepOutput(
            loss=loss.item(), reconstructed=reconstructed, mask=mask, latent=latent
        )

    def train(self) -> None:
        self.model.train()
        val_loss = self.validate()
        self.logger.info(f"Initial validation loss: {val_loss:.4f}")

        pbar = tqdm(total=self.cfg.total_samples, desc="Training", unit="img")
        t_start = time.perf_counter()

        while self.imgs_seen < self.cfg.total_samples:
            self.epoch += 1
            t_epoch = time.perf_counter()

            for images, _ in self.train_loader:
                batch_size = images.shape[0]
                out = self.training_step(images)
                self.log_metrics(out)
                if self.scheduler:
                    self.scheduler.step()

                # Update counters and progress
                self.imgs_seen += batch_size
                self.samples_since_val += batch_size
                self.samples_since_viz += batch_size
                pbar.update(batch_size)
                pbar.set_postfix(
                    {
                        "epoch": self.epoch,
                        "step": f"{self.step_idx}",
                        "loss": f"{out.loss:.4f}",
                        "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    }
                )

                # Periodic logging
                if self.samples_since_viz >= self.cfg.samples_per_viz:
                    self.log_visuals("train", images, out)
                    self.samples_since_viz = 0

                # Periodic validation
                if self.samples_since_val >= self.cfg.samples_per_val:
                    val_loss = self.validate()
                    self.logger.info(
                        f"Step {self.step_idx} (epoch {self.epoch}): "
                        f"train_loss={out.loss:.4f}, val_loss={val_loss:.4f}"
                    )
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(Path(self.cfg.ckpt_dir) / "best_model.pt")
                    self.samples_since_val = 0
                    self.model.train()

                # Regular checkpoints
                if self.imgs_seen % self.cfg.samples_per_ckpt == 0:
                    self.save_checkpoint(
                        Path(self.cfg.ckpt_dir) / f"checkpoint_{self.imgs_seen:08d}.pt"
                    )

                if self.imgs_seen >= self.cfg.total_samples:
                    break

            self.writer.add_scalar(
                "train/epoch_time", time.perf_counter() - t_epoch, self.epoch
            )

        pbar.close()
        t_total = time.perf_counter() - t_start
        self.logger.info(f"Training completed in {t_total:.2f}s")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(
            f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
        )

    def train_peak_performance(self):
        """Train with constant tensor to measure peak performance."""
        self.model.train()

        # Create fixed training tensor
        B = self.cfg.batch_size
        H = W = self.cfg.patch_size * 14  # 224 for patch_size=16
        images = torch.randn(B, 3, H, W, device=self.device)

        # Setup timing
        t_start = time.perf_counter()

        # Core training loop
        total_steps = self.cfg.total_samples // B
        pbar = tqdm(total=self.cfg.total_samples, desc="Peak Training", unit="img")

        for step in range(total_steps):
            with autocast("cuda", enabled=self.cfg.amp):
                loss, pred, mask, latent = self.model(
                    images, mask_ratio=self.cfg.mask_ratio
                )

                self.scaler.scale(loss).backward()

                if self.cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # Update progress
            self.imgs_seen += B
            pbar.update(B)

        torch.cuda.synchronize()
        wall_time = time.perf_counter() - t_start

        self.logger.info("Peak Performance Results:")
        self.logger.info(f"Total steps: {total_steps}")
        self.logger.info(f"Wall time: {wall_time:.2f}s")
        self.logger.info(f"Steps per second: {total_steps / wall_time:.1f}")
        self.logger.info(f"Images per second: {(total_steps * B) / wall_time:.1f}")
        self.logger.info(
            f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
        )

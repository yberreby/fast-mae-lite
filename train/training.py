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

    @property
    def step_idx(self) -> int:
        return self.imgs_seen // self.cfg.batch_size

    def save_checkpoint(self, path: Path) -> None:
        self.logger.info(f"Saving checkpoint to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "imgs_seen": self.imgs_seen,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        if self.scheduler:
            ckpt["scheduler"] = self.scheduler.state_dict()

        tmp_path = path.with_suffix(".tmp")
        torch.save(ckpt, tmp_path)
        tmp_path.rename(path)

    @torch.no_grad()
    def validation_step(self, images: torch.Tensor) -> StepOutput:
        self.model.eval()
        torch.manual_seed(0)  # Consistent masking for validation
        images = images.to(self.device, non_blocking=True)

        with autocast("cuda", enabled=self.cfg.amp):
            loss, pred, mask, latent = self.model(
                images, mask_ratio=self.cfg.mask_ratio
            )
            reconstructed = self.model.unpatchify(pred)

        return StepOutput(
            loss=loss.item(), reconstructed=reconstructed, mask=mask, latent=latent
        )

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
        self.optimizer.zero_grad(set_to_none=True)
        if self.profiler:
            self.profiler.step()

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
            f"{prefix}/lr": self.optimizer.param_groups[0]["lr"],
        }
        if prefix == "train":
            metrics["train/memory_gb"] = torch.cuda.memory_allocated() / 1e9

        for name, value in metrics.items():
            self.writer.add_scalar(name, value, self.imgs_seen)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        vis_images = next(iter(self.val_loader))[0]

        for batch_idx, (images, _) in enumerate(
            tqdm(self.val_loader, desc="Validating", leave=False)
        ):
            out = self.validation_step(images)

            if batch_idx == 0:
                self.log_visuals("val", vis_images, out)

            batch_size = images.shape[0]
            total_loss += out.loss * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        self.log_metrics(StepOutput(avg_loss, None, None, None), prefix="val")
        return avg_loss

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

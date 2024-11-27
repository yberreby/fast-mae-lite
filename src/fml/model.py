from pathlib import Path
from safetensors.torch import save_file
import json
from typing import Optional, Tuple

import torch
import torch.nn as nn
import re
from .encoder import MAEEncoder
from .decoder import MAEDecoder
from .config import MAEConfig


class MAELite(nn.Module):
    """MAE model with ViT backbone."""

    def __init__(self, config: MAEConfig):
        super().__init__()
        self.config = config
        self.encoder = MAEEncoder(config)
        self.decoder = MAEDecoder(config)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches."""
        p = self.config.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images."""
        p = self.config.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # mean loss per patch

        if self.config.masked_loss:
            loss = loss * mask

        loss = loss.sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        ids_shuffle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        latent, mask, ids_restore, ids_shuffle = self.encoder(
            imgs, mask_ratio, ids_shuffle
        )
        pred = self.decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, ids_shuffle

    def save_weights(self, path: str | Path) -> None:
        """Save weights + config in safetensors format."""
        path = Path(path)
        save_file(self.state_dict(), path.with_suffix(".safetensors"))
        json.dump(self.config.__dict__, open(path.with_suffix(".json"), "w"))

    def load_legacy_weights(self, path: str | Path, device: torch.device = 'cuda') -> "MAELite":
        """Load weights from original MAELite repo format, remapping keys to new structure."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state_dict = ckpt.get("model", ckpt)

        # Map legacy state dict keys to our new structure
        key_transforms = {
            r"^patch_embed": "encoder.patch_embed",
            r"^cls_token": "encoder.cls_token",
            r"^pos_embed": "encoder.pos_embed",
            r"^blocks\.": "encoder.blocks.",
            r"^norm\.": "encoder.norm.",
            r"^decoder_embed\.": "decoder.decoder_embed.",
            r"^decoder_pos_embed": "decoder.decoder_pos_embed",
            r"^decoder_blocks\.": "decoder.decoder_blocks.",
            r"^decoder_norm\.": "decoder.decoder_norm.",
            r"^decoder_pred\.": "decoder.decoder_pred.",
            r"^mask_token": "decoder.mask_token",
        }

        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.model.", "")  # Remove legacy prefix
            # Apply our transforms
            for pattern, replacement in key_transforms.items():
                new_k = re.sub(pattern, replacement, new_k)
            new_state_dict[new_k] = v

        self.load_state_dict(new_state_dict)
        return self.to(device)

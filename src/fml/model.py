from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed

from timm.models.vision_transformer import DropPath, Mlp, Attention as BaseAttn
from .utils import *


@dataclass
class MAEConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 192  # Tiny variant
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 96
    decoder_depth: int = 1
    decoder_num_heads: int = 3
    mlp_ratio: float = 4.0
    norm_pix_loss: bool = True
    mask_ratio: float = 0.75  # Default masking ratio
    norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Identity for getting attention maps
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.identity(attn)  # For visualization/distillation
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MAELite(nn.Module):
    """MAE Lite model with ViT backbone."""

    def __init__(self, config: MAEConfig):
        super().__init__()
        self.config = config
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Encoder specifics
        self.patch_embed = PatchEmbed(
            config.img_size, config.patch_size, config.in_channels, config.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    norm_layer=norm_layer,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = norm_layer(config.embed_dim)

        # Decoder specifics
        self.decoder_embed = nn.Linear(
            config.embed_dim, config.decoder_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_embed_dim),
            requires_grad=False,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    config.decoder_embed_dim,
                    config.decoder_num_heads,
                    config.mlp_ratio,
                    norm_layer=norm_layer,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim,
            config.patch_size**2 * config.in_channels,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize other parameters
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Complete forward_encoder implementation
    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        ids_shuffle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor of shape (B, C, H, W)
            mask_ratio: Ratio of patches to mask
            ids_shuffle: Optional tensor of pre-computed shuffle indices

        Returns:
            Tuple of (encoded tokens, mask, restore indices, shuffle indices)
        """
        # Embed patches
        x = self.patch_embed(x)

        # Add positional embeddings without cls token
        x = x + self.pos_embed[:, 1:, :]

        # Masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_shuffle = self.random_masking(
            x, mask_ratio, ids_shuffle
        )

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_shuffle

    # Complete forward_decoder implementation
    def forward_decoder(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: Encoded tokens
            ids_restore: Indices for restoring original token order

        Returns:
            Decoded patches
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.

        Args:
            imgs: Original input images
            pred: Predicted patches
            mask: Binary mask indicating masked patches

        Returns:
            Reconstruction loss
        """
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        ids_shuffle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end forward pass.

        Args:
            imgs: Input images
            mask_ratio: Ratio of patches to mask
            ids_shuffle: Optional pre-computed shuffle indices

        Returns:
            Tuple of (loss, predictions, mask, shuffle indices)
        """
        latent, mask, ids_restore, ids_shuffle = self.forward_encoder(
            imgs, mask_ratio, ids_shuffle
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, ids_shuffle

    # Additional utility methods

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches.

        Args:
            imgs: Input tensor of shape (B, C, H, W)

        Returns:
            Tensor of patches
        """
        p = self.config.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images.

        Args:
            x: Tensor of patches

        Returns:
            Reconstructed images
        """
        p = self.config.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        ids_shuffle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking by per-sample shuffling.

        Args:
            x: Sequence of tokens
            mask_ratio: Ratio of tokens to mask
            ids_shuffle: Optional pre-computed shuffle indices

        Returns:
            Tuple of (masked sequence, mask, restore indices, shuffle indices)
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

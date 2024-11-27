from typing import Optional, Tuple
from torch import nn

# models.py
from .config import MAEConfig
import torch
from .pos import get_2d_sincos_pos_embed
from .block import Block

# TODO eliminate this timm dependency
from timm.models.vision_transformer import PatchEmbed


class MAEEncoder(nn.Module):
    """MAE Encoder with masking."""

    def __init__(self, config: MAEConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(
            config.img_size, config.patch_size, config.in_channels, config.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Embeddings and tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim), requires_grad=False
        )

        # Main encoder blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.embed_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    norm_layer=config.norm_layer,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = config.norm_layer(config.embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize tokens and weights
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        ids_shuffle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform per-sample random masking by per-sample shuffling."""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if ids_shuffle is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        ids_shuffle: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

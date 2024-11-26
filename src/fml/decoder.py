import torch
import torch.nn as nn
from .config import MAEConfig
from .block import Block
from .pos import get_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """MAE Decoder."""

    def __init__(self, config: MAEConfig):
        super().__init__()
        self.config = config

        self.decoder_embed = nn.Linear(
            config.embed_dim, config.decoder_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_embed_dim))

        num_patches = (config.img_size // config.patch_size) ** 2
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
                    norm_layer=config.norm_layer,
                )
                for _ in range(config.decoder_depth)
            ]
        )

        self.decoder_norm = config.norm_layer(config.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim,
            config.patch_size**2 * config.in_channels,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.decoder_pos_embed.shape[1] ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize other weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
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

        # Add pos embed
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

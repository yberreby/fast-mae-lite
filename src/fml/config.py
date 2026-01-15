from dataclasses import dataclass, field
import torch.nn as nn
from typing import Callable
from functools import partial

DECODER_LARGE_DIM = 512  # Fixed decoder dim for base/large/huge variants


@dataclass
class MAEConfig:
    """MAE model configuration with variant support."""

    # Core architecture (varies by variant)
    embed_dim: int
    patch_size: int
    depth: int
    num_heads: int
    decoder_depth: int

    decoder_embed_dim: int | None = None  # Will be set in post_init if None
    decoder_num_heads: int | None = None  # Will be set in post_init if None

    # Stable
    img_size: int = 224
    in_channels: int = 3

    # Standard hyperparameters (constant across variants)
    mlp_ratio: float = 4.0
    norm_pix_loss: bool = True
    masked_loss: bool = True
    mask_ratio: float = 0.75
    norm_layer: Callable = field(
        default_factory=lambda: partial(nn.LayerNorm, eps=1e-6)
    )

    def __post_init__(self):
        # Handle decoder dimensions according to variant rules
        if self.decoder_embed_dim is None:
            self.decoder_embed_dim = (
                DECODER_LARGE_DIM if self.embed_dim >= 768 else self.embed_dim // 2
            )

        if self.decoder_num_heads is None:
            self.decoder_num_heads = self.decoder_embed_dim // 32

    @classmethod
    def tiny(cls, **kwargs) -> "MAEConfig":
        """ViT-Tiny variant."""
        return cls(
            embed_dim=192,
            patch_size=16,
            depth=12,
            num_heads=12,
            decoder_depth=1,
            **kwargs,
        )

    @classmethod
    def small(cls, **kwargs) -> "MAEConfig":
        """ViT-Small variant."""
        return cls(
            embed_dim=384,
            patch_size=16,
            depth=12,
            num_heads=12,
            decoder_depth=1,
            **kwargs,
        )

    @classmethod
    def base(cls, **kwargs) -> "MAEConfig":
        """ViT-Base variant."""
        return cls(
            embed_dim=768,
            patch_size=16,
            depth=12,
            num_heads=12,
            decoder_depth=8,  # First variant with deeper decoder
            **kwargs,
        )

    @classmethod
    def large(cls, **kwargs) -> "MAEConfig":
        """ViT-Large variant."""
        return cls(
            embed_dim=1024,
            patch_size=16,
            depth=24,  # Deeper encoder
            num_heads=16,  # More heads
            decoder_depth=8,
            **kwargs,
        )

    @classmethod
    def huge(cls, **kwargs) -> "MAEConfig":
        """ViT-Huge variant."""
        return cls(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            decoder_depth=8,
            patch_size=14,  # Only variant with different patch size
            **kwargs,
        )

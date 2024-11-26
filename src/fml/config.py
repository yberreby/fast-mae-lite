from dataclasses import dataclass
import torch.nn as nn
from typing import Callable
from functools import partial


@dataclass
class MAEConfig:
    """
    Currently only for tiny variant.
    """

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
    masked_loss: bool = True
    mask_ratio: float = 0.75
    norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6)

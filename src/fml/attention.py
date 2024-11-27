import torch
from torch import nn


class Attention(nn.Module):
    """Drop-in replacement for custom Attention using PyTorch's MultiheadAttention."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        # Change from self.attn to self.mha to avoid nested 'attn' in state dict
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True
        )
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mha(x, x, x, need_weights=False)  # Changed from self.attn to self.mha
        return self.proj_drop(out)

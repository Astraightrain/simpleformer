import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForwardBlock(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        activation = nn.SiLU()
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feedforward = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.Layernorm(dim_h),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(dim_h, dim_in),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x
    ):
        x = self.feedforward(x)
        
        return x

class Attention(nn.Module):

    def __init__(
        self,
        dim_in,
        nheads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        inner_dim = dim_head * nheads

        self.nheads = nheads
        self.scale = dim_head ** -0.5
        
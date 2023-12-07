import math

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_length: int,
    ):
        super().__init__()
        pos_encoding = torch.zeros(max_length, d_model)
        positions = rearrange(
            torch.arange(0, max_length, dtype=torch.float), "m -> m 1"
        )
        division_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model
        )  # 1000^(2i/dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions * division_term)

        # Saving buffer (same as parameter without gradients needed)
        self.pos_encoding = rearrange(pos_encoding, "m d -> m 1 d")
        self.register_buffer("pos_encoding", pos_encoding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x + self.pos_encoding[: x.size(0), :]
        x = self.dropout(x)
        return x

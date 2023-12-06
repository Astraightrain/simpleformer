from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layer: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.silu,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        encoder_layer = VanillaEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        encoder_norm = nn.LayernNorm(d_model, **factory_kwargs)
        self.encoder = VanillaEncoder(
            encoder_layer=encoder_layer,
            encoder_norm=encoder_norm,
            num_layers=num_encoder_layers,
            bias=bias,
        )

    def forward(
        self,
        src: Tensor,
        target: Tensor,
        src_mask: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ):
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(
            target, memory, target_mask=target_mask, memory_mask=memory_mask
        )

        return output

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


class SelfAttention(nn.Module):
    def __init__(self, temperature: float = 0.1, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(0)
        k = rearrange(k, "b n m d -> b n d m")
        attn = torch.matmul(q / self.temperature, k)
        attn = torch.div(attn, np.sqrt(d_k))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    """
    b: batch size
    m: max_seq length
    n: nheads
    h: head_dim
    """

    def __init__(
        self,
        d_model,
        nheads,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        bias=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        assert self.d_model % nheads == 0

        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.k = nn.Linear(d_model, d_model, bias=bias)
        self.v = nn.Linear(d_model, d_model, bias=bias)
        self.selfattn = SelfAttention(temperature=0.1, attn_dropout=attn_dropout)
        self.o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, src_mask=None):
        # MultiheadAttention
        q = rearrange(self.q(x), "b m (n h) -> b n m h", n=self.nheads)
        k = rearrange(self.k(x), "b m (n h) -> b n m h", n=self.nheads)
        v = rearrange(self.k(x), "b m (n h) -> b n m h", n=self.nheads)

        output = self.selfattn(q, k, v, mask=src_mask)
        output = rearrange(output, "b n m h -> b m (n h)", n=self.nheads)
        output = self.o(output)
        output = self.dropout(output)
        return output


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation=nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.feedforward(x)


class ResidualConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, **kwargs):
        return self.layer(x, **kwargs) + x


class PostNormalization(nn.Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self.d_model = d_model
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layernorm(self.layer(x))


class PreNormalization(nn.Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self.d_model = d_model
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        return self.layer(self.layernorm(x), **kwargs)


class VanillaEncoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout, attn_dropout):
        super().__init__()

        self.attn = PostNormalization(
            ResidualConnection(
                MultiHeadAttention(
                    d_model=d_model,
                    nheads=nheads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            ),
            d_model=d_model,
        )

        self.feedforward = PostNormalization(
            ResidualConnection(
                FeedForwardBlock(
                    d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
                )
            ),
            d_model=d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # self attention & residual connection
        output = self.attn(src, src_mask=src_mask)
        output = self.feedforward(output)

        return output


class VanillaDecoderLayer(nn.Module):
    def __init__(self, d_model, nheads, dim_feedforward, dropout, attn_dropout):
        super().__init__()

        self.attn = PostNormalization(
            ResidualConnection(
                MultiHeadAttention(
                    d_model=d_model,
                    nheads=nheads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
            ),
            d_model=d_model,
        )

        self.feedforward = PostNormalization(
            ResidualConnection(
                FeedForwardBlock(
                    d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout
                )
            ),
            d_model=d_model,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, target, target_mask=None):
        # self attention & residual connection
        output = self.attn(target, src_mask=target_mask)
        output = self.feedforward(output)

        return output

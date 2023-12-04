from einops import rearrange

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional, Union, Callable

import numpy as np


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
            device = None,
            dtype = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        encoder_layer = VanillaEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            bias = bias,
        )
        encoder_norm = nn.LayernNorm(d_model, **factory_kwargs)
        self.encoder = VanillaEncoder(
            encoder_layer = encoder_layer,
            encoder_norm = encoder_norm,
            num_layers = num_encoder_layers,
            bias = bias
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
        output = self.decoder(target, memory, target_mask = target_mask, memory_mask = memory_mask)

        return output

    def _reset_parameters(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


class VanillaEncoderLayer(nn.Module):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.0,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.silu,

    ):
        


class SelfAttention(nn.Module):
    
    def __init__(
            self,
            temperature,
            attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(
            self,
            q,
            k,
            v,
            mask = None
    ):  
        d_k = k.size()[-1]
        k = rearrange(k, 'b x y -> b y x')

        attn = torch.matmul(q / self.temperature, k)
        attn = attn / np.sqrt(d_k)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        nheads,
        dropout: float = 0.1,
        bias = True,

    ):
        self.dim_emb = d_model
        self.nheads = nheads
        self.dropout = dropout
        self.head_dim = d_model // nheads
        assert self.head_dim * nheads == self.dim_emb

        self.q = nn.Linear(d_model, d_model, bias = bias)
        self.k = nn.Linear(d_model, d_model, bias = bias)
        self.v = nn.Linear(d_model, d_model, bias = bias)
        self.selfattn = SelfAttention()

    def forward(
            self,
            q,
            k,
            v,
            mask = None
    ):
        batch_size = q.size()[0]
        q, k, v = rearrange(self.q(q), 'b d1 (h d2) -> b -1 h d2'), rearrange(self.q(k), 'b d1 (h d2) -> b -1 h d2'), rearrange(self.q(v), 'b d1 (h d2) -> b -1 h d2')
        output = self.selfattn(q, k, v, mask = mask)
        output = rearrange(output, 'b -1 h d2 -> b d1 (h d2)', b=batch_size, d2=self.nheads)
        return output

class FeedForwardBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: Union[str, Callable[[Tensor], Tensor]] = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.feedforward(x)

class ResidualConnection(nn.Module):
    
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, **kwargs):
        return self.layer(x, **kwargs) + x
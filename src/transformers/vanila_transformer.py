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

class SelfAttention(nn.Module):

    def __init__(
            self,
            temperature: float = 0.1,
            attn_dropout: float = 0.1

    ):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
            self,
            q,
            k,
            v,
            mask = None   
    ):
        d_k = q.size(0)
        k = rearrange(k, 'b n m d -> b n d m')
        attn = torch.matmul(q / self.temperature, k)
        attn = torch.div(attn, np.sqrt(d_k))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim = -1)
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
        bias = True,

    ):
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        self.dropout = nn.Dropout(dropout)
        assert self.head_dim * nheads == self.d_model

        self.q = nn.Linear(d_model, d_model, bias = bias)
        self.k = nn.Linear(d_model, d_model, bias = bias)
        self.v = nn.Linear(d_model, d_model, bias = bias)
        self.selfattn = SelfAttention(
            temperature = 0.1,
            attn_dropout = attn_dropout
        )
        self.o = nn.Linear(d_model, d_model, bias = bias)

    def forward(
            self,
            src,
            src_mask = None
    ):
        # MultiheadAttention
        q = rearrange(self.q(src), 'b m (n h) -> b n m h', n = self.nheads)
        k = rearrange(self.k(src), 'b m (n h) -> b n m h', n = self.nheads)
        v = rearrange(self.k(src), 'b m (n h) -> b n m h', n = self.nheads)

        output = self.selfattn(q, k, v, mask = src_mask)
        output = rearrange(output, 'b n m h -> b m (n h)', n = self.nheads)
        output = self.o(output)
        output = self.dropout(output)
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


class VanillaEncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 nheads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, nheads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src
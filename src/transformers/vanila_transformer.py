from einops import rearrange

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional, Union, Callable


class ScaledDotProductAttention(nn.Module):
    
    def __init__(
            self,
            temperature,
            attn_dropout: float = 0.1
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
        attn = torch.matmul(q / self.temperature, rearrange('b a c -> b c a', k))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim = -1)
        attn = self.dropout(attn)
        ouptut = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        dim_emb,
        nheads,
        dropout: float = 0.1,
        bias = True,

    ):
        self.dim_emb = dim_emb
        self.nheads = nheads
        self.dropout = dropout
        self.head_dim = dim_emb // nheads
        assert self.head_dim * nheads == self.dim_emb

        self.q_dim = dim_emb
        self.k_dim = dim_emb
        self.v_dim = dim_emb
        
        self.attention_weight = Parameter(torch.empty((3 * dim_emb, dim_emb)))
        


class VanillaTransformer(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, context, sentence):
        return self.decode(context, sentence)

    def forward(self, x, sentence):
        
        context = self.encode(x)
        y = self.decode(context, sentence)

        return y
        

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

        encoder_layer = VanillaTransformerEncoderLayer(
            d_model = d_model,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            
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
                
from torch import nn


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

class VanillaHead(nn.Module):
    def __init__(self, d_model: int, activation=nn.SiLU()):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, 1),
            activation,
            nn.Softmax(),
        )

    def forward(self, x):
        return self.layer(x)

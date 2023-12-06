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

from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer


class MLP(torch.nn.Module):
    """
    If d_in \neq width, you must specify it .
    """

    # If d_in \neq width, you must specify it
    def __init__(self, width, depth, dropout=None, d_in=None, output_dims=None):
        super().__init__()
        layers = []
        if d_in is not None:
            layers.append(Linear(d_in, width))
        layers.extend([ResidualLayer(width, dropout=dropout) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out

class MLPOut1(torch.nn.Module):
    def __init__(self, width, depth, dropout=None, output_dims=None):
        super().__init__()
        layers = []
        layers.extend([ResidualLayer(width, dropout=dropout) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out

from pylab import *
import torch
from integrator.layers import Linear,ResidualLayer


class MLP(torch.nn.Module):
    def __init__(self, width, depth, d_in=None, output_dims=None):
        super().__init__()
        layers = []
        if d_in is not None:
            layers.append(Linear(d_in, width))
        layers.extend([ResidualLayer(width) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out


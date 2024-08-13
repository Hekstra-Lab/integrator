from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer


class MLP(torch.nn.Module):
    """
    If d_in \neq width, you must specify it .
    """

    # If d_in \neq width, you must specify it
    def __init__(self, width, depth, dropout=None, d_in=None, output_dims=None):
        """
        Multi-layer perceptron (MLP) module

        Args:
            width (int): Width of the hidden layers
            depth (int): Number of residual layers
            dropout (float, optional): Dropout probability. Defaults to None.
            d_in (int, optional): Input dimension. If not equal to width, it must be specified. Defaults to None.
            output_dims (int, optional): Output dimension. If specified, an additional linear layer is added at the end. Defaults to None.
        """
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

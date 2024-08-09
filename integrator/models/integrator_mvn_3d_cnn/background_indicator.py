import torch
from integrator.layers import Linear

class BackgroundIndicator(torch.nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.linear = Linear(dmodel, 3 * 21 * 21)

    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)
# %%


import torch
from integrator.layers import SimpleCNN

# from integrator.layers import Linear


class BackgroundIndicator(torch.nn.Module):
    def __init__(self, dmodel):
        super().__init__()
        self.cnn = SimpleCNN()
        # self.linear = Linear(dmodel, 3 * 21 * 21)

    def forward(self, x):
        x = self.linear(x)
        bg_profile_params = self.cnn(x)
        bg_profile = torch.sigmoid(bg_profile_params)
        return bg_profile


# %%

import torch
from integrator.layers import SimpleCNN


class BackgroundIndicator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = SimpleCNN()

    def forward(self, representation):
        bg_profile_params = self.cnn(representation)
        bg_profile = torch.sigmoid(bg_profile_params)
        return bg_profile

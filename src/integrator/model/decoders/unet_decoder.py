import torch
import torch.nn as nn
from integrator.model.decoders import BaseDecoder


class UnetDecoder(BaseDecoder):
    def __init__(
        self,
        mc_samples=100,
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.relu = nn.ReLU(inplace=True)

    def forward(self, q_bg, q_p, counts):
        # Sample from variational distributions
        zbg = q_bg.rsample([100, 1323]).permute(2, 0, 1)
        zp = q_p.rsample([self.mc_samples]).permute(1, 0, 2)

        intensity = (self.relu(counts.unsqueeze(1) - zbg) * zp).sum(-1) / zp.pow(2).sum(
            -1
        )
        intensity_mean = intensity.mean(1)
        intensity_variance = intensity.var(1)

        rate = intensity_mean.unsqueeze(1).unsqueeze(1) * zp + zbg

        return rate, intensity_mean, intensity_variance

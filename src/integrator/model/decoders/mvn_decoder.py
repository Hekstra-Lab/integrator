import torch
from integrator.model.decoders import BaseDecoder


class MVNDecoder(BaseDecoder):
    def __init__(self, mc_samples=100):
        super().__init__()
        self.mc_samples = mc_samples

    def forward(self, q_I, q_bg, profile):
        # Sample from variational distributions
        z = q_I.rsample([self.mc_samples]).unsqueeze(-1)
        bg = q_bg.rsample([self.mc_samples]).unsqueeze(-1)

        # Use deterministic profile (no sampling needed)
        # Expand profile to match MC samples dimension
        profile_expanded = profile.unsqueeze(1).expand(-1, self.mc_samples, -1)

        rate = z.permute(1, 0, 2) * profile_expanded + bg.permute(1, 0, 2)
        return rate

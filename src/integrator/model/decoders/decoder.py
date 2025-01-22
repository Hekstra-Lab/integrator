import torch
from integrator.model.decoders import BaseDecoder


class Decoder(BaseDecoder):
    def __init__(
        self,
        mc_samples=100,
    ):
        super().__init__()
        self.mc_samples = mc_samples

    def forward(
        self,
        q_I,
        q_bg,
        q_p,
    ):
        # Sample from variational distributions
        z = q_I.rsample([self.mc_samples]).unsqueeze(-1)
        bg = q_bg.rsample([self.mc_samples]).unsqueeze(-1)
        qp = q_p.rsample([self.mc_samples])

        rate = z.permute(1, 0, 2) * qp.permute(1, 0, 2) + bg.permute(1, 0, 2)

        return rate

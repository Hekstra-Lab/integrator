import torch
import torch.nn as nn
from integrator.model.decoders import BaseDecoder
from integrator.layers import Constraint


class UnetDecoder(BaseDecoder):
    def __init__(
        self,
        mc_samples=100,
        eps=1e-6,
        constraint=Constraint(),
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.relu = nn.ReLU(inplace=True)
        self.constraint = constraint
        self.eps = eps

    def forward(self, q_bg, q_p, counts, mask):
        # Sample from variational distributions
        # zbg = q_bg.rsample([100, 1323]).permute(2, 0, 1)
        zbg = (
            q_bg.rsample([self.mc_samples])
            .unsqueeze(-1)
            .expand(self.mc_samples, counts.shape[0], counts.shape[1])
        ).permute(
            1, 0, 2
        )  # [batch_size, mc_samples, pixels]
        zp = q_p.rsample([self.mc_samples]).permute(1, 0, 2)

        sigma_sq = zbg + counts.unsqueeze(1) + self.eps
        w = 1.0 / sigma_sq

        intensity = (
            self.relu(counts.unsqueeze(1) - zbg) * mask.unsqueeze(1) * zp * w
        ).sum(-1) / ((zp.pow(2) * w * mask.unsqueeze(1)).sum(-1) + self.eps)

        intensity_mean = intensity.mean(1)
        intensity_variance = intensity.var(1)

        rate = (intensity_mean.unsqueeze(1).unsqueeze(1) * zp + zbg) * mask.unsqueeze(
            1
        ) + self.eps

        return rate, intensity_mean, intensity_variance


class UnetDecoder2(BaseDecoder):
    def __init__(
        self,
        mc_samples=100,
        eps=1e-6,
        constraint=Constraint(),
    ):
        super().__init__()
        self.mc_samples = mc_samples
        self.relu = nn.ReLU(inplace=True)
        self.constraint = constraint
        self.eps = eps

    def forward(self, q_bg, q_p, q_I, mask):
        # Sample from variational distributions
        # zbg = q_bg.rsample([100, 1323]).permute(2, 0, 1)
        zbg = (
            q_bg.rsample([self.mc_samples])
            .unsqueeze(-1)
            .expand(self.mc_samples, mask.shape[0], mask.shape[1])
        ).permute(
            1, 0, 2
        )  # [batch_size, mc_samples, pixels]
        zI = (
            q_I.rsample([self.mc_samples])
            .unsqueeze(-1)
            .expand(self.mc_samples, mask.shape[0], mask.shape[1])
        ).permute(1, 0, 2)
        zp = q_p.rsample([self.mc_samples]).permute(1, 0, 2)

        rate = (zI * zp + zbg) * mask.unsqueeze(1) + self.eps

        return rate, q_I.mean, q_I.variance


if __name__ == "__main__":
    # Example usage
    relu = torch.nn.ReLU(inplace=True)
    qbg = torch.distributions.gamma.Gamma(torch.ones(10), torch.ones(10))
    qp = torch.distributions.dirichlet.Dirichlet(torch.ones(10, 1323))
    counts = torch.rand(10, 1323)
    # mask of randomly selected 0 or 1
    masks = torch.randint(0, 2, (10, 1323)).float()

    zbg = (
        qbg.rsample([100]).unsqueeze(-1).expand(100, 10, 1323).permute(1, 0, 2)
    )  # [batch_size, mc_samples, pixels]

    zp = qp.rsample([100]).permute(1, 0, 2)  # [batch_size, mc_samples, pixels]

    sigma_sq = zbg + counts.unsqueeze(1) + eps
    w = 1 / sigma_sq

    intensity = (relu(counts.unsqueeze(1) - zbg) * masks.unsqueeze(1) * zp * w).sum(
        -1
    ) / ((zp.pow(2) * w).sum(-1) + eps)

    decoder = UnetDecoder()
    rate, intensity_mean, intensity_variance = decoder(qbg, qp, counts, masks)

    rate * masks.unsqueeze(1)

    torch.distributions.Poisson(rate)

    print("Rate:", rate)
    print("Intensity Mean:", intensity_mean)
    print("Intensity Variance:", intensity_variance)

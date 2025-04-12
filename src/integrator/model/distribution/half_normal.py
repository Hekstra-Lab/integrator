import torch
from integrator.layers import Linear, Constraint
from torch.distributions import HalfNormal


class HalfNormalDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=1323,
    ):
        super().__init__()
        self.fc = Linear(
            in_features=dmodel,
            out_features=out_features,
        )
        self.constraint = constraint
        self.min_value = 1e-3
        self.max_value = 100.0

    def distribution(self, params):
        scale = self.constraint(params)
        scale = torch.clamp(scale, min=self.min_value, max=self.max_value)
        return torch.distributions.half_normal.HalfNormal(scale)

    def forward(self, representation):
        params = self.fc(representation)
        norm = self.distribution(params)
        return norm


if __name__ == "__main__":
    # Example usage

    dmodel = 64
    half_normal_dist = HalfNormalDistribution(dmodel)

    representation = torch.randn(10, dmodel)  # Example input

    qbg = half_normal_dist(representation)

    qbg.variance.mean(-1)

    qbg.sample([100]).permute(1, 0, 2)

    q = integrator.qbg(torch.randn(10, 1323))
    q2 = HalfNormal(2)

    torch.distributions.kl.kl_divergence(q, q2).shape

import torch

from integrator.layers import Constraint, Linear


class HalfNormalDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features=1,
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
        scale = self.constraint(params + 1e-6)
        return torch.distributions.half_normal.HalfNormal(scale.flatten())

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
    qbg.rsample([100])

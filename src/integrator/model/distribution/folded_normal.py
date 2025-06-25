from math import pi, sqrt

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform

from integrator.layers import Constraint, Linear


class FoldedNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True

    def __init__(self, loc, scale, validate_args=None):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = self.base_dist
        return torch.logaddexp(n.log_prob(value), n.log_prob(-value))

    @property
    def loc(self) -> Tensor:
        return self.base_dist.loc

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self):
        loc, scale = self.base_dist.loc, self.base_dist.scale
        a = loc / scale
        return scale * sqrt(2 / pi) * torch.exp(-0.5 * a**2) + loc * (
            1 - 2 * torch.distributions.Normal(0.0, 1.0).cdf(-a)
        )

    @property
    def variance(self):
        loc, scale = self.base_dist.loc, self.base_dist.scale
        return loc**2 + scale**2 - self.mean**2

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        rt2 = torch.sqrt(torch.tensor(2.0))
        a = (value + self.loc) / (self.scale * rt2)
        b = (value - self.loc) / (self.scale * rt2)
        return 0.5 * (torch.erf(a) + torch.erf(b))


class tempFoldedNormalDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features: int = 2,
        use_metarep: bool = False,
    ):
        super().__init__()
        self.use_metarep = use_metarep
        self.out_features = out_features
        self.constraint = constraint
        self.dmodel = dmodel
        self.fc = Linear(dmodel, self.out_features)
        self.q = FoldedNormal

    def distribution(self, loc, scale):
        scale = self.constraint(scale)
        return self.q(loc=loc, scale=scale)

    def forward(self, representation):
        params = self.fc(representation)
        loc = params[..., 0]
        scale = params[..., 1]
        return self.distribution(loc, scale)


class FoldedNormalDistribution(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        transform="relative",
        I_max=2**20 - 1,
        beta=1.0,
        eps=1e-6,
        out_features=2,
        use_metarep=False,
    ):
        super().__init__()
        self.fc = torch.nn.Linear(dmodel, 2)  # raw_loc, raw_scale
        self.transform = transform
        self.I_max = float(I_max)
        self.eps = eps  # floor for σ
        self.beta = beta  # softplus sharpness

    def _post_process(self, raw_loc, raw_scale):
        if self.transform == "log":  # --- LOG VERSION
            loc = torch.exp(raw_loc)  # μ ≥ 0
            scale = F.softplus(raw_scale, beta=self.beta) + self.eps
        elif self.transform == "squash":  # --- SIGMOID VERSION
            loc_raw = torch.sigmoid(raw_loc)  # (0,1)
            loc = loc_raw * self.I_max
            scale = F.softplus(raw_scale, beta=self.beta) + self.eps
        elif self.transform == "relative":  # --- μ, σ/μ VERSION
            loc = torch.exp(raw_loc)  # μ ≥ 0
            scale = torch.exp(raw_scale)  # σ/μ ≥ 0
        else:
            raise ValueError("unknown transform")

        return loc, scale.clamp_max(1e30)  # avoid infs

    def forward(self, representation):
        raw_loc, raw_scale = self.fc(representation).unbind(-1)
        loc, scale = self._post_process(raw_loc, raw_scale)
        return FoldedNormal(loc, scale)


if __name__ == "main":
    foldednormal = FoldedNormalDistribution(dmodel=64)
    representation = torch.randn(10, 64)
    q = foldednormal(representation)

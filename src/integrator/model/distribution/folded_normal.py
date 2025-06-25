import numpy as np
import torch
from torch import distributions as dist
from torch.distributions import Distribution, Normal, constraints
import torch.nn.functional as F


from integrator.layers import Constraint, Linear

class FoldedNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.nonnegative

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.broadcast_tensors(loc, scale)
        super().__init__(self.loc.shape, validate_args=validate_args)

    # --------------------------------------------------------------------- #
    # sampling                                                              #
    # --------------------------------------------------------------------- #
    def sample(self, sample_shape=torch.Size()):
        """Non-differentiable sampling (fast inference)."""
        shape = self._extended_shape(sample_shape)
        z = self.loc + self.scale * torch.randn(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        return z.abs()

    def rsample(self, sample_shape=torch.Size()):
        """Reparameterised (differentiable) sample."""
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        z = self.loc + self.scale * eps          #   N(loc, scale)
        return z.abs()                           # |N|  keeps autograd graph

    # --------------------------------------------------------------------- #
    # log-probability                                                       #
    # --------------------------------------------------------------------- #
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = Normal(self.loc, self.scale)
        return torch.logaddexp(n.log_prob(value), n.log_prob(-value))

    # --------------------------------------------------------------------- #
    # analytic statistics                                                   #
    # --------------------------------------------------------------------- #
    @property
    def mean(self):
        a = self.loc / self.scale
        n0 = Normal(0.0, 1.0)
        return (
            self.scale
            * torch.sqrt(torch.tensor(2.0) / torch.pi)
            * torch.exp(-0.5 * a ** 2)
            + self.loc * (1 - 2 * n0.cdf(-a))
        )

    @property
    def variance(self):
        return self.loc ** 2 + self.scale ** 2 - self.mean ** 2

    # optional convenience
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
    def __init__(self, dmodel, transform="relative", I_max=2**20-1,
                 beta=1.0, eps=1e-6,out_features=2,use_metarep=False):
        super().__init__()
        self.fc = torch.nn.Linear(dmodel, 2)     # raw_loc, raw_scale
        self.transform = transform
        self.I_max = float(I_max)
        self.eps = eps                     # floor for σ
        self.beta = beta                   # softplus sharpness

    def _post_process(self, raw_loc, raw_scale):
        if self.transform == "log":                # --- LOG VERSION
            loc   = torch.exp(raw_loc)             # μ ≥ 0
            scale = F.softplus(raw_scale, beta=self.beta) + self.eps
        elif self.transform == "squash":           # --- SIGMOID VERSION
            loc_raw = torch.sigmoid(raw_loc)       # (0,1)
            loc   = loc_raw * self.I_max
            scale = F.softplus(raw_scale, beta=self.beta) + self.eps
        elif self.transform == "relative":         # --- μ, σ/μ VERSION
            loc   = torch.exp(raw_loc)             # μ ≥ 0
            scale   = torch.exp(raw_scale)           # σ/μ ≥ 0
        else:
            raise ValueError("unknown transform")

        return loc, scale.clamp_max(1e30)          # avoid infs

    def forward(self, representation):
        raw_loc, raw_scale = self.fc(representation).unbind(-1)
        loc, scale = self._post_process(raw_loc, raw_scale)
        return FoldedNormal(loc, scale)



if __name__ == "main":
    foldednormal = FoldedNormalDistribution(dmodel=64)
    representation = torch.randn(10, 64)
    q = foldednormal(representation)

import torch
from math import pi, sqrt
import torch.nn.functional as F
from torch import distributions as dist
from torch.distributions.transformed_distribution import TransformedDistribution
from torch import distributions as dist
from torch.distributions.transforms import AbsTransform
from torch.distributions import Normal, constraints
from torch import Tensor





class NormalIRSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loc, scale, samples, dFdmu, dFdsig, q):
        dzdmu = -dFdmu / q
        dzdsig = -dFdsig / q
        ctx.save_for_backward(dzdmu, dzdsig)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        (
            dzdmu,
            dzdsig,
        ) = ctx.saved_tensors
        return grad_output * dzdmu, grad_output * dzdsig, None, None, None, None


class tempFoldedNormal(dist.Distribution):
    """
    Folded Normal distribution class

    Args:
        loc (float or Tensor): location parameter of the distribution
        scale (float or Tensor): scale parameter of the distribution (must be positive)
        validate_args (bool, optional): Whether to validate the arguments of the distribution.
        Default is None.
    """

    arg_constraints = {"loc": dist.constraints.real, "scale": dist.constraints.positive}
    support = torch.distributions.constraints.nonnegative

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = torch.distributions.utils.broadcast_all(loc, scale)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, validate_args=validate_args)
        self._irsample = NormalIRSample.apply

    def log_prob(self, value):
        """
        Compute the log-probability of the given values under the Folded Normal distribution

        Args:
            value (Tensor): The values at which to evaluate the log-probability

        Returns:
            Tensor: The log-probabilities of the given values
        """
        if self._validate_args:
            self._validate_sample(value)
        loc = self.loc
        scale = self.scale
        log_prob = torch.logaddexp(
            dist.Normal(loc, scale).log_prob(value),
            dist.Normal(-loc, scale).log_prob(value),
        )
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        """
        Generate random samples from the Folded Normal distribution

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        samples = torch.abs(eps * self.scale + self.loc)

        return samples

    @property
    def mean(self):
        """
        Compute the mean of the Folded Normal distribution

        Returns:
            Tensor: The mean of the distribution.
        """
        loc = self.loc
        scale = self.scale
        return scale * torch.sqrt(torch.tensor(2.0) / torch.pi) * torch.exp(
            -0.5 * (loc / scale) ** 2
        ) + loc * (1 - 2 * dist.Normal(0, 1).cdf(-loc / scale))

    @property
    def variance(self):
        """
        Compute the variance of the Folded Normal distribution

        Returns:
            Tensor: The variance of the distribution
        """
        loc = self.loc
        scale = self.scale
        return loc**2 + scale**2 - self.mean**2

    def cdf(self, value):
        """
        Args:
            value (Tensor): The values at which to evaluate the CDF

        Returns:
            Tensor: The CDF values at the given values
        """
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        # return dist.Normal(loc, scale).cdf(value) - dist.Normal(-loc, scale).cdf(-value)
        return 0.5 * (
            torch.erf((value + self.loc) / (self.scale * np.sqrt(2.0)))
            + torch.erf((value - self.loc) / (self.scale * np.sqrt(2.0)))
        )

    def dcdfdmu(self, value):
        return torch.exp(
            dist.Normal(-self.loc, self.scale).log_prob(value)
        ) - torch.exp(dist.Normal(self.loc, self.scale).log_prob(value))

    def dcdfdsigma(self, value):
        A = (-(value + self.loc) / self.scale) * torch.exp(
            dist.Normal(-self.loc, self.scale).log_prob(value)
        )
        B = (-(value - self.loc) / self.scale) * torch.exp(
            dist.Normal(self.loc, self.scale).log_prob(value)
        )
        return A + B

    def pdf(self, value):
        return torch.exp(self.log_prob(value))

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate differentiable random samples from the Folded Normal distribution.
        Gradients are implemented using implicit reparameterization (https://arxiv.org/abs/1805.08498).

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        samples = self.sample(sample_shape)
        # F = self.cdf(samples)
        q = self.pdf(samples)
        dFdmu = self.dcdfdmu(samples)
        dFdsigma = self.dcdfdsigma(samples)
        return self._irsample(self.loc, self.scale, samples, dFdmu, dFdsigma, q)

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



class FoldedNormalDistribution(torch.nn.Module):
    """
    Attributes:
        fc:
        transform:
        I_max:
        eps:
        beta:
    """

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

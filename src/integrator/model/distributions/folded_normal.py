from math import pi, sqrt
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal, constraints
from torch.distributions.constraints import Constraint
from torch.distributions.transformed_distribution import (
    TransformedDistribution,
)
from torch.distributions.transforms import AbsTransform

from integrator.layers import Constrain


class FoldedNormal(TransformedDistribution):
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        self._normal = Normal(loc, scale, validate_args=validate_args)
        super().__init__(
            self._normal, AbsTransform(), validate_args=validate_args
        )

    @property
    def has_rsample(self) -> bool:
        return True

    @property
    def support(self) -> Constraint:
        return constraints.nonnegative

    @property
    def loc(self) -> Tensor:
        return self._normal.loc

    @property
    def scale(self) -> Tensor:
        return self._normal.scale

    @property
    def mean(self) -> Tensor:
        loc, scale = self._normal.loc, self._normal.scale
        a = loc / scale
        return scale * sqrt(2 / pi) * torch.exp(-0.5 * a**2) + loc * (
            1 - 2 * torch.distributions.Normal(0.0, 1.0).cdf(-a)
        )

    @property
    def variance(self) -> Tensor:
        loc, scale = self._normal.loc, self._normal.scale
        return loc**2 + scale**2 - self.mean**2

    def cdf(self, value) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        rt2 = torch.sqrt(torch.tensor(2.0))
        a = (value + self.loc) / (self.scale * rt2)
        b = (value - self.loc) / (self.scale * rt2)
        return 0.5 * (torch.erf(a) + torch.erf(b))

    def log_prob(self, value) -> Tensor:
        if self._validate_args:
            self._validate_sample(value)
        n = self._normal
        return torch.logaddexp(n.log_prob(value), n.log_prob(-value))


# -
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
        return (
            grad_output * dzdmu,
            grad_output * dzdsig,
            None,
            None,
            None,
            None,
        )


# class FoldedNormal(Distribution):
#     """
#     Folded Normal distribution class
#
#     Args:
#         loc (float or Tensor): location parameter of the distribution
#         scale (float or Tensor): scale parameter of the distribution (must be positive)
#         validate_args (bool, optional): Whether to validate the arguments of the distribution.
#         Default is None.
#     """
#
#     arg_constraints = {
#         "loc": dist.constraints.real,
#         "scale": dist.constraints.positive,
#     }
#     support = torch.distributions.constraints.nonnegative
#
#     def __init__(self, loc, scale, var_thresh=5, validate_args=None):
#         self.loc, self.scale = torch.distributions.utils.broadcast_all(
#             loc, scale
#         )
#         batch_shape = self.loc.shape
#         super().__init__(batch_shape, validate_args=validate_args)
#         self._irsample = NormalIRSample.apply
#         self.var_thresh = var_thresh
#
#     def log_prob(self, value):
#         """
#         Compute the log-probability of the given values under the Folded Normal distribution
#
#         Args:
#             value (Tensor): The values at which to evaluate the log-probability
#
#         Returns:
#             Tensor: The log-probabilities of the given values
#         """
#         if self._validate_args:
#             self._validate_sample(value)
#         loc = self.loc
#         scale = self.scale
#         log_prob = torch.logaddexp(
#             torch.distributions.Normal(loc, scale).log_prob(value),
#             torch.distributions.Normal(-loc, scale).log_prob(value),
#         )
#         return log_prob
#
#     def sample(self, sample_shape=torch.Size()):
#         """
#         Generate random samples from the Folded Normal distribution
#
#         Args:
#             sample_shape (torch.Size, optional): The shape of the samples to generate.
#             Default is an empty shape
#
#         Returns:
#             Tensor: The generated random samples
#         """
#         shape = self._extended_shape(sample_shape)
#         eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
#         samples = torch.abs(eps * self.scale + self.loc)
#
#         return samples
#
#     @property
#     def mean(self):
#         """
#         Compute the mean of the Folded Normal distribution
#
#         Returns:
#             Tensor: The mean of the distribution.
#         """
#         loc = self.loc
#         scale = self.scale
#         return scale * torch.sqrt(torch.tensor(2.0) / torch.pi) * torch.exp(
#             -0.5 * (loc / scale) ** 2
#         ) + loc * (1 - 2 * torch.distributions.Normal(0, 1).cdf(-loc / scale))
#
#     #
#     # @property
#     # def variance(self):
#     #     """
#     #     Compute the variance of the Folded Normal distribution
#     #
#     #     Returns:
#     #         Tensor: The variance of the distribution
#     #     """
#     #     loc = self.loc
#     #     scale = self.scale
#     #
#     #     return loc**2 + scale**2 - self.mean**2
#     #
#     @property
#     def variance(self):
#         """
#         Compute the variance of the Folded Normal distribution
#         with numerical safeguards for large loc/scale ratios.
#         """
#         loc = self.loc
#         scale = self.scale
#         a = loc / scale
#         mean = self.mean
#
#         # threshold logic: use normal variance when a > 5
#         var = torch.empty_like(loc)
#         large = a > self.var_thresh
#         small = ~large
#
#         if large.any():
#             var[large] = scale[large] ** 2
#
#         if small.any():
#             # stable variant of μ²+σ²−E[X]²
#
#             var[small] = loc[small] ** 2 + scale[small] ** 2 - mean[small] ** 2
#
#         return var
#
#     def cdf(self, value):
#         """
#         Args:
#             value (Tensor): The values at which to evaluate the CDF
#
#         Returns:
#             Tensor: The CDF values at the given values
#         """
#         if self._validate_args:
#             self._validate_sample(value)
#         value = torch.as_tensor(
#             value, dtype=self.loc.dtype, device=self.loc.device
#         )
#         # return dist.Normal(loc, scale).cdf(value) - dist.Normal(-loc, scale).cdf(-value)
#         return 0.5 * (
#             torch.erf((value + self.loc) / (self.scale * np.sqrt(2.0)))
#             + torch.erf((value - self.loc) / (self.scale * np.sqrt(2.0)))
#         )
#
#     def dcdfdmu(self, value):
#         return torch.exp(
#             torch.distributions.Normal(-self.loc, self.scale).log_prob(value)
#         ) - torch.exp(dist.Normal(self.loc, self.scale).log_prob(value))
#
#     def dcdfdsigma(self, value):
#         A = (-(value + self.loc) / self.scale) * torch.exp(
#             torch.distributions.Normal(-self.loc, self.scale).log_prob(value)
#         )
#         B = (-(value - self.loc) / self.scale) * torch.exp(
#             torch.distributions.Normal(self.loc, self.scale).log_prob(value)
#         )
#         return A + B
#
#     def pdf(self, value):
#         return torch.exp(self.log_prob(value))
#
#     def rsample(self, sample_shape=torch.Size()):
#         """
#         Generate differentiable random samples from the Folded Normal distribution.
#         Gradients are implemented using implicit reparameterization (https://arxiv.org/abs/1805.08498).
#
#         Args:
#             sample_shape (torch.Size, optional): The shape of the samples to generate.
#             Default is an empty shape
#
#         Returns:
#             Tensor: The generated random samples
#         """
#         samples = self.sample(sample_shape)
#         # F = self.cdf(samples)
#         q = self.pdf(samples)
#         dFdmu = self.dcdfdmu(samples)
#         dFdsigma = self.dcdfdsigma(samples)
#         return self._irsample(
#             self.loc, self.scale, samples, dFdmu, dFdsigma, q
#         )
#
#
# # class FoldedNormalDistribution(nn.Module):
#     """
#     FoldedNormal distribution with parameters predicted by a linear layer.
#     """
#
#     def __init__(
#         self,
#         in_features: int = 64,
#         constraint: Literal["exp", "softplus"] | None = "softplus",
#         out_features: int = 2,
#         eps: float = 0.1,
#         beta: int = 1,
#     ):
#         super().__init__()
#         self.fc = torch.nn.Linear(
#             in_features,
#             out_features,
#         )
#         self.constrain_fn = Constrain(
#             constraint_fn=constraint,
#             eps=eps,
#             beta=beta,
#         )
#
#     def forward(
#         self,
#         x: Tensor,
#     ) -> FoldedNormal:
#         # raw params
#         raw_loc, raw_scale = self.fc(x).unbind(-1)
#
#         # transform
#         loc = torch.exp(raw_loc)
#         scale = self.constrain_fn(raw_scale)
#
#         return FoldedNormal(loc, scale)


class FoldedNormalDistribution(nn.Module):
    """
    FoldedNormal distribution with parameters predicted by a linear layer.
    """

    def __init__(
        self,
        in_features: int = 64,
        constraint: Literal["exp", "softplus"] | None = "softplus",
        out_features: int = 2,
        eps: float = 0.1,
        beta: int = 1,
    ):
        super().__init__()
        self.linear_loc = torch.nn.Linear(in_features, 1)
        self.linear_scale = torch.nn.Linear(in_features, 1)
        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )

    def forward(
        self,
        x: Tensor,
        x_: Tensor,
    ) -> FoldedNormal:
        # raw params

        raw_loc = self.linear_loc(x)
        raw_scale = self.linear_scale(x_)

        # transform
        loc = torch.exp(raw_loc)
        scale = self.constrain_fn(raw_scale)

        return FoldedNormal(loc.flatten(), scale.flatten())


if __name__ == "main":
    foldednormal = FoldedNormalDistribution(in_features=64)
    representation = torch.randn(10, 64)

    q = foldednormal(representation, representation)


torch.distributions.Gamma(10, 0.0001).mean

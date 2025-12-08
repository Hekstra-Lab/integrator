from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Dirichlet

from integrator.layers import Constrain, Linear

# class DirichletDistribution(nn.Module):
#     """
#     Dirichlet distribution with parameters predicted by a linear layer.
#     """
#
#     input_shape: tuple[int, ...]
#     """Shape of an input shoebox as ``(C, H, W)`` or ``(H, W)``."""
#
#     def __init__(
#         self,
#         in_features: int = 64,
#         out_features: tuple[int, ...] = (3, 21, 21),
#         constraint: Literal["exp", "softplus"] | None = "softplus",
#         eps: float = 0.01,
#         beta: int = 1,
#     ):
#         """
#         Args:
#             in_features: Input feature dimension.
#             out_features: ``(C, H, W)`` or ``(H, W)`` used to calculate dimension of the Diritchlet concentration paramter.
#         """
#         super().__init__()
#
#         if len(out_features) == 3:
#             self.num_components = (
#                 out_features[0] * out_features[1] * out_features[2]
#             )
#         elif len(out_features) == 2:
#             self.num_components = out_features[0] * out_features[1]
#
#         if out_features is not None:
#             self.alpha_layer = Linear(
#                 in_features,
#                 self.num_components,
#             )
#
#         self.constrain_fn = Constrain(
#             constraint_fn=constraint,
#             eps=eps,
#             beta=beta,
#         )
#         self.beta = beta
#         self.eps = eps
#         self.min_log_alpha = math.log(1e-3)
#         self.max_log_alpha = math.log(1e3)
#
#     def forward(self, x: Tensor) -> Dirichlet:
#         """
#         Return a `torch.distributions.Dirichlet` from an input shoebox
#
#         Args:
#             x: Input batch of shoeboxes
#
#         Returns:
#             qp: A `torch.distributions.Dirichlet(x)`
#
#         """
#         # x = self.alpha_layer(x)
#         # x = self.constrain_fn(x) + self.eps
#
#         if torch.isnan(x).any():
#             raise RuntimeError("NaNs in Dirichlet input x")
#
#         log_alpha = self.alpha_layer(x)
#
#         if torch.isnan(log_alpha).any():
#             raise RuntimeError("NaNs right after Dirichlet fc")
#
#         log_alpha = torch.clamp(
#             log_alpha, self.min_log_alpha, self.max_log_alpha
#         )
#         alpha = torch.exp(log_alpha)
#
#         if torch.isnan(alpha).any() or (alpha <= 0).any():
#             raise RuntimeError(
#                 "NaNs or nonpositive alpha before constructing Dirichlet"
#             )
#         # keep log_alpha within [-4, 4]
#         # log_alpha = torch.tanh(log_alpha) * 4.0
#         # alpha = torch.exp(log_alpha)
#         qp = Dirichlet(alpha)
#
#         return qp
#


class DirichletDistribution(nn.Module):
    """
    Dirichlet distribution with parameters predicted by a linear layer.
    """

    input_shape: tuple[int, ...]

    def __init__(
        self,
        in_features: int = 64,
        out_features: tuple[int, ...] = (3, 21, 21),
        constraint: Literal["exp", "softplus"] | None = "softplus",
        eps: float = 0.01,
        beta: int = 1,
        alpha_min: float = 0.05,
        alpha_max: float = 30.0,
        total_min: float = 10.0,
        total_max: float = 200.0,
    ):
        super().__init__()

        if len(out_features) == 3:
            self.num_components = (
                out_features[0] * out_features[1] * out_features[2]
            )
        elif len(out_features) == 2:
            self.num_components = out_features[0] * out_features[1]
        else:
            raise ValueError("out_features must be (C,H,W) or (H,W)")

        self.alpha_layer = Linear(in_features, self.num_components, bias=False)

        # optional extra head to control total concentration (scalar)
        self.total_layer = Linear(in_features, 1, bias=True)

        self.constrain_fn = Constrain(
            constraint_fn=constraint,
            eps=eps,
            beta=beta,
        )

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.total_min = total_min
        self.total_max = total_max

    def forward(self, x: Tensor) -> Dirichlet:
        if torch.isnan(x).any():
            raise RuntimeError("NaNs in Dirichlet input x")

        # Base logits for the *shape* of the profile
        logits = self.alpha_layer(x)  # (B, K)
        if torch.isnan(logits).any():
            raise RuntimeError("NaNs right after Dirichlet fc")

        # Convert to probabilities π over pixels
        pi = torch.softmax(logits, dim=-1)  # (B, K)

        # Predict a scalar total concentration s in [total_min, total_max]
        total_raw = self.total_layer(x)  # (B, 1)
        s = torch.sigmoid(total_raw)  # (0,1)
        s = self.total_min + (self.total_max - self.total_min) * s  # (B,1)

        # α = s * π, then clip each component to [alpha_min, alpha_max]
        alpha = s * pi
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)

        if torch.isnan(alpha).any() or (alpha <= 0).any():
            raise RuntimeError("NaNs or nonpositive alpha before Dirichlet")

        return Dirichlet(alpha)


if __name__ == "__main__":
    pass

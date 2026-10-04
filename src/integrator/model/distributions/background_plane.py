"""Variational tilted-plane background over the shoebox.

The background is modelled as one scalar per shoebox, but every specialized Laue
package fits a plane: Precognition carries per-spot background slopes `px, py`,
PrOW and Epinorm fit a local surface. Laue shoeboxes are 96.6% background and the
detector background varies strongly with radius, so across a 21x21 box a constant is
misspecified, and the residual lands on the weak reflections whose intensity is a
small perturbation on that background.

Fitting the plane by maximum likelihood per shoebox would reintroduce the problem the
geometry-conditioned profile exists to solve: two more nuisance parameters estimated
from the same pixels that are then integrated, which makes the intensity estimator
nonlinear in the counts and its attenuation random from one observation to the next.
So the slopes get a variational posterior and a prior that shrinks them to flat, and
the tilt is only paid for where the pixels actually support it.

The tilt is multiplicative and bounded,

    b_p = b0 * (1 + g_x u_p + g_y v_p),   g = max_gradient * tanh(g_raw),

with (u, v) the pixel offsets from the box centre scaled to [-1, 1]. Multiplicative
because the gradient of a smooth background scales with its level, which also makes
the prior scale-free and transferable across datasets; bounded because a Poisson rate
must stay positive and `|g_x| + |g_y| <= 2 * max_gradient < 1` guarantees it without a
clamp that would distort the plane. Since (u, v) are centred, the tilt averages to one
over the box, so the box total is unchanged and `qbg.mean` keeps its meaning as the
mean background per pixel; only its distribution across pixels changes.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

__all__ = ["BackgroundPlaneOutput", "BackgroundPlaneSurrogate"]


@dataclass
class BackgroundPlaneOutput:
    """Per-pixel multiplicative background tilt and the posterior that produced it."""

    tilt: Tensor
    """Sampled tilt factors, shape (B, S, P), mean one over the box."""
    mean_tilt: Tensor
    """Tilt at the posterior mean slopes, shape (B, P)."""
    q: Normal
    """Posterior over the unbounded slopes, batch shape (B, 2)."""
    gradient: Tensor
    """Bounded slopes at the posterior mean, shape (B, 2), as a fraction of `b0`."""
    prior_scale: float


class BackgroundPlaneSurrogate(nn.Module):
    """Amortized posterior over the background plane's two slopes."""

    def __init__(
        self,
        in_features: int,
        sbox_shape: list[int] | tuple[int, int, int],
        max_gradient: float = 0.45,
        prior_scale: float = 0.1,
        init_scale: float = 0.05,
        zero_init: bool = True,
    ) -> None:
        """
        Args:
            in_features: Width of the encoder output this reads.
            sbox_shape: Shoebox shape `(d, h, w)`; the plane is over `(h, w)`.
                Injected by `resolve_config`, so configs need not set it.
            max_gradient: Bound on each slope. Must stay below 0.5 so that
                `|g_x| + |g_y| < 1` and the tilt cannot reach zero.
            prior_scale: Std of the `N(0, prior_scale)` prior on the raw slopes.
                Small values shrink the background to flat unless the pixels say
                otherwise.
            init_scale: Initial posterior std, before the scale head learns.
            zero_init: Start the slope head at zero, so training begins from the
                flat background the model had before and the plane has to be earned.
        """
        super().__init__()
        if not 0.0 < max_gradient < 0.5:
            raise ValueError(
                f"max_gradient must be in (0, 0.5) to keep the rate positive, "
                f"got {max_gradient}"
            )
        d, h, w = sbox_shape
        self.max_gradient = float(max_gradient)
        self.prior_scale = float(prior_scale)

        self.loc_head = nn.Linear(in_features, 2)
        self.scale_head = nn.Linear(in_features, 2)
        if zero_init:
            nn.init.zeros_(self.loc_head.weight)
            nn.init.zeros_(self.loc_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        # softplus(x) = init_scale
        inv = torch.log(torch.expm1(torch.tensor(float(init_scale))))
        nn.init.constant_(self.scale_head.bias, float(inv))

        # pixel offsets from the box centre, scaled to [-1, 1] and shared by every
        # reflection: the plane lives in the detector, so it repeats across depth
        ys = torch.linspace(-1.0, 1.0, h) if h > 1 else torch.zeros(1)
        xs = torch.linspace(-1.0, 1.0, w) if w > 1 else torch.zeros(1)
        vv, uu = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)  # (h*w, 2)
        self.register_buffer("coords", coords.repeat(d, 1), persistent=False)

    def forward(self, x: Tensor, mc_samples: int = 1) -> BackgroundPlaneOutput:
        loc = self.loc_head(x)
        scale = F.softplus(self.scale_head(x)) + 1e-6
        q = Normal(loc, scale)

        g_raw = q.rsample([mc_samples])  # (S, B, 2)
        g = self.max_gradient * torch.tanh(g_raw)
        coords = self.coords.to(g.dtype)
        # (S, B, 2) @ (2, P) -> (S, B, P), then to (B, S, P) like the other samples
        tilt = 1.0 + g @ coords.T
        tilt = tilt.permute(1, 0, 2)

        g_mean = self.max_gradient * torch.tanh(loc)
        mean_tilt = 1.0 + g_mean @ coords.T

        return BackgroundPlaneOutput(
            tilt=tilt,
            mean_tilt=mean_tilt,
            q=q,
            gradient=g_mean,
            prior_scale=self.prior_scale,
        )


def compute_background_plane_kl(
    plane: BackgroundPlaneOutput | None,
    weight: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """KL(q(g_raw) || N(0, prior_scale)), summed over the two slopes.

    Analytic because the bound is applied after sampling, so the posterior and prior
    are both Normal on the same unbounded coordinates.
    """
    if plane is None or weight == 0.0:
        return torch.zeros(batch_size, device=device)
    q = plane.q
    p = Normal(
        torch.zeros_like(q.loc),
        torch.full_like(q.scale, plane.prior_scale),
    )
    return torch.distributions.kl_divergence(q, p).sum(-1) * weight

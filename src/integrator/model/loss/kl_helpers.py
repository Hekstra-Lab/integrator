import torch
from torch import Tensor
from torch.distributions import Distribution, Normal, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


def compute_profile_kl(
    qp: Distribution | ProfileSurrogateOutput,
    prior_scale: float | Tensor,
    pprf_weight: float,
    device: torch.device,
) -> Tensor:
    if isinstance(qp, ProfileSurrogateOutput):
        q = Normal(qp.loc, qp.scale)
        if isinstance(prior_scale, Tensor) and prior_scale.dim() >= 1:
            prior_scale = prior_scale.unsqueeze(-1)
        p = Normal(torch.zeros_like(qp.loc), prior_scale)
        kl = kl_divergence(q, p).sum(dim=-1)
        return kl * pprf_weight

    from torch.distributions import Dirichlet

    if isinstance(qp, Dirichlet):
        n_pixels = qp.concentration.shape[-1]
        prior = Dirichlet(torch.ones(n_pixels, device=device))
        kl = kl_divergence(qp, prior)
        return kl * pprf_weight

    raise NotImplementedError(
        f"Profile surrogate of type {type(qp).__name__} is not supported "
        "by compute_profile_kl."
    )

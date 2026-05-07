import torch
from torch import Tensor
from torch.distributions import Distribution, Normal, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


def _load_buffer(value: list[float] | str) -> Tensor:
    """Load a tensor from a list of floats or a .pt file path."""
    if isinstance(value, str):
        loaded = torch.load(value, weights_only=True)
        if isinstance(loaded, dict):
            return next(iter(loaded.values())).float()
        return loaded.float()
    return torch.tensor(value, dtype=torch.float32)


def compute_profile_kl(
    qp: Distribution | ProfileSurrogateOutput,
    prior_scale: float,
    pprf_weight: float,
    device: torch.device,
) -> Tensor:
    """Compute profile KL divergence.

    Supports:
    - ProfileSurrogateOutput: Normal-Normal KL with global N(0, prior_scale^2 I)
    - Dirichlet: KL(q || Dirichlet(1,...,1)) with uniform prior
    """
    if isinstance(qp, ProfileSurrogateOutput):
        q = Normal(qp.loc, qp.scale)
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

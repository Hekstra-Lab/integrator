from pathlib import Path

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal, kl_divergence

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


def compute_profile_kl(
    qp: Distribution | ProfileSurrogateOutput,
    prior_scale: float | Tensor,
    profile_kl_weight: float,
    device: torch.device,
) -> Tensor:
    if isinstance(qp, ProfileSurrogateOutput):
        # a surrogate whose prior differs per latent component reports its own KL,
        # since a single scalar scale cannot express it
        if qp.kl is not None:
            return qp.kl * profile_kl_weight

        q = Normal(qp.loc, qp.scale)
        if isinstance(prior_scale, Tensor) and prior_scale.dim() >= 1:
            prior_scale = prior_scale.unsqueeze(-1)
        p = Normal(torch.zeros_like(qp.loc), prior_scale)
        kl = kl_divergence(q, p).sum(dim=-1)
        return kl * profile_kl_weight

    from torch.distributions import Dirichlet

    if isinstance(qp, Dirichlet):
        n_pixels = qp.concentration.shape[-1]
        prior = Dirichlet(torch.ones(n_pixels, device=device))
        kl = kl_divergence(qp, prior)
        return kl * profile_kl_weight

    raise NotImplementedError(
        f"Profile surrogate of type {type(qp).__name__} is not supported "
        "by compute_profile_kl."
    )


def _kl(
    q: Distribution,
    p: Distribution,
    n_samples: int = 8,
    eps: float = 1.0e-8,
) -> Tensor:
    """KL(q || p), falling back to Monte Carlo where no closed form exists.

    PyTorch registers exact KL formulas for some pairs, Gamma-Gamma among them,
    which is why a Gamma `qi` against the Gamma Wilson prior works directly. A
    LogNormal or FoldedNormal `qi` against the same Gamma prior has no
    registered formula and raises `NotImplementedError`, so the estimate
    `mean(log q(x) - log p(x))` over `x ~ q` stands in. The sampled branch is
    noisy but unbiased, and `n_samples` trades that noise against cost.

    `eps` floors the samples before `log_prob`, because a Gamma prior's density
    diverges at zero and a surrogate with support down to 0 will occasionally
    land there.
    """
    try:
        return kl_divergence(q, p)
    except NotImplementedError:
        x = q.rsample((n_samples,)) if q.has_rsample else q.sample((n_samples,))
        x = x.clamp_min(eps)
        return (q.log_prob(x) - p.log_prob(x)).mean(0)


def compute_bg_kl(
    qbg: Distribution,
    groups: Tensor | None,
    bg_rate_per_group: Tensor | None,
    bg_concentration_per_group: Tensor | None,
    bg_concentration: float | Tensor,
    bg_kl_weight: float,
    n_samples: int = 8,
    eps: float = 1.0e-8,
) -> Tensor:
    """KL of the background posterior against its Gamma prior.

    The prior is per-resolution-bin when `bg_*_per_group` are supplied -- Laue
    shoeboxes are ~97% background and its level varies strongly with resolution,
    so one global rate misprices most of the box -- and a single shared Gamma
    otherwise. Mirrors the branch in `WilsonLoss`, and routes through `_kl` so a
    non-Gamma background surrogate degrades to sampling rather than raising.
    """
    from torch.distributions import Gamma

    if bg_rate_per_group is not None and groups is not None:
        g = groups.long()
        conc = (
            bg_concentration_per_group[g]
            if bg_concentration_per_group is not None
            else torch.as_tensor(
                bg_concentration, device=bg_rate_per_group.device
            )
        )
        p_bg = Gamma(concentration=conc, rate=bg_rate_per_group[g])
    else:
        conc = torch.as_tensor(bg_concentration)
        p_bg = Gamma(concentration=conc, rate=conc)

    return _kl(qbg, p_bg, n_samples, eps=eps) * bg_kl_weight


def _load_buffer(value) -> Tensor:
    """Coerce a config value into a 1-D tensor, loading it from disk if needed.

    A buffer-valued knob can be written inline in the YAML as a list, or given
    as a filename that `factory_utils._resolve_data_path` has already turned
    into an absolute path under the dataset directory (see the
    `wavelength_bin_edges` branch there). Both reach the loss as this one
    argument, so both are handled here rather than at each call site.
    """
    import numpy as np

    if isinstance(value, Tensor):
        return value.detach().flatten().float()
    if isinstance(value, (str, Path)):
        p = Path(value)
        if not p.is_file():
            raise FileNotFoundError(f"buffer file not found: {p}")
        obj = (
            np.load(p)
            if p.suffix in (".npy", ".npz")
            else torch.load(p, weights_only=False, map_location="cpu")
        )
        if isinstance(obj, dict):
            # a saved state-dict-like payload: take its only array
            vals = [v for v in obj.values() if hasattr(v, "__len__")]
            if len(vals) != 1:
                raise ValueError(
                    f"{p} holds {len(vals)} array-like entries; expected one"
                )
            obj = vals[0]
        return torch.as_tensor(np.asarray(obj)).flatten().float()
    return torch.as_tensor(np.asarray(value)).flatten().float()


def compute_profile_kl_global(
    loc: Tensor,
    scale: Tensor,
    prior_scale: float | Tensor,
) -> Tensor:
    """KL of a latent profile code against a zero-mean Normal prior.

    The `ProfileSurrogateOutput` branch of `compute_profile_kl` without the
    weight applied, kept separate because `loss.py` passes the three tensors
    rather than the surrogate output. `prior_scale` is broadcast over the latent
    dimension, so a per-observation 1-D scale gains the trailing axis here.
    """
    if isinstance(prior_scale, Tensor) and prior_scale.dim() >= 1:
        prior_scale = prior_scale.unsqueeze(-1)
    q = Normal(loc, scale)
    p = Normal(torch.zeros_like(loc), prior_scale)
    return kl_divergence(q, p).sum(dim=-1)

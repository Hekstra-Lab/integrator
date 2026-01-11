from math import prod

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import (
    Dirichlet,
    Distribution,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    LogNormal,
    Poisson,
)

from integrator.configs.config_utils import shallow_dict
from integrator.configs.priors import DirichletParams, PriorConfig

PRIOR_MAP = {
    "gamma": Gamma,
    "log_normal": LogNormal,
    "half_normal": HalfNormal,
    "half_cauchy": HalfCauchy,
    "exponential": Exponential,
    "dirichlet": Dirichlet,
}


def create_center_focused_dirichlet_prior(
    shape: tuple[int, ...] = (3, 21, 21),
    base_alpha: float = 0.1,  # outer region
    center_alpha: float = 100.0,  # high alpha at the center
    decay_factor: float = 1.0,
    peak_percentage: float = 0.1,
) -> Tensor:
    channels, height, width = shape
    alpha_3d = np.ones(shape) * base_alpha

    # center indices
    center_c = channels // 2
    center_h = height // 2
    center_w = width // 2

    # loop over voxels
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                # Normalized distance from center
                dist_c = abs(c - center_c) / (channels / 2)
                dist_h = abs(h - center_h) / (height / 2)
                dist_w = abs(w - center_w) / (width / 2)
                distance = np.sqrt(
                    dist_c**2 + dist_h**2 + dist_w**2
                ) / np.sqrt(3)

                if distance < peak_percentage * 5:
                    alpha_value = (
                        center_alpha
                        - (center_alpha - base_alpha)
                        * (distance / (peak_percentage * 5)) ** decay_factor
                    )
                    alpha_3d[c, h, w] = alpha_value

    alpha_vector = torch.tensor(alpha_3d.flatten(), dtype=torch.float32)
    return alpha_vector


def _get_dirichlet_prior(
    prior: PriorConfig[DirichletParams] | None,
) -> dict[str, torch.Tensor] | None:
    if prior is None:
        return None

    conc = prior.params.concentration

    if isinstance(conc, float) or isinstance(conc, int):
        k = prod(prior.params.shape)
        return {"concentration": torch.full((k,), float(conc))}

    if isinstance(conc, str):
        loaded = torch.load(conc)
        loaded[loaded > 2] *= 40
        loaded /= loaded.sum()
        return {"concentration": loaded.reshape(-1)}

    raise TypeError(f"Unsupported concentration type: {type(conc)}")


def _build_prior(
    prior_cfg: PriorConfig,
    params: dict[str, torch.Tensor],
    device: torch.device,
) -> Distribution:
    dist = PRIOR_MAP.get(prior_cfg.name)
    if dist is None:
        raise ValueError(f"Unknown prior name: {prior_cfg.name}")

    return dist(**{k: v.to(device) for k, v in params.items()})


def _kl(
    q: Distribution,
    p: Distribution,
    mc_samples: int,
):
    try:
        return torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError:
        samples = q.rsample(torch.Size([mc_samples]))
        log_q = q.log_prob(samples)
        log_p = p.log_prob(samples)
        return (log_q - log_p).mean(dim=0)


def _prior_kl(
    prior_cfg: PriorConfig,
    q: Distribution,
    params: dict[str, torch.Tensor],
    weight: float,
    device: torch.device,
    mc_samples: int,
) -> torch.Tensor:
    p = _build_prior(
        prior_cfg,
        params,
        device,
    )
    kl_prior = _kl(q, p, mc_samples)
    return kl_prior * weight


def _params_as_tensors(
    prior: PriorConfig | None,
) -> dict[str, torch.Tensor] | None:
    if prior is None or prior.params is None:
        return None
    params = shallow_dict(prior.params)
    return {k: torch.tensor(v) for k, v in params.items()}


class Loss(nn.Module):
    def __init__(
        self,
        *,
        pprf_cfg: PriorConfig | None,
        pi_cfg: PriorConfig | None,
        pbg_cfg: PriorConfig | None,
        mc_samples: int = 100,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps
        self.mc_samples = mc_samples

        if pi_cfg is not None:
            self.pi_params = _params_as_tensors(pi_cfg)
            self.pi_cfg = pi_cfg
        if pbg_cfg is not None:
            self.pbg_params = _params_as_tensors(pbg_cfg)
            self.pbg_cfg = pbg_cfg
        if pprf_cfg is not None:
            self.pprf_params = _get_dirichlet_prior(pprf_cfg)
            self.pprf_cfg = pprf_cfg

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
    ):
        # batch metadata
        device = rate.device
        print(device)
        batch_size = rate.shape[0]

        # moving data to device
        counts = counts.to(device)
        mask = mask.to(device)

        # total KL and per-prior KLs
        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)

        # Calculating per prior KL temrms
        if self.pprf_cfg is not None and self.pprf_params is not None:
            kl_prf = _prior_kl(
                prior_cfg=self.pprf_cfg,
                q=qp,
                params=self.pprf_params,
                weight=self.pprf_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
            kl += kl_prf

        if self.pi_cfg is not None and self.pi_params is not None:
            kl_i = _prior_kl(
                prior_cfg=self.pi_cfg,
                q=qi,
                params=self.pi_params,
                weight=self.pi_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
            kl += kl_i

        if self.pbg_cfg is not None and self.pbg_params is not None:
            kl_bg = _prior_kl(
                prior_cfg=self.pbg_cfg,
                q=qbg,
                params=self.pbg_params,
                weight=self.pbg_cfg.weight,
                device=device,
                mc_samples=self.mc_samples,
            )
            kl += kl_bg

        # Calculating log likelihood
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # Total loss
        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }

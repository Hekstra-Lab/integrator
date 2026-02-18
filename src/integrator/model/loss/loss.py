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
        n_components = loaded.numel()
        loaded = loaded / loaded.sum() * n_components  # avg alpha = 1.0
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


def _joint_prior_kl(
    q_ib: Distribution,
    pi_cfg: PriorConfig | None,
    pbg_cfg: PriorConfig | None,
    pi_params: dict[str, torch.Tensor] | None,
    pbg_params: dict[str, torch.Tensor] | None,
    device: torch.device,
    mc_samples: int,
    weight: float = 1.0,
) -> torch.Tensor:
    """MC estimate of KL(q(I,B) || p(I)*p(B)) for an independent prior.

    Draws joint samples from q_ib, evaluates the joint log-prob under q,
    and compares to the sum of independent prior log-probs:

        KL ≈ E_q[log q(I,B) - log p(I) - log p(B)]

    Args:
        q_ib:       Joint BivariateLogNormal posterior, batch_shape=[B].
        pi_cfg:     PriorConfig for the intensity prior p(I).
        pbg_cfg:    PriorConfig for the background prior p(B).
        pi_params:  Tensor-valued params for p(I).
        pbg_params: Tensor-valued params for p(B).
        device:     Target device.
        mc_samples: Number of MC samples for the KL estimate.
        weight:     KL regularization weight (β in β-VAE).

    Returns:
        Tensor of shape [B] with per-sample KL estimates.
    """
    p_i = _build_prior(pi_cfg, pi_params, device) if (pi_cfg and pi_params) else None
    p_bg = (
        _build_prior(pbg_cfg, pbg_params, device) if (pbg_cfg and pbg_params) else None
    )

    if p_i is None and p_bg is None:
        return torch.zeros(q_ib.batch_shape, device=device)

    samples_ib = q_ib.rsample([mc_samples])  # [S, B, 2]
    log_q = q_ib.log_prob(samples_ib)  # [S, B]

    log_p = torch.zeros_like(log_q)
    if p_i is not None:
        log_p = log_p + p_i.log_prob(samples_ib[..., 0])  # [S, B]
    if p_bg is not None:
        log_p = log_p + p_bg.log_prob(samples_ib[..., 1])  # [S, B]

    return (log_q - log_p).mean(dim=0) * weight


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

        # Always set attributes so forward() can do unconditional `is not None` checks.
        self.pi_cfg = pi_cfg
        self.pi_params = _params_as_tensors(pi_cfg) if pi_cfg is not None else None

        self.pbg_cfg = pbg_cfg
        self.pbg_params = _params_as_tensors(pbg_cfg) if pbg_cfg is not None else None

        self.pprf_cfg = pprf_cfg
        self.pprf_params = _get_dirichlet_prior(pprf_cfg) if pprf_cfg is not None else None

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution,
        mask: Tensor,
        qi: Distribution | None = None,
        qbg: Distribution | None = None,
        q_ib: Distribution | None = None,
    ):
        """Compute the ELBO loss.

        Supports two modes for the (I, B) posterior:
          - Independent:  pass ``qi`` and ``qbg`` (original mean-field).
          - Joint:        pass ``q_ib`` (BivariateLogNormal); ``qi``/``qbg``
                          are then ignored and the joint KL is computed instead.
        """
        # batch metadata
        device = rate.device
        batch_size = rate.shape[0]

        # moving data to device
        counts = counts.to(device)
        mask = mask.to(device)

        # total KL and per-prior KLs
        kl = torch.zeros(batch_size, device=device)
        kl_prf = torch.zeros(batch_size, device=device)
        kl_i = torch.zeros(batch_size, device=device)
        kl_bg = torch.zeros(batch_size, device=device)

        # Profile prior KL (always independent of the I/B mode)
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

        if q_ib is not None:
            # Joint (I, B) mode: single KL term for the bivariate posterior.
            # The joint KL weight is taken from pi_cfg; pbg_cfg.weight is ignored.
            weight = self.pi_cfg.weight if self.pi_cfg is not None else 1.0
            kl_i = _joint_prior_kl(
                q_ib=q_ib,
                pi_cfg=self.pi_cfg,
                pbg_cfg=self.pbg_cfg,
                pi_params=self.pi_params,
                pbg_params=self.pbg_params,
                device=device,
                mc_samples=self.mc_samples,
                weight=weight,
            )
            kl += kl_i
        else:
            # Independent mean-field mode: separate KL for qi and qbg.
            if self.pi_cfg is not None and self.pi_params is not None and qi is not None:
                kl_i = _prior_kl(
                    prior_cfg=self.pi_cfg,
                    q=qi,
                    params=self.pi_params,
                    weight=self.pi_cfg.weight,
                    device=device,
                    mc_samples=self.mc_samples,
                )
                kl += kl_i

            if (
                self.pbg_cfg is not None
                and self.pbg_params is not None
                and qbg is not None
            ):
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

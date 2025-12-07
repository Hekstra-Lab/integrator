from dataclasses import dataclass
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


PRIOR_MAP = {
    "gamma": Gamma,
    "log_normal": LogNormal,
    "half_normal": HalfNormal,
    "half_cauchy": HalfCauchy,
    "exponential": Exponential,
    "dirichlet": Dirichlet,
}


@dataclass
class PriorConfig:
    name: str
    params: dict[str, float | str]
    weight: float


@dataclass
class LossConfig:
    pprf: PriorConfig | None
    pi: PriorConfig | None
    pbg: PriorConfig | None
    shape: tuple = (3, 21, 21)
    mc_smpls: int = 100
    eps: float = 1e-6


def get_dirichlet_prior(
    prior: PriorConfig | None,
    shape: tuple[int, ...],
) -> dict[str, torch.Tensor] | None:
    if prior is None:
        return None

    conc = prior.params.get("concentration")

    if isinstance(conc, float) or isinstance(conc, int):
        K = prod(shape)
        return {"concentration": torch.full((K,), float(conc))}

    if isinstance(conc, str):
        loaded = torch.load(conc)
        return {"concentration": loaded.reshape(-1)}

    if isinstance(conc, torch.Tensor):
        return {"concentration": conc.reshape(-1)}

    raise TypeError(f"Unsupported concentration type: {type(conc)}")


def _params_as_tensors(
    prior: PriorConfig | None,
) -> dict[str, torch.Tensor] | None:
    if prior is None or prior.params is None:
        return None
    return {k: torch.tensor(v) for k, v in prior.params.items()}


def _build_prior(prior, params, device):
    Dist = PRIOR_MAP.get(prior.name)
    if Dist is None:
        raise ValueError(f"Unknown prior name: {prior.name}")

    return Dist(**{k: v.to(device) for k, v in params.items()})


def _kl(
    q: Distribution,
    p: Distribution,
    cfg: LossConfig,
):
    try:
        return torch.distributions.kl.kl_divergence(q, p)
    except NotImplementedError:
        samples = q.rsample(torch.Size([cfg.mc_smpls]))
        log_q = q.log_prob(samples)
        log_p = p.log_prob(samples)
        return (log_q - log_p).mean(dim=0)


def _prior_kl(
    prior: PriorConfig,
    q: Distribution,
    params: dict[str, torch.Tensor],
    weight: float,
    device: torch.device,
    cfg: LossConfig,
) -> torch.Tensor:
    p = _build_prior(
        prior,
        params,
        device,
    )
    kl_prior = _kl(q, p, cfg)
    return kl_prior * weight


class Loss(nn.Module):
    def __init__(
        self,
        cfg: LossConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.eps = cfg.eps

        if cfg.pi is not None:
            self.pi_params = _params_as_tensors(cfg.pi)
        if cfg.pbg is not None:
            self.bg_params = _params_as_tensors(cfg.pbg)
        if cfg.pprf is not None:
            self.prf_params = get_dirichlet_prior(
                cfg.pprf,
                cfg.shape,
            )

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

        # calculating per prior KL temrms

        if self.cfg.pprf is not None and self.prf_params is not None:
            kl_prf = _prior_kl(
                prior=self.cfg.pprf,
                q=qp,
                params=self.prf_params,
                weight=self.cfg.pprf.weight,
                device=device,
                cfg=self.cfg,
            )
            kl += kl_prf

        if self.cfg.pi is not None and self.pi_params is not None:
            kl_i = _prior_kl(
                prior=self.cfg.pi,
                q=qi,
                params=self.pi_params,
                weight=self.cfg.pi.weight,
                device=device,
                cfg=self.cfg,
            )
            kl += kl_i

        if self.cfg.pbg is not None and self.bg_params is not None:
            kl_bg = _prior_kl(
                prior=self.cfg.pbg,
                q=qbg,
                params=self.bg_params,
                weight=self.cfg.pbg.weight,
                device=device,
                cfg=self.cfg,
            )
            kl += kl_bg

        # Calculating log likelihood
        ll = Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        neg_ll = (-ll_mean).sum(1)

        # Total loss
        loss = (neg_ll + kl).mean()

        print("Loss module:")
        print("min nll", neg_ll.min())
        print("max nll", neg_ll.max())
        print("max nll", neg_ll.mean())

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }


# %%
if __name__ == "__main__":
    import torch

    from integrator.model.distributions import (
        DirichletDistribution,
        FoldedNormalDistribution,
    )
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        load_config,
    )
    from utils import CONFIGS

    cfg = list(CONFIGS.glob("*"))[0]
    cfg = load_config(cfg)

    integrator = create_integrator(cfg)
    data = create_data_loader(cfg)

    # hyperparameters
    mc_samples = 100

    # distributions
    qbg_ = FoldedNormalDistribution(in_features=64)
    qi_ = FoldedNormalDistribution(in_features=64)
    qp_ = DirichletDistribution(in_features=64, out_features=(3, 21, 21))

    # load a batch
    counts, sbox, mask, meta = next(iter(data.train_dataloader()))

    shoebox_rep = integrator.encoder1(
        sbox.reshape(sbox.shape[0], 1, 3, 21, 21)
    )
    intensity_rep = integrator.encoder2(
        sbox.reshape(sbox.shape[0], 1, 3, 21, 21)
    )

    # get distributinos
    qbg = qbg_(intensity_rep)
    qi = qi_(intensity_rep)
    qp = qp_(shoebox_rep)

    # get samples
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zprf = qp.rsample([mc_samples]).permute(1, 0, 2)
    zi = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)

    rate = zi * zprf + zbg  # [B,S,Pix]

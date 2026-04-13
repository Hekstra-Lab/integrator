from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from integrator.configs.integrator import IntegratorCfg
from integrator.model.distributions.logistic_normal import ProfilePosterior


@dataclass
class IntegratorBaseOutputs:
    rates: Tensor
    counts: Tensor
    mask: Tensor
    qbg: Any
    qp: Any
    qi: Any
    zp: Tensor
    zbg: Tensor
    metadata: dict[str, torch.Tensor]
    concentration: Tensor | None = None


@dataclass
class IntegratorModelArgs:
    cfg: IntegratorCfg
    loss: nn.Module
    surrogates: dict[str, nn.Module]
    encoders: dict[str, nn.Module]


def _assemble_outputs(
    out: IntegratorBaseOutputs,
) -> dict[str, Any]:
    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": out.qp.mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": out.qp.mean,
        "concentration": out.concentration,
    }

    if isinstance(out.qp, ProfilePosterior):
        base["qp_mu_h"] = out.qp.mu_h  # (B, d) posterior mean of h
        base["qp_std_h"] = out.qp.std_h  # (B, d) posterior std of h

    if out.metadata is None:
        return base

    # Storing the surrogate distribution parameters
    distribution_params = {
        "qbg_params": {
            name: getattr(out.qbg, name) for name in out.qbg.arg_constraints
        },
        "qi_params": {
            name: getattr(out.qi, name) for name in out.qi.arg_constraints
        },
    }

    # Update base dictionary
    base.update(out.metadata)
    base.update(distribution_params)

    return base

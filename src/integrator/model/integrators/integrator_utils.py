from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)


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


def _assemble_outputs(
    out: IntegratorBaseOutputs,
) -> dict[str, Any]:
    is_profile_output = isinstance(out.qp, ProfileSurrogateOutput)

    qp_mean = out.qp.mean_profile if is_profile_output else out.qp.mean

    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": qp_mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": qp_mean,
    }

    if is_profile_output:
        base["qp_loc"] = out.qp.loc
        base["qp_scale"] = out.qp.scale

    if out.metadata is None:
        return base

    distribution_params = {
        "qbg_params": {
            name: getattr(out.qbg, name) for name in out.qbg.arg_constraints
        },
        "qi_params": {
            name: getattr(out.qi, name) for name in out.qi.arg_constraints
        },
    }
    # qp is a Distribution
    if not is_profile_output:
        distribution_params["qp_params"] = {
            name: getattr(out.qp, name) for name in out.qp.arg_constraints
        }

    base.update(out.metadata)
    base.update(distribution_params)

    return base

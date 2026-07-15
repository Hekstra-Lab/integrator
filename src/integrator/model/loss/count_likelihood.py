import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import NegativeBinomial, Poisson

_RATE_FLOOR = 1e-12

POISSON = "poisson"
NEGATIVE_BINOMIAL = "negative_binomial"
_VALID = (POISSON, NEGATIVE_BINOMIAL)

_GLOBAL = "global"
_PER_BIN = "per_bin"
_VALID_SCOPE = (_GLOBAL, _PER_BIN)


def _inv_softplus(y: Tensor) -> Tensor:
    """Inverse of softplus; stable for large y where softplus is near-identity."""
    return y + torch.log(-torch.expm1(-y))


class CountLikelihood(nn.Module):
    """Per-pixel count likelihood: Poisson or Negative Binomial.

    Args:
        name: `poisson` (default) or `negative_binomial`.
        dispersion_init: Initial Negative Binomial dispersion r (torch
            `total_count`); larger is closer to Poisson. Ignored for Poisson.
        dispersion_scope: `global` (one shared r) or `per_bin` (one r per
            resolution bin, indexed by `group_labels`). Ignored for Poisson.
        n_bins: Number of resolution bins used when `dispersion_scope` is
            `per_bin`.
        dispersion_floor: Floor added after softplus for numerical stability.
    """

    def __init__(
        self,
        name: str = POISSON,
        *,
        dispersion_init: float = 10.0,
        dispersion_scope: str = _GLOBAL,
        n_bins: int = 1,
        dispersion_floor: float = 1e-3,
    ):
        super().__init__()
        if name not in _VALID:
            raise ValueError(
                f"Unknown likelihood {name!r}; expected one of {_VALID}."
            )
        self.name = name
        self.dispersion_scope = dispersion_scope
        self.dispersion_floor = dispersion_floor

        if name == NEGATIVE_BINOMIAL:
            if dispersion_scope not in _VALID_SCOPE:
                raise ValueError(
                    f"Unknown dispersion_scope {dispersion_scope!r}; "
                    f"expected one of {_VALID_SCOPE}."
                )
            n = n_bins if dispersion_scope == _PER_BIN else 1
            raw_init = _inv_softplus(torch.full((n,), float(dispersion_init)))
            # softplus(raw_dispersion) + floor = dispersion_init at init.
            self.raw_dispersion = nn.Parameter(raw_init)

    def dispersion(self, group_labels: Tensor | None = None) -> Tensor:
        """Return the Negative Binomial dispersion r, shape () or (B,)."""
        r = F.softplus(self.raw_dispersion) + self.dispersion_floor
        if self.dispersion_scope == _PER_BIN:
            if group_labels is None:
                raise ValueError("per_bin dispersion requires group_labels.")
            return r[group_labels.long()]  # (B,)
        return r.squeeze(0)  # ()

    def neg_ll(
        self,
        rate: Tensor,
        counts: Tensor,
        mask: Tensor,
        group_labels: Tensor | None = None,
    ) -> Tensor:
        rate = rate.clamp(min=_RATE_FLOOR)
        target = counts.unsqueeze(1)  # (B, 1, P)

        if self.name == POISSON:
            ll = Poisson(rate).log_prob(target)
        else:
            r = self.dispersion(group_labels).reshape(-1, 1, 1)  # broadcasts
            logits = rate.log() - r.log()
            ll = NegativeBinomial(
                total_count=r, logits=logits, validate_args=False
            ).log_prob(target)

        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        return (-ll_mean).sum(1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import NegativeBinomial, Normal, Poisson

_RATE_FLOOR = 1e-12
_VAR_FLOOR = 1e-8

POISSON = "poisson"
NEGATIVE_BINOMIAL = "negative_binomial"
NORMAL = "normal"
_VALID = (POISSON, NEGATIVE_BINOMIAL, NORMAL)

_GLOBAL = "global"
_PER_BIN = "per_bin"
_VALID_SCOPE = (_GLOBAL, _PER_BIN)

_COUPLED = "coupled"
_FREE = "free"
_VALID_VARIANCE = (_COUPLED, _FREE)


def _inv_softplus(y: Tensor) -> Tensor:
    """Inverse of softplus; stable for large y where softplus is near-identity."""
    return y + torch.log(-torch.expm1(-y))


class CountLikelihood(nn.Module):
    """Per-pixel count likelihood: Poisson, Negative Binomial, or Normal.

    The Normal option is for real-valued detector output (e.g. JUNGFRAU calibrated
    photon-equivalents, which are continuous and can be negative), where a discrete
    count likelihood does not apply. Its variance is either:

    - `coupled`: `Var = gain * rate + read_noise^2`, the variance of a counting
      process of gain `gain` plus Gaussian read noise. `read_noise` is in the same
      units as `rate`; JUNGFRAU G0 is ~0.024. This is the best a Gaussian can do.

      `gain` is counts of variance per count of recorded signal, and the three
      regimes it spans are why this form is worth having over Poisson:

          gain 1, read_noise 0     Poisson's variance; photons, no read noise
          gain 1, read_noise > 0   a real detector: shot noise plus a floor
          gain 0, read_noise > 0   no counting statistics at all, only a floor

      The last is not hypothetical. The nanoBragg DHFR Laue simulation measures
      a variance of 9.2 counts^2 that does not move while the local mean runs
      13 -> 31 counts/px, a signal gain of -0.046 +/- 0.05, a skewness of
      +0.0011 where Poisson needs +0.2473, and a minimum count of -2, which a
      Poisson image cannot produce. Its sub-images were averaged into a smooth
      expectation and only read noise was added. Poisson on that data assigns a
      bright pixel at rate 500 a variance of 500 against a true 9.2, so it
      down-weights by 54x exactly the pixels that carry the intensity.

      Neither of the discrete alternatives can represent this. Negative binomial
      has Var = rate + rate^2/r >= rate for every r, so it only goes ABOVE
      Poisson. A Gamma of fixed shape k has Var = rate^2/k, a constant relative
      error, which grows fastest where this needs to stay flat; a Gamma of fixed
      rate gives Var proportional to rate, which is the gain alone and cannot
      hold a floor that survives at zero signal. A mean-independent additive
      term is what is required, and only a Normal of this form has one.
    - `free`: a single learned `sigma`, ignoring the mean-variance coupling. This is
      the naive baseline; it is over-confident on bright pixels because one variance
      cannot describe both the quiet background and the noisy peak.

    Args:
        name: `poisson` (default), `negative_binomial`, or `normal`.
        dispersion_init: Initial Negative Binomial dispersion r (torch
            `total_count`); larger is closer to Poisson. Ignored unless NB.
        dispersion_scope: `global` (one shared r) or `per_bin` (one r per
            resolution bin, indexed by `group_labels`). Ignored unless NB.
        n_bins: Number of resolution bins used when `dispersion_scope` is
            `per_bin`.
        dispersion_floor: Floor added after softplus for numerical stability.
        learn_dispersion: If True (default) the dispersion r is a learned
            `nn.Parameter`; if False it is a fixed buffer held at `dispersion_init`.
        read_noise: Gaussian read noise, in `rate` units, for the `coupled` Normal.
            Ignored unless `name == "normal"`.
        gain: Counts of variance per count of recorded signal, for the `coupled`
            Normal. Defaults to 1.0, which is the Poisson coupling and leaves
            every existing configuration bit-identical. Ignored unless
            `name == "normal"` and `variance == "coupled"`.
        variance: `coupled` or `free`; the Normal variance model (see above).
        sigma_init: Initial sigma for the `free` Normal.
        learn_sigma: If True (default) the `free` Normal sigma is a learned
            `nn.Parameter`; if False a fixed buffer at `sigma_init`.
    """

    def __init__(
        self,
        name: str = POISSON,
        *,
        dispersion_init: float = 10.0,
        dispersion_scope: str = _GLOBAL,
        n_bins: int = 1,
        dispersion_floor: float = 1e-3,
        learn_dispersion: bool = True,
        read_noise: float = 0.0,
        gain: float = 1.0,
        variance: str = _COUPLED,
        sigma_init: float = 1.0,
        learn_sigma: bool = True,
    ):
        super().__init__()
        if name not in _VALID:
            raise ValueError(
                f"Unknown likelihood {name!r}; expected one of {_VALID}."
            )
        self.name = name
        self.dispersion_scope = dispersion_scope
        self.dispersion_floor = dispersion_floor
        self.learn_dispersion = learn_dispersion

        if name == NEGATIVE_BINOMIAL:
            if dispersion_scope not in _VALID_SCOPE:
                raise ValueError(
                    f"Unknown dispersion_scope {dispersion_scope!r}; "
                    f"expected one of {_VALID_SCOPE}."
                )
            n = n_bins if dispersion_scope == _PER_BIN else 1
            raw_init = _inv_softplus(torch.full((n,), float(dispersion_init)))
            # softplus(raw_dispersion) + floor = dispersion_init at init.
            if learn_dispersion:
                self.raw_dispersion = nn.Parameter(raw_init)
            else:
                # fixed r: a persistent buffer, so it is saved but never trained.
                self.register_buffer("raw_dispersion", raw_init)

        if name == NORMAL:
            if variance not in _VALID_VARIANCE:
                raise ValueError(
                    f"Unknown variance {variance!r}; expected one of "
                    f"{_VALID_VARIANCE}."
                )
            self.variance = variance
            # read_noise is a fixed config constant; non-persistent keeps it out of
            # the state_dict, so a coupled Normal (like Poisson) adds no saved state.
            self.register_buffer(
                "read_noise",
                torch.as_tensor(float(read_noise)),
                persistent=False,
            )
            if gain < 0.0:
                raise ValueError(f"gain must be >= 0; got {gain}")
            self.register_buffer(
                "gain", torch.as_tensor(float(gain)), persistent=False
            )
            if variance == _FREE:
                raw_init = _inv_softplus(torch.tensor(float(sigma_init)))
                if learn_sigma:
                    self.raw_sigma = nn.Parameter(raw_init)
                else:
                    self.register_buffer("raw_sigma", raw_init)

    def dispersion(self, group_labels: Tensor | None = None) -> Tensor:
        """Return the Negative Binomial dispersion r, shape () or (B,)."""
        r = F.softplus(self.raw_dispersion) + self.dispersion_floor
        if self.dispersion_scope == _PER_BIN:
            if group_labels is None:
                raise ValueError("per_bin dispersion requires group_labels.")
            return r[group_labels.long()]  # (B,)
        return r.squeeze(0)  # ()

    def sigma(self) -> Tensor:
        """Learned sigma of the `free` Normal, shape ()."""
        return F.softplus(self.raw_sigma) + self.dispersion_floor

    def diagnostics(self) -> dict[str, Tensor]:
        """Scalar diagnostics to log; empty for Poisson."""
        if self.name == NEGATIVE_BINOMIAL:
            r = F.softplus(self.raw_dispersion) + self.dispersion_floor
            return {"nb_dispersion": r.mean().detach()}
        if self.name == NORMAL:
            if self.variance == _FREE:
                return {"normal_sigma": self.sigma().detach()}
            return {
                "normal_read_noise": self.read_noise.detach(),
                "normal_gain": self.gain.detach(),
            }
        return {}

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
        elif self.name == NEGATIVE_BINOMIAL:
            r = self.dispersion(group_labels).reshape(-1, 1, 1)  # broadcasts
            logits = rate.log() - r.log()
            ll = NegativeBinomial(
                total_count=r, logits=logits, validate_args=False
            ).log_prob(target)
        else:  # NORMAL
            if self.variance == _COUPLED:
                # Var = gain * rate + read_noise^2: a counting process of this
                # gain plus Gaussian read noise. gain = 1 is the Poisson coupling.
                std = (
                    (self.gain * rate + self.read_noise**2)
                    .clamp_min(_VAR_FLOOR)
                    .sqrt()
                )
            else:  # free: one learned sigma for every pixel
                std = self.sigma()
            ll = Normal(rate, std).log_prob(target)

        ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
        return (-ll_mean).sum(1)

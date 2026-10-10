import logging
import math
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import (
    Distribution,
    Gamma,
    kl_divergence,
)

from integrator.model.distributions.background_plane import (
    compute_background_plane_kl,
)
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogateOutput,
)
from integrator.model.loss.count_likelihood import (
    CountLikelihood,
    _inv_softplus,
)
from integrator.model.loss.kl_helpers import compute_profile_kl

_DEFAULT_PROFILE_PRIOR_SCALE = 3.0


logger = logging.getLogger(__name__)


class WilsonLoss(nn.Module):
    """Base ELBO loss with Wilson intensity prior.

    Subclasses implement `_get_tau` to define how the Wilson prior rate
    is computed (scalar G for monochromatic, G(lambda) for polychromatic).
    """

    def __init__(
        self,
        *,
        # Background Gamma prior: scalar, or per-resolution-bin prior
        bg_rate: float | list[float] = 1.0,
        bg_concentration: float | list[float] = 1.0,
        # B factor: B = softplus(raw_B) + b_min, so init_B is the physical B
        # (A^2) at init -- floored at b_min, unbounded above.
        init_B: float = 30.0,
        b_min: float = 0.0,
        # Scale stabilization (DIALS-free): log-space G + a one-time Wilson-plot
        # scale init from raw counts. Off by default (legacy behavior,
        # checkpoint-compatible). See _maybe_init_scale / MonochromaticWilsonLoss.
        stabilize_scale: bool = False,
        learn_B: bool = True,
        init_scale_from_counts: bool = True,
        wilson_init_bins: int = 20,
        # Periodic re-estimation of (G, B) from raw counts. 0 disables, which
        # is the default so existing configs are unchanged.
        refit_prior_every_n_epochs: int = 0,
        refit_bins: int = 30,
        # shoebox shape, needed to place the foreground disc
        shoebox_dhw: tuple[int, int, int] | None = None,
        refit_fg_radius: int = 5,
        refit_guard: int = 3,
        # per-bin correction to the fitted Wilson line, shrunk toward zero
        refit_residual: bool = False,
        refit_residual_max: float = 1.5,
        refit_start_epoch: int = 1,
        refit_damping: float = 0.5,
        # bins below this resolution are excluded from the SLOPE: the Wilson
        # plot is not linear through the solvent region
        wilson_fit_d_max: float = 3.5,
        # Resolution bins for per-bin background prior
        n_bins: int = 1,
        # Pixel count likelihood: "poisson" (default) or "negative_binomial"
        likelihood: str = "poisson",
        nb_dispersion_init: float = 10.0,
        nb_dispersion_scope: str = "global",
        nb_dispersion_floor: float = 1e-3,
        nb_learn_dispersion: bool = True,
        # Prior configs from yaml
        pi_cfg=None,
        pbg_cfg=None,
        pprf_cfg=None,
        # KL weights
        profile_kl_weight: float = 1.0,
        background_kl_weight: float = 1.0,
        background_plane_kl_weight: float = 1.0,
        intensity_kl_weight: float = 1.0,
        # Self-calibrating, per-resolution-bin intensity KL weight. Off by
        # default: a scalar intensity_kl_weight, exactly the legacy behaviour.
        # When enabled (needs n_bins > 1) each shell holds its own weight,
        # recalibrated at epoch end so the posterior acentric coefficient of
        # variation sd/mean matches `intensity_kl_target` -- the Wilson moment
        # the data should show (acentric sd/mean = 1). A too-strong prior
        # collapses sd/mean below the target in the weak high-resolution shells;
        # the controller relaxes just those shells until it recovers, so no
        # single global weight has to be swept per dataset.
        intensity_kl_calibrate: bool = False,
        intensity_kl_target: float = 1.0,
        # Temper the intensity prior by the inverse multiplicity. The Wilson
        # prior belongs to one unique reflection, but it is applied here once
        # per OBSERVATION, so with N observations of a reflection it enters N
        # times. Raising it to the power 1/N makes the product over
        # observations telescope back to a single application,
        #     prod_i [ p(I)^(1/N) L_i(I) ] = p(I) prod_i L_i(I),
        # and since the KL weight is the prior's exponent in the ELBO that is
        # exactly a per-reflection weight of 1/N. Needs a `multiplicity` column
        # in the metadata (scripts/add_multiplicity.py); only the INTENSITY KL
        # is tempered, because the background and profile are genuine
        # per-observation nuisance parameters and are not over-counted.
        kl_per_multiplicity: bool = False,
        intensity_kl_gain: float = 0.7,
        intensity_kl_min: float = 0.02,
        intensity_kl_max: float = 1.0,
        # Coset auxiliary penalty, used when coset_mode == "aux": pushes the
        # predicted intensity of background-only reflections toward a floor.
        # Zero weight leaves it inert, which is the default.
        eps: float = 1e-6,
        coset_aux_weight: float = 0.0,
        coset_i_floor: float = 1.0,
    ):
        super().__init__()
        # warn once, not once per batch, if the dataset predates the flag
        self._warned_centric = False
        self.eps = eps
        # the merging branch optionally fits spatially varying profile and
        # background priors; those models are not carried over, so the
        # attributes exist and are inert
        self.profile_prior = None
        self.profile_prior_scale = _DEFAULT_PROFILE_PRIOR_SCALE
        self.bg_prior = None
        self.coset_aux_weight = coset_aux_weight
        self.coset_i_floor = coset_i_floor
        self.b_min = b_min  # minimum B-factor
        self.n_bins = n_bins
        self.stabilize_scale = stabilize_scale
        self.learn_B = learn_B
        self.refit_every = int(refit_prior_every_n_epochs)
        self.refit_bins = int(refit_bins)
        self.shoebox_dhw = tuple(shoebox_dhw) if shoebox_dhw else None
        self.refit_fg_radius = int(refit_fg_radius)
        self.refit_guard = int(refit_guard)
        self.refit_residual = bool(refit_residual)
        self.refit_residual_max = float(refit_residual_max)
        self._fg_mask = None
        self._ann_mask = None
        self.refit_start_epoch = int(refit_start_epoch)
        self.refit_damping = float(refit_damping)
        self.wilson_fit_d_max = float(wilson_fit_d_max)
        self.refit_enabled = self.refit_every > 0
        self._last_fit: dict[str, float] = {}
        self.init_scale_from_counts = init_scale_from_counts
        self.wilson_init_bins = wilson_init_bins
        self.register_buffer(
            "bg_concentration",
            torch.as_tensor(bg_concentration, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "bg_rate",
            torch.as_tensor(bg_rate, dtype=torch.float32),
            persistent=False,
        )
        # keep the prior cfgs so run artifacts can record them
        self.pprf_cfg = pprf_cfg
        self.pbg_cfg = pbg_cfg
        self.pi_cfg = pi_cfg

        self.profile_kl_weight = (
            pprf_cfg.weight if pprf_cfg is not None else profile_kl_weight
        )
        self.background_kl_weight = (
            pbg_cfg.weight if pbg_cfg is not None else background_kl_weight
        )
        self.background_plane_kl_weight = background_plane_kl_weight
        self.intensity_kl_weight = (
            pi_cfg.weight if pi_cfg is not None else intensity_kl_weight
        )

        # The merging branch spells these pi_/pbg_/pprf_weight: poly_wilson_loss
        # multiplies by self.pi_weight, and HierarchicalSVAEIntegrator checks it
        # to refuse double-counting the intensity KL. One value, two names.
        self.pi_weight = self.intensity_kl_weight
        self.pbg_weight = self.background_kl_weight
        self.pprf_weight = self.profile_kl_weight

        self.kl_per_multiplicity = bool(kl_per_multiplicity)
        self._warned_multiplicity = False

        # Per-resolution-bin intensity KL weight, self-calibrated at epoch end.
        # Only active with more than one bin; otherwise the scalar above is used
        # and the loss behaves exactly as before.
        self.intensity_kl_target = float(intensity_kl_target)
        self.intensity_kl_gain = float(intensity_kl_gain)
        self.intensity_kl_min = float(intensity_kl_min)
        self.intensity_kl_max = float(intensity_kl_max)
        self.intensity_kl_calibrate = bool(intensity_kl_calibrate) and n_bins > 1
        # Persistent only when calibration is on. With it off every entry is
        # just the scalar intensity_kl_weight, so checkpointing it adds state
        # that carries no information and makes checkpoints from before this
        # feature fail a strict load in both directions. Calibrated runs do
        # need it in the checkpoint, since the per-bin weights are learned.
        self.register_buffer(
            "intensity_kl_bins",
            torch.full((n_bins,), float(self.intensity_kl_weight)),
            persistent=self.intensity_kl_calibrate,
        )
        if self.intensity_kl_calibrate:
            # per-bin sums of the posterior mean intensity over an epoch's
            # acentric reflections, reduced to sd/mean at its end. float64 so a
            # 2M-reflection epoch of squared intensities does not lose precision.
            for name in ("_ikl_n", "_ikl_mu", "_ikl_mu2"):
                self.register_buffer(
                    name, torch.zeros(n_bins, dtype=torch.float64), persistent=True
                )

        # invert B = softplus(raw_B) + b_min so that B == init_B at init
        y = torch.tensor(max(float(init_B) - b_min, 1e-3))
        self.raw_B = nn.Parameter(_inv_softplus(y))
        if not learn_B:
            # B is well determined by a Wilson plot, and learning it jointly
            # is not reliable: on SBGrid 845 it collapsed 47 -> 1.5 and on 821
            # 12 -> 4.0, against classical fits of 22-44 A^2 in every lp
            # frame. A flattened prior costs little where the likelihood is
            # strong, and destroys the weak shells where it is not.
            # Kept as a Parameter so checkpoints stay compatible.
            self.raw_B.requires_grad_(False)
        if stabilize_scale:
            self.register_buffer(
                "scale_initialized", torch.tensor(False), persistent=True
            )
        if self.refit_enabled:
            if not stabilize_scale:
                raise ValueError(
                    "refit_prior_every_n_epochs needs stabilize_scale: true. "
                    "In the softplus-G frame a raw-space step moves G by about "
                    "one count, so G cannot travel to a refit value; log-space "
                    "makes the step multiplicative."
                )
            # per-bin sums, accumulated over an epoch and solved at its end. A
            # single batch is far too noisy for a slope; a whole epoch is not.
            if self.refit_bins < 10:
                raise ValueError(
                    f"refit_bins={self.refit_bins} is too few to fit a slope; "
                    "use at least 10"
                )
            for name in ("_fit_n", "_fit_i", "_fit_i2", "_fit_s2", "_fit_qi", "_fit_qi2"):
                self.register_buffer(
                    name, torch.zeros(self.refit_bins), persistent=True
                )
            # log-space correction to the fitted line, one value per bin, and
            # the bin edges needed to look it up
            self.register_buffer(
                "_resid", torch.zeros(self.refit_bins), persistent=True
            )
            # the s^2 range is a property of the dataset, learned from the
            # first batch and then held fixed so bins mean the same thing in
            # every epoch
            self.register_buffer("_fit_s2_max", torch.tensor(0.0), persistent=True)

        self.count_likelihood = CountLikelihood(
            likelihood,
            dispersion_init=nb_dispersion_init,
            dispersion_scope=nb_dispersion_scope,
            n_bins=n_bins,
            dispersion_floor=nb_dispersion_floor,
            learn_dispersion=nb_learn_dispersion,
        )

    def diagnostics(self) -> dict[str, Tensor]:
        """Scalars to log each step: the Wilson B-factor + likelihood diagnostics."""
        out = {
            "wilson_B": self.get_B().detach(),
            **self.count_likelihood.diagnostics(),
        }
        # what the count-evidence fit last said, so a trace shows at a glance
        # whether the prior is tracking the data or drifting from it
        for key, value in self._last_fit.items():
            out[key] = torch.tensor(float(value))
        return out

    def get_B(self) -> Tensor:
        return F.softplus(self.raw_B) + self.b_min

    @abstractmethod
    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        """Compute Wilson prior rate tau per reflection."""

    def _set_scale_from_fit(self, G0: float) -> None:
        """Set the overall scale from a data-driven estimate. Base: no-op.

        Overridden by MonochromaticWilsonLoss, whose scale is a single scalar G.
        """

    @staticmethod
    def wilson_fit(
        i_hat: Tensor, s_sq: Tensor, n_bins: int = 20
    ) -> tuple[float, float]:
        """Binned Wilson-plot fit of log(mean I per bin) vs s^2, giving (G, B).

        The line is `log<I> = log G - 2 B s^2`, so G is the intercept and B is
        -slope/2. Bins are equal-count in s^2, and each bin contributes its MEAN
        intensity: a per-reflection log-mean is biased low by the
        Euler-Mascheroni constant for exponentially distributed intensities.
        """
        valid = i_hat > 0
        i_hat, s2 = i_hat[valid], s_sq[valid]
        if i_hat.numel() < 3 * n_bins:
            return (float(i_hat.mean()) if i_hat.numel() else 1.0), 0.0
        order = torch.argsort(s2)
        s2s, is_ = s2[order], i_hat[order]
        idx = torch.linspace(0, len(s2s), n_bins + 1).long()
        xb, yb = [], []
        for i in range(n_bins):
            lo, hi = int(idx[i]), int(idx[i + 1])
            if hi - lo < 3:
                continue
            xb.append(s2s[lo:hi].mean())
            yb.append(torch.log(is_[lo:hi].mean().clamp_min(1e-6)))
        x, y = torch.stack(xb), torch.stack(yb)
        xm, ym = x.mean(), y.mean()
        slope = ((x - xm) * (y - ym)).sum() / (x - xm).pow(2).sum().clamp_min(
            1e-12
        )
        return float(torch.exp(ym - slope * xm)), float(-slope / 2.0)

    @staticmethod
    def wilson_fit_binned(
        n: Tensor, sum_i: Tensor, sum_i2: Tensor, sum_s2: Tensor,
        d_max: float = 3.5, min_count: int = 30,
    ) -> tuple[float, float, float, int]:
        """(G, B, weighted R^2, bins used) from per-bin sums of raw counts.

        Regresses log(mean I per bin) on s^2, the same line as `wilson_fit`,
        but from streaming sums and with three differences that matter once B
        is taken from the slope and not just G from the intercept:

        Negative bin members are KEPT. `wilson_fit` drops reflections with
        i_hat <= 0, which at a bin I/sigma near 0.8 inflates the bin mean by
        roughly 46% -- and only in the weak outer bins, so it tilts the line.
        Over an s^2 lever arm of ~0.05 that is 3-4 A^2 of B, biased low,
        exactly where B matters. The mean of noisy unbiased estimates is
        unbiased; the truncation is what breaks it.

        Bins below `d_max` are dropped from the fit: the Wilson plot is not
        linear through the solvent region.

        Bins are weighted by n / (1 + Var/mean^2). For clean exponential
        intensities that is uniform over equal-count bins; where background
        subtraction dominates it discounts the noise.
        """
        keep = n >= min_count
        if d_max is not None:
            # s^2 = 1/(4 d^2), so d > d_max is s^2 below this
            keep = keep & (sum_s2 / n.clamp_min(1) > 1.0 / (4.0 * d_max**2))
        if int(keep.sum()) < 3:
            return float("nan"), float("nan"), 0.0, int(keep.sum())

        nb = n[keep].clamp_min(1.0)
        mean = sum_i[keep] / nb
        var = (sum_i2[keep] / nb - mean.pow(2)).clamp_min(0.0)
        x = sum_s2[keep] / nb
        # a negative bin mean means the background subtraction is biased.
        # Dropping those bins silently deletes the evidence of the problem and
        # tilts the line, so fail instead and let the sanity gate skip.
        if bool((mean <= 0).any()):
            return float("nan"), float("nan"), 0.0, 0
        y = torch.log(mean)

        w = nb / (1.0 + var / mean.pow(2).clamp_min(1e-12))
        w = w / w.sum().clamp_min(1e-12)
        xm = (w * x).sum()
        ym = (w * y).sum()
        sxx = (w * (x - xm).pow(2)).sum().clamp_min(1e-12)
        slope = (w * (x - xm) * (y - ym)).sum() / sxx
        resid = y - (ym + slope * (x - xm))
        ss_tot = (w * (y - ym).pow(2)).sum().clamp_min(1e-12)
        r2 = float(1.0 - (w * resid.pow(2)).sum() / ss_tot)
        return (
            float(torch.exp(ym - slope * xm)),
            float(-slope / 2.0),
            r2,
            int(len(y)),
        )

    def _region_masks(self, n_pixels: int, device):
        """Flat boolean masks for the foreground disc and its background annulus.

        Built once. The disc is a fixed in-plane radius across every z slice --
        the shoebox is centred on the predicted position, so a concentric disc
        captures the peak while the annulus beyond a guard ring sees only
        background (profile leak is ~1e-4 of the mass at these radii).
        """
        if self._fg_mask is not None and self._fg_mask.numel() == n_pixels:
            return self._fg_mask, self._ann_mask
        if not self.shoebox_dhw:
            if not getattr(self, "_warned_dhw", False):
                logger.warning(
                    "refit needs shoebox_dhw to place the foreground disc; "
                    "skipping accumulation"
                )
                self._warned_dhw = True
            return None, None
        d, h, w = self.shoebox_dhw
        if d * h * w != n_pixels:
            raise ValueError(
                f"shoebox_dhw {self.shoebox_dhw} is {d * h * w} pixels but the "
                f"batch has {n_pixels}"
            )
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        r = torch.sqrt((yy - (h - 1) / 2).pow(2) + (xx - (w - 1) / 2).pow(2))
        fg2d = r <= self.refit_fg_radius
        ann2d = r > self.refit_fg_radius + self.refit_guard
        self._fg_mask = fg2d.reshape(1, -1).expand(d, -1).reshape(-1)
        self._ann_mask = ann2d.reshape(1, -1).expand(d, -1).reshape(-1)
        logger.info(
            "refit regions: %d foreground px, %d annulus px of %d",
            int(self._fg_mask.sum()), int(self._ann_mask.sum()), n_pixels,
        )
        return self._fg_mask, self._ann_mask

    @staticmethod
    def _trimmed_mean(counts, valid, n_valid):
        """Annulus mean with the bright tail trimmed.

        A neighbouring spot clipping the annulus would otherwise raise the
        background and eat real signal. A mean over values below the 90th
        percentile is enough; a median is a poor choice here because at a
        background of a few counts the Poisson median is badly quantized.
        """
        big = torch.where(valid, counts, torch.zeros_like(counts))
        total = big.sum(-1)
        mean = total / n_valid
        # one pass: drop anything above 3x the mean, which at bg ~ 2.5 counts
        # keeps the Poisson bulk and removes neighbour contamination
        cut = (3.0 * mean + 3.0).unsqueeze(-1)
        keep = valid & (counts <= cut)
        n_keep = keep.sum(-1).clamp_min(1.0)
        return torch.where(counts <= cut, counts, torch.zeros_like(counts)).mul(
            keep
        ).sum(-1) / n_keep

    def _accumulate_fit(
        self, counts, mask, s_sq, qi, metadata, device
    ) -> None:
        """Add this batch's per-bin sums toward the next refit.

        The intensity proxy is summed counts above the model's own background
        posterior -- raw pixels, no DIALS. The posterior is used rather than
        the background prior because it is both more accurate and, since the
        per-bin background prior is fitted offline from a reflection table,
        the more self-contained of the two. It does not close the intensity
        echo loop: q(bg) is pinned by the pixel likelihood, and G and B enter
        only the intensity KL.
        """
        with torch.no_grad():
            m = mask.squeeze(-1) if mask.dim() > 2 else mask
            c = counts.squeeze(-1) if counts.dim() > 2 else counts
            fg, ann = self._region_masks(c.shape[-1], c.device)
            if fg is None:
                return
            # foreground sum minus a background measured in an annulus the
            # signal cannot reach. Deliberately model-free: at the ELBO's
            # stationary point the background solves b = (C - E_q[I])/npix, so
            # subtracting qbg over the whole box returns E_q[I] almost exactly
            # -- the "raw count" proxy was the posterior mean in disguise, and
            # its npix-fold gain on any resolution-correlated qbg error is what
            # tilted the fit. Measuring background off-peak removes both.
            fg_v, ann_v = m & fg, m & ann
            n_fg = fg_v.sum(-1)
            n_ann = ann_v.sum(-1).clamp_min(1.0)
            bg = self._trimmed_mean(c, ann_v, n_ann)
            i_hat = (c * fg_v).sum(-1) - bg * n_fg
            usable = (n_fg > 0) & (ann_v.sum(-1) >= 20)
            if self._apply_lp and "lp" in metadata:
                # with lp_correction the prior mean of the raw counts is
                # G exp(-2Bs^2)/lp, so i_hat*lp is the Wilson-distributed one
                i_hat = i_hat * metadata["lp"].to(device).reshape(-1).clamp(min=1e-8)
            # own binning, equal width in s^2. Borrowing the background
            # prior's labels would tie the refit to a separate feature and
            # silently disable it whenever that prior is a scalar.
            flat_s2 = s_sq.reshape(-1)
            if float(self._fit_s2_max) <= 0:
                self._fit_s2_max.fill_(float(flat_s2.max()) * 1.001)
            nb = self._fit_n.numel()
            idx = (flat_s2 / self._fit_s2_max.clamp_min(1e-12) * nb).long()
            idx = idx.clamp(0, nb - 1)
            keep = usable.float()
            qi_m = qi.mean.detach().reshape(-1)
            self._fit_n.scatter_add_(0, idx, keep)
            self._fit_i.scatter_add_(0, idx, i_hat * keep)
            self._fit_i2.scatter_add_(0, idx, i_hat.pow(2) * keep)
            self._fit_s2.scatter_add_(0, idx, flat_s2 * keep)
            self._fit_qi.scatter_add_(0, idx, qi_m * keep)
            self._fit_qi2.scatter_add_(0, idx, qi_m.pow(2) * keep)

    def refit_prior(self, epoch: int) -> dict[str, float]:
        """Solve for (G, B) from the epoch's sums and take a damped step.

        This is the M-step. Given q, the objective in (log G, B) is convex and
        its exact solution IS this weighted regression, so an iterative
        optimizer on the same fixed point would only be a slower version of
        it. Fitting raw counts rather than E_q[I] is what keeps the prior from
        grading its own homework: in the weak shells q collapses onto p, and a
        fit to q would then confirm whatever p already said.
        """
        out: dict[str, float] = {}
        if not self.refit_enabled or int(self._fit_n.sum()) == 0:
            return out
        G_hat, B_hat, r2, nbins = self.wilson_fit_binned(
            self._fit_n, self._fit_i, self._fit_i2, self._fit_s2,
            d_max=self.wilson_fit_d_max,
        )
        # the same regression on the posterior means: not used to update
        # anything, logged as an echo meter
        _, B_hat_q, _, _ = self.wilson_fit_binned(
            self._fit_n, self._fit_qi, self._fit_qi2, self._fit_s2,
            d_max=self.wilson_fit_d_max,
        )
        out.update(
            wilson_G_fit=G_hat, wilson_B_fit=B_hat, wilson_fit_r2=r2,
            wilson_fit_bins=float(nbins), wilson_B_fit_q=B_hat_q,
            wilson_echo_gap=(B_hat_q - B_hat)
            if (B_hat == B_hat and B_hat_q == B_hat_q) else float("nan"),
        )
        if self.refit_residual:
            out.update(self._solve_residual(G_hat, B_hat))
        self._zero_fit_buffers()

        current_G = float(self.get_G().reshape(-1)[0])
        sane = (
            epoch >= self.refit_start_epoch
            and B_hat == B_hat and G_hat == G_hat  # not NaN
            and r2 > 0.95
            and self.b_min <= B_hat <= 200.0
            and current_G / 10.0 <= G_hat <= current_G * 10.0
        )
        out["wilson_fit_skipped"] = 0.0 if sane else 1.0
        if not sane:
            return out

        # damped in the frame each parameter lives in: multiplicative for G,
        # additive for B
        eta = self.refit_damping
        with torch.no_grad():
            self._set_scale_from_fit(
                float(torch.exp(
                    (1 - eta) * torch.log(torch.tensor(max(current_G, 1e-8)))
                    + eta * torch.log(torch.tensor(max(G_hat, 1e-8)))
                ))
            )
            target_B = max(B_hat, self.b_min + 1e-3)
            new_B = (1 - eta) * float(self.get_B()) + eta * target_B
            y = torch.tensor(max(new_B - self.b_min, 1e-3))
            self.raw_B.copy_(_inv_softplus(y))
        self._last_fit = out
        return out

    def _resid_for(self, s_sq: Tensor, device) -> Tensor:
        """The per-bin correction, looked up for each reflection."""
        if float(self._fit_s2_max) <= 0:
            return torch.zeros_like(s_sq.reshape(-1))
        nb = self._resid.numel()
        idx = (s_sq.reshape(-1) / self._fit_s2_max.clamp_min(1e-12) * nb).long()
        return self._resid.to(device)[idx.clamp(0, nb - 1)]

    def _solve_residual(self, G_hat: float, B_hat: float) -> dict[str, float]:
        """Per-bin departure of the data from the fitted Wilson line.

        The line is fitted only above `wilson_fit_d_max`, because the Wilson
        plot bends through the solvent region -- and then the prior
        extrapolates that line straight back through the very bins the fit
        excluded. On 845 the result was a prior mean of 1650 counts in the
        99-4.0 A shell against a true 1140, a 45% overestimate with a
        perfectly good B.

        So the residual is measured where the line was never asked to hold.
        Each bin gets ln(mean observed) - ln(line), shrunk toward zero by its
        own standard error, so a noisy outer bin with three reflections cannot
        move the prior while a well-populated low-resolution bin can. Clamped,
        because this corrects a smooth misfit and is not a licence to fit
        anything.
        """
        if G_hat != G_hat or B_hat != B_hat:
            return {}
        n = self._fit_n
        ok = n >= 30
        if int(ok.sum()) < 3:
            return {}
        mean = torch.where(ok, self._fit_i / n.clamp_min(1), torch.zeros_like(n))
        x = torch.where(ok, self._fit_s2 / n.clamp_min(1), torch.zeros_like(n))
        line = math.log(max(G_hat, 1e-8)) - 2.0 * B_hat * x
        resid = torch.where(mean > 0, torch.log(mean.clamp_min(1e-8)) - line,
                            torch.zeros_like(mean))

        # James-Stein style shrinkage by the bin's own noise: var(log mean) is
        # about Var/(n*mean^2)
        var = (self._fit_i2 / n.clamp_min(1) - mean.pow(2)).clamp_min(0.0)
        se2 = (var / n.clamp_min(1) / mean.pow(2).clamp_min(1e-12)).clamp_min(1e-12)
        spread = resid[ok].pow(2).mean().clamp_min(1e-12)
        shrunk = resid * (spread / (spread + se2))
        shrunk = shrunk.clamp(-self.refit_residual_max, self.refit_residual_max)
        shrunk = torch.where(ok, shrunk, torch.zeros_like(shrunk))

        eta = self.refit_damping
        with torch.no_grad():
            self._resid.mul_(1 - eta).add_(eta * shrunk)
        return {
            "wilson_resid_max": float(self._resid.abs().max()),
            "wilson_resid_rms": float(self._resid.pow(2).mean().sqrt()),
        }

    def _zero_fit_buffers(self) -> None:
        for name in ("_fit_n", "_fit_i", "_fit_i2", "_fit_s2", "_fit_qi", "_fit_qi2"):
            getattr(self, name).zero_()

    def _maybe_init_scale(
        self, counts: Tensor, s_sq: Tensor, mask: Tensor
    ) -> None:
        """One-time DIALS-free scale init from raw counts (first batch only).

        Skipped when `init_scale_from_counts` is off, which an explicit `init_G`
        also turns off so that a pinned scale is never overwritten.
        """
        if not (self.stabilize_scale and self.init_scale_from_counts):
            return
        if bool(self.scale_initialized):
            return
        with torch.no_grad():
            m = mask.squeeze(-1)  # (B, P)
            npix = m.sum(-1).clamp_min(1.0)
            bg_mean = self.bg_concentration / self.bg_rate.clamp_min(1e-6)
            bg_mean = bg_mean.mean()  # scalar proxy (per-bin or scalar prior)
            # intensity proxy = summed counts above background; raw data, no DIALS
            i_hat = (counts * m).sum(-1) - bg_mean * npix
            g0, _ = self.wilson_fit(i_hat, s_sq, self.wilson_init_bins)
            self._set_scale_from_fit(g0)
            self.scale_initialized.fill_(True)

    def _multiplicity_weight(self, metadata: dict, like: Tensor, device) -> Tensor:
        """1/N per reflection, N being how many observations share its latent.

        Falls back to 1 with one warning when the column is absent, so datasets
        cut before `multiplicity` existed keep the previous behaviour instead of
        failing. N is clamped at 1 because a reflection observed once needs the
        full prior.
        """
        mult = metadata.get("multiplicity")
        if mult is None:
            if not self._warned_multiplicity:
                logger.warning(
                    "kl_per_multiplicity is on but metadata has no "
                    "'multiplicity' column; the intensity prior is being "
                    "applied once per observation. Add it with "
                    "scripts/add_multiplicity.py."
                )
                self._warned_multiplicity = True
            return torch.ones((), device=device)
        mult = mult.to(device).reshape(like.shape).to(like.dtype)
        return 1.0 / mult.clamp(min=1.0)

    def _intensity_kl_weight_for(self, group_labels, device):
        """Per-reflection intensity KL weight.

        A per-resolution-bin vector gathered by `group_labels` when calibration
        is on; otherwise the scalar `intensity_kl_weight` -- the legacy path,
        bit-for-bit unchanged.
        """
        if not self.intensity_kl_calibrate:
            return self.intensity_kl_weight
        if group_labels is None:
            raise ValueError(
                "intensity_kl_calibrate needs per-reflection group_labels "
                "(set n_bins > 1 so resolution bins exist)"
            )
        groups = group_labels.to(device).long()
        return self.intensity_kl_bins.to(device)[groups]

    def _accumulate_intensity_kl(self, qi, group_labels, centric, device):
        """Accumulate per-bin sums of the posterior mean intensity over the
        epoch's acentric reflections, for the epoch-end sd/mean calibration.

        The per-batch bincount is done in float32 (safe on CUDA) and added into
        the float64 running buffers.
        """
        groups = group_labels.to(device).long()
        mu = qi.mean.detach().reshape(-1).to(device).float()
        if centric is not None:
            keep = ~centric.reshape(-1).to(device).bool()
            groups, mu = groups[keep], mu[keep]
        n = self.intensity_kl_bins.shape[0]
        self._ikl_n += torch.bincount(groups, minlength=n).double()
        self._ikl_mu += torch.bincount(groups, weights=mu, minlength=n).double()
        self._ikl_mu2 += torch.bincount(
            groups, weights=mu * mu, minlength=n
        ).double()

    def calibrate_intensity_kl(self, epoch: int) -> dict[str, float]:
        """Epoch-end M-step for the per-bin intensity KL weight.

        Drives each shell's posterior acentric sd/mean toward
        `intensity_kl_target` (1.0 = the Wilson moment). A lower weight lets the
        posterior spread, so a shell measured below target has its weight cut.
        The step is multiplicative, damped by `intensity_kl_gain`, and clamped
        both per step (x0.5..x2) and to [min, max], so one noisy shell cannot
        swing the weight. Shells with too little acentric evidence hold.
        """
        dev = self.intensity_kl_bins.device
        n = self._ikl_n.clamp(min=1.0)
        mean = self._ikl_mu / n
        var = (self._ikl_mu2 / n - mean * mean).clamp(min=0.0)
        cv = var.sqrt() / mean.clamp(min=1e-6)
        seen = self._ikl_n >= 50
        ratio = (cv / self.intensity_kl_target).clamp(min=1e-3)
        step = ratio.pow(self.intensity_kl_gain).clamp(0.5, 2.0)
        step = step.to(dev, dtype=self.intensity_kl_bins.dtype)
        new = (self.intensity_kl_bins * step).clamp(
            self.intensity_kl_min, self.intensity_kl_max
        )
        updated = torch.where(seen.to(dev), new, self.intensity_kl_bins)
        self.intensity_kl_bins.copy_(updated)
        stats = {
            "ikl/cv_mean": float(cv[seen].mean()) if bool(seen.any()) else float("nan"),
            "ikl/weight_mean": float(self.intensity_kl_bins.mean()),
            "ikl/weight_min": float(self.intensity_kl_bins.min()),
            "ikl/weight_max": float(self.intensity_kl_bins.max()),
        }
        self._ikl_n.zero_()
        self._ikl_mu.zero_()
        self._ikl_mu2.zero_()
        return stats


    def intensity_prior_params(
        self, metadata: dict, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """`(concentration, rate)` of the Wilson intensity prior, per reflection.

        This is the same prior `forward` builds for the KL, exposed so a
        prediction can be divided by it. A Gamma posterior over a Gamma prior is
        again a Gamma, so the likelihood-only distribution is recoverable in
        closed form -- which is what lets us hand careless something that has not
        already been shrunk toward this prior once.

        Returns the acentric `Gamma(1, tau)` or, where the metadata carries the
        flag, the centric `Gamma(1/2, tau/2)`.
        """
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        tau = self._get_tau(metadata, s_sq, device)
        if self.refit_residual and self.refit_enabled:
            tau = tau * torch.exp(-self._resid_for(s_sq, device))

        concentration = torch.ones_like(tau)
        prior_rate = tau
        centric = metadata.get("centric")
        if centric is not None:
            centric = centric.to(device).reshape(tau.shape).bool()
            concentration = torch.where(centric, 0.5, 1.0)
            prior_rate = torch.where(centric, tau * 0.5, tau)
        return concentration, prior_rate

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        qp: Distribution | ProfileSurrogateOutput,
        qi: Distribution,
        qbg: Distribution,
        mask: Tensor,
        group_labels: Tensor | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        device = rate.device
        batch_size = rate.shape[0]
        counts = counts.to(device)
        mask = mask.to(device)

        kl = torch.zeros(batch_size, device=device)

        # get metadata

        metadata = kwargs.get("metadata")
        if metadata is None or "d" not in metadata:
            raise ValueError("Wilson loss requires metadata['d'].")

        # Coset (background-only) reflections carry no Bragg latent. In the
        # "no-KL" modes ("override_no_kl", "supervised", "aux") we drop their
        # intensity and profile KL (toward the Wilson/profile priors) while
        # keeping the background KL. `lattice` stays None for plain "override",
        # reproducing the legacy behavior exactly.
        coset_mode = kwargs.get("coset_mode", "override")
        coset = (
            metadata["is_coset"].bool().to(device)
            if "is_coset" in metadata
            else None
        )
        lattice = None
        if coset is not None and coset_mode in (
            "override_no_kl",
            "supervised",
            "aux",
        ):
            lattice = (~coset).float()

        # profile kl-divergence
        prf_prior_scale = getattr(
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )

        # Wilson intensity KL
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        self._maybe_init_scale(counts, s_sq, mask)

        if self.refit_enabled and self.training:
            self._accumulate_fit(counts, mask, s_sq, qi, metadata, device)

        tau = self._get_tau(metadata, s_sq, device)
        if self.refit_residual and self.refit_enabled:
            # the prior mean is G exp(-2Bs^2) * exp(r(s^2)), so tau divides by
            # the same factor. Bins are the ones the fit used, so a reflection
            # is corrected by the departure measured where it actually sits.
            tau = tau * torch.exp(-self._resid_for(s_sq, device))
        if self.refit_enabled:
            # the scalars are set by the M-step, so they take no gradient.
            # Detaching also drops them out of the global gradient clip, where
            # the one-sided dKL/dB would otherwise compete with the network for
            # the clip budget.
            tau = tau.detach()

        # Wilson statistics differ between centric and acentric reflections.
        # An acentric |F|^2 is exponential -- Gamma(1, 1/Sigma) -- while a
        # centric one is Sigma times a chi-squared with ONE degree of freedom,
        # Gamma(1/2, 1/(2 Sigma)). Both have mean Sigma, but the centric has
        # twice the variance and far more mass near zero, so imposing shape 1
        # on a centric under-weights small intensities and pulls weak centrics
        # up. They are 13.8% of reflections on SBGrid 821, and 23% below 3 A.
        #
        # `centric` is a per-reflection flag carried in the metadata. Datasets
        # cut before it existed do not have it, and fall back to the acentric
        # form -- the previous behaviour -- rather than failing.
        concentration = torch.ones_like(tau)
        # not `rate`: that name already holds the Poisson pixel rate, which is
        # passed to the likelihood further down
        prior_rate = tau
        centric = metadata.get("centric")
        if centric is not None:
            centric = centric.to(device).reshape(tau.shape).bool()
            concentration = torch.where(centric, 0.5, 1.0)
            # halving the rate alongside the shape holds the prior mean at
            # Sigma; only the shape of the distribution changes
            prior_rate = torch.where(centric, tau * 0.5, tau)
        elif not self._warned_centric:
            logger.warning(
                "metadata has no 'centric' flag; treating every reflection as "
                "acentric. Centric reflections are Gamma(1/2), so their prior "
                "is too tight near zero. Add the column with "
                "scripts/sbgrid/add_centric_flag.py."
            )
            self._warned_centric = True

        p_i = Gamma(concentration=concentration, rate=prior_rate)

        ikl_w = self._intensity_kl_weight_for(group_labels, device)
        kl_i = kl_divergence(qi, p_i) * ikl_w
        if self.kl_per_multiplicity:
            kl_i = kl_i * self._multiplicity_weight(metadata, kl_i, device)
        kl = kl + kl_i
        if self.intensity_kl_calibrate and self.training:
            self._accumulate_intensity_kl(qi, group_labels, centric, device)

        # background prior: shared Gamma, or per-resolution-bin
        if self.bg_concentration.ndim == 1:
            if group_labels is None:
                raise ValueError(
                    "per-bin background prior requires group_labels"
                )
            groups = group_labels.to(device).long()
            bg_conc = self.bg_concentration[groups]
            bg_rate = self.bg_rate[groups]
        else:
            bg_conc = self.bg_concentration
            bg_rate = self.bg_rate
        p_bg = Gamma(concentration=bg_conc, rate=bg_rate)
        kl_bg = kl_divergence(qbg, p_bg) * self.background_kl_weight
        kl = kl + kl_bg

        # tilted-plane background, when one is configured. The slopes are shrunk to
        # flat by their own prior, so the tilt is only paid for where the pixels
        # support it -- otherwise two more per-shoebox nuisances would make the
        # intensity estimator nonlinear in the counts it is fitted to.
        kl_bg_plane = compute_background_plane_kl(
            kwargs.get("qbg_plane"),
            self.background_plane_kl_weight,
            batch_size,
            device,
        )
        kl = kl + kl_bg_plane

        neg_ll = self.count_likelihood.neg_ll(
            rate, counts, mask, group_labels=group_labels
        )

        # Auxiliary L2 penalty pulling coset intensity toward a small floor
        # ("aux" mode). Penalized in log-space (toward log I_floor) so the
        # gradient does not vanish as intensity shrinks — an intensity-space L2
        # would, via the exp Jacobian. Per-reflection, nonzero only on cosets.
        aux = torch.zeros_like(neg_ll)
        if (
            coset_mode == "aux"
            and coset is not None
            and self.coset_aux_weight > 0
        ):
            log_mu = torch.log(qi.mean.clamp(min=self.eps))
            target = math.log(max(self.coset_i_floor, 1e-6))
            aux = coset.float() * (log_mu - target).pow(2)

        loss = (neg_ll + kl + self.coset_aux_weight * aux).mean()

        # Average intensity/profile KL over contributing (lattice) reflections so
        # the logged values stay interpretable when cosets are masked out.
        if lattice is not None:
            n_lat = lattice.sum().clamp(min=1.0)
            kl_prf_mean = kl_prf.sum() / n_lat
            kl_i_mean = kl_i.sum() / n_lat
        else:
            kl_prf_mean = kl_prf.mean()
            kl_i_mean = kl_i.mean()

        if coset is not None:
            n_coset = coset.float().sum().clamp(min=1.0)
            coset_aux_mean = (self.coset_aux_weight * aux).sum() / n_coset
        else:
            coset_aux_mean = torch.zeros((), device=device)

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf_mean,
            "kl_i_mean": kl_i_mean,
            "kl_bg_mean": kl_bg.mean(),
            "kl_bg_plane_mean": kl_bg_plane.mean(),
            "coset_aux_mean": coset_aux_mean,
        }

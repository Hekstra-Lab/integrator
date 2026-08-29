import logging
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Gamma, kl_divergence

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
        intensity_kl_weight: float = 1.0,
    ):
        super().__init__()
        # warn once, not once per batch, if the dataset predates the flag
        self._warned_centric = False
        self.b_min = b_min  # minimum B-factor
        self.n_bins = n_bins
        self.stabilize_scale = stabilize_scale
        self.learn_B = learn_B
        self.refit_every = int(refit_prior_every_n_epochs)
        self.refit_bins = int(refit_bins)
        self.shoebox_dhw = tuple(shoebox_dhw) if shoebox_dhw else None
        self.refit_fg_radius = int(refit_fg_radius)
        self.refit_guard = int(refit_guard)
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
        self.intensity_kl_weight = (
            pi_cfg.weight if pi_cfg is not None else intensity_kl_weight
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

        # profile kl-divergence
        prf_prior_scale = getattr(
            qp, "prior_scale", _DEFAULT_PROFILE_PRIOR_SCALE
        )
        kl_prf = compute_profile_kl(
            qp, prf_prior_scale, self.profile_kl_weight, device
        )
        kl = kl + kl_prf

        # Wilson intensity KL
        d = metadata["d"].to(device)
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))

        self._maybe_init_scale(counts, s_sq, mask)

        if self.refit_enabled and self.training:
            self._accumulate_fit(counts, mask, s_sq, qi, metadata, device)

        tau = self._get_tau(metadata, s_sq, device)
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

        kl_i = kl_divergence(qi, p_i) * self.intensity_kl_weight
        kl = kl + kl_i

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

        neg_ll = self.count_likelihood.neg_ll(
            rate, counts, mask, group_labels=group_labels
        )

        loss = (neg_ll + kl).mean()

        return {
            "loss": loss,
            "neg_ll_mean": neg_ll.mean(),
            "kl_mean": kl.mean(),
            "kl_prf_mean": kl_prf.mean(),
            "kl_i_mean": kl_i.mean(),
            "kl_bg_mean": kl_bg.mean(),
        }

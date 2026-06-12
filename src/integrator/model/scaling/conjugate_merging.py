"""Conjugate Bayesian merging via Poisson-Gamma sufficient statistics.

Generative model. For each shoebox i (HKL h, pixel p):

    counts_{i,p} ~ Poisson(rate_{i,p})
    rate_{i,p}   = s_i * I_h * profile_{i,p} + bg_i
    I_h          ~ Gamma(alpha_W, tau_h)         (Wilson prior approx)

With a Gamma prior on `I_h` and Poisson likelihood that is *linear* in
`I_h`, the conditional posterior `q(I_h | profile, bg, scale, counts)`
is Gamma in closed form via data augmentation (signal/background count
split):

    pi_{i,p} = s_i * Itil_h * profile_{i,p} /
               (s_i * Itil_h * profile_{i,p} + bg_i)

    alpha_h = alpha_W + sum_{i in h, p} pi_{i,p} * c_{i,p} * mask
    beta_h  = tau_h    + sum_{i in h, p} s_i  * profile_{i,p}  * mask

    q(I_h) = Gamma(alpha_h, beta_h)

The responsibility uses the geometric-mean intensity Itil_h = exp(E_q[log I_h])
= exp(psi(alpha_h) - log beta_h) (the exact CAVI update), solved as a per-HKL
fixed point *within the batch* and differentiated via the implicit-function
theorem (1/(1-K)) -- the same machinery as `ConjugateIntegrator._conjugate_em`,
aggregated over each HKL's complete group. I_h is solved fresh per batch (no
cross-batch state); the per-HKL merged posterior for the MTZ is computed by
`finalize_merge` -- a clean pass over the data that calibrates via quadrature.

What the neural net does:
    1. Encoders predict q(profile_i), q(bg_i) per observation.
    2. Scale s_i from physics (LP, geometry) with optional learnable correction.
    3. Per-HKL I_h is *derived* from sufficient statistics — no encoder for I,
       no per-HKL embedding, no learned merging operator.

Pair with `GroupedAsuIdBatchSampler` so every HKL in a batch has all its
observations present (the fixed point needs complete groups).
"""

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma, kl_divergence

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _get_normalized_position,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import (
    ChebyshevScale,
    LaueMLPScale,
    LaueSpectralScale,
    MLPScale,
    SpatialChebyshevScale,
)

logger = logging.getLogger(__name__)


def _scatter_sum_compact(
    src: Tensor, index: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Scatter sum over unique indices.

    Returns:
        out: (n_unique,) — sum of src grouped by index.
        inverse: (B,) — maps each row in src to its position in out.
        unique_idx: (n_unique,) — the unique values of index.
    """
    unique_idx, inverse = torch.unique(index, return_inverse=True)
    n_groups = len(unique_idx)
    out = torch.zeros(n_groups, device=src.device, dtype=src.dtype)
    out.scatter_add_(0, inverse, src)
    return out, inverse, unique_idx


class ConjugateMergingIntegrator(BaseIntegrator):
    """Per-HKL Gamma intensity via Poisson-Gamma conjugate update.

    Best paired with GroupedAsuIdBatchSampler (`group_by_asu_id: true` in
    the YAML data loader) so each batch contains complete HKL groups.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
    }

    def __init__(
        self,
        cfg: configs.IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
    ):
        super().__init__(cfg, loss, encoders, surrogates)

        if cfg.n_hkl is None:
            raise ValueError("ConjugateMergingIntegrator requires n_hkl.")

        # LP is applied through the scale here (scale = scale_fn / lp), so I_h is
        # already the LP-corrected intensity. lp_correction would also multiply
        # the Wilson prior by LP -- a double count -- so forbid the combination.
        if getattr(self.loss, "_apply_lp", False):
            raise ValueError(
                "ConjugateMergingIntegrator applies LP through the scale, so "
                "I_h is LP-corrected; enabling lp_correction would multiply the "
                "Wilson prior by LP too and double-count it. Set "
                "loss.args.lp_correction: false."
            )

        self.n_hkl = cfg.n_hkl
        # Wilson prior shape (alpha_W = 1 → Exponential, the acentric Wilson)
        self.alpha_W = float(getattr(cfg, "wilson_alpha", 1.0))

        # Per-HKL merged posterior (alpha_h, beta_h), populated ONLY by
        # finalize_merge (a clean pass over the data). Non-persistent: derived
        # state, recomputed at inference, not carried in the checkpoint. There is
        # no EMA: the E-step solves I_h fresh per batch (no cross-batch carry).
        self.register_buffer(
            "alpha_buffer",
            torch.full((cfg.n_hkl,), self.alpha_W),
            persistent=False,
        )
        self.register_buffer(
            "beta_buffer", torch.ones(cfg.n_hkl), persistent=False
        )
        self.register_buffer(
            "buffer_seen",
            torch.zeros(cfg.n_hkl, dtype=torch.bool),
            persistent=False,
        )

        # Closed-form per-HKL Wilson KL weight. _step scales the per-HKL KL to
        # the per-observation scale (sum / n_obs), so merge_kl_weight = 1.0 is
        # ELBO-consistent; raise above 1.0 for stronger Wilson regularization.
        self.merge_kl_weight = float(getattr(cfg, "merge_kl_weight", 1.0))

        # Cross-observation scaling-consistency loss weight (0 = off). Gives the
        # per-observation scale a direct gradient via internal consistency of
        # symmetry-equivalents (DIALS-style, data-only) -- see _consistency_loss.
        self.consistency_weight = float(getattr(cfg, "consistency_weight", 0.0))

        # Calibrated-posterior export. finalize_merge computes the mean-field per
        # HKL (Pass 1) then overwrites it with the exact collapsed-posterior
        # quadrature (Pass 2 -- the per-HKL analogue of
        # ConjugateIntegrator.exact_intensity_posterior). n_grid from cfg;
        # n_nuisance is fixed at 1 (Fix A) because the merging quadrature already
        # passes the whole dataset.
        self.exact_posterior_n_grid = int(
            getattr(cfg, "exact_posterior_n_grid", 1024)
        )

        # Inner per-HKL CAVI fixed point (ConjugateIntegrator machinery): max
        # responsibility iterations and the relative-change early-stop tolerance.
        self.n_em_iters = int(getattr(cfg, "n_em_iters", 40))
        self.em_tol = float(getattr(cfg, "em_tol", 1e-3))

        # If true, sample I_h from q(I_h) for the reconstruction NLL (proper
        # MC ELBO). If false, use the posterior mean (point estimate, simpler).
        self.sample_I_h = bool(getattr(cfg, "sample_I_h", True))

        # Optional decoupled LR for the (weakly-identified, slow-to-equilibrate)
        # scale field — its own Adam param group in _build_optimizer. None => lr.
        self.scaling_lr = getattr(cfg, "scaling_lr", None)

        # Scale function
        if getattr(cfg, "scale_laue_spectral", False):
            # Polychromatic (Laue stills): structured spectral scale. G(lambda) as
            # a ChebyshevSpectrum (warm-startable from the working polychromatic_
            # wilson model) x optional physical Lorentz/polarization x optional
            # geometry residual. The conjugate responsibility split derives I_h, so
            # the scale only needs to carry the per-observation G(lambda)/geometry.
            self.scale_fn = LaueSpectralScale(
                degree=getattr(cfg, "scale_spectrum_degree", 40),
                lambda_min=cfg.scale_lambda_min,
                lambda_max=cfg.scale_lambda_max,
                spectrum_init_from=getattr(cfg, "scale_spectrum_init_from", None),
                freeze_spectrum=getattr(cfg, "scale_freeze_spectrum", False),
                lorentz=getattr(cfg, "scale_lorentz", False),
                polarization=getattr(cfg, "scale_polarization", False),
                polarization_fraction=getattr(
                    cfg, "scale_polarization_fraction", 0.99
                ),
                beam_center=cfg.scale_beam_center,
                residual=getattr(cfg, "scale_residual", False),
                residual_hidden=cfg.scale_mlp_hidden,
                residual_layers=cfg.scale_mlp_layers,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
            )
        elif getattr(cfg, "scale_laue_mlp", False):
            # Polychromatic black-box MLP scale (A/B baseline for the spectral one).
            n_abs_sh = 0
            if getattr(cfg, "scale_mlp_absorption", False):
                n_abs_sh = (int(cfg.scale_sh_lmax) + 1) ** 2 - 1
            self.scale_fn = LaueMLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                lambda_min=cfg.scale_lambda_min,
                lambda_max=cfg.scale_lambda_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                n_images=getattr(cfg, "scale_n_images", None),
                head_init_std=getattr(cfg, "scale_head_init_std", 0.0),
                n_abs_sh=n_abs_sh,
                absorption_even_only=getattr(
                    cfg, "scale_mlp_absorption_even_only", True
                ),
            )
        elif cfg.scale_mlp:
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                d_max=60.0,
                head_init_std=getattr(cfg, "scale_head_init_std", 0.0),
            )
        elif cfg.scale_spatial:
            self.scale_fn = SpatialChebyshevScale(
                degree_frame=cfg.scale_degree,
                degree_radius=cfg.scale_degree_radius,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_min=cfg.scale_r_min,
                r_max=cfg.scale_r_max,
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

    # ------------------------------------------------------------------

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        if isinstance(self.scale_fn, (LaueSpectralScale, LaueMLPScale)):
            # Laue stills: the scale owns the spectrum + LP/geometry; no frame or
            # lp column exists. Inputs [lambda, x, y, d] (+ image/SH for the MLP).
            wavelength = metadata["wavelength"].to(device).float()
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            if isinstance(self.scale_fn, LaueSpectralScale):
                return self.scale_fn(wavelength, x_det, y_det, d)
            image_num = None
            if self.scale_fn.n_images > 0:
                image_num = metadata["image_num"].to(device).long()
            a = None
            if self.scale_fn.n_abs_sh > 0:
                a = metadata["absorption_sh"].to(device).float()
            return self.scale_fn(wavelength, x_det, y_det, d, image_num, a)
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d)
        elif isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            return self.scale_fn(frame, x_det, y_det) / lp
        else:
            return self.scale_fn(frame) / lp

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam; with scaling_lr set, the scale field gets its own param group.

        The per-frame scale is the merging model's slowest-identified field and
        it gates the anomalous signal, so a decoupled scaling_lr lets it
        equilibrate without raising the encoder LR. When scaling_lr is None this
        defers to the base optimizer (byte-identical behavior). The base
        decoder_weight_decay group is preserved. Any LambdaLR warmup/schedule
        scales all groups by the same factor, keeping the relative LRs.
        """
        if self.scaling_lr is None:
            return super()._build_optimizer()
        scale_params: list[nn.Parameter] = []
        decoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("scale_fn."):
                scale_params.append(param)
            elif (
                self.decoder_weight_decay is not None
                and name.endswith("surrogates.qp.decoder.weight")
            ):
                decoder_params.append(param)
            else:
                other_params.append(param)
        groups: list[dict] = [
            {"params": other_params, "weight_decay": self.weight_decay}
        ]
        if decoder_params:
            groups.append(
                {
                    "params": decoder_params,
                    "weight_decay": self.decoder_weight_decay,
                }
            )
        groups.append(
            {
                "params": scale_params,
                "weight_decay": self.weight_decay,
                "lr": self.scaling_lr,
            }
        )
        return torch.optim.Adam(groups, lr=self.lr)

    def _wilson_tau(self, d: Tensor) -> Tensor:
        """Wilson prior rate tau from resolution d (per-obs or per-HKL).

        No LP factor enters here: this model applies LP through the scale, so the
        prior stays on the corrected-intensity scale. lp_correction is forbidden
        in __init__, so _get_tau never takes its lp branch (and never needs an
        'lp' key in the metadata it is handed).
        """
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _conjugate_em_merged(
        self,
        counts: Tensor,
        profile_mean: Tensor,
        bg_mean: Tensor,
        scale: Tensor,
        tau_h: Tensor,
        mask: Tensor,
        inverse: Tensor,
        n_unique: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-HKL geometric-CAVI fixed point with the implicit-function gradient.

        The merging analogue of `ConjugateIntegrator._conjugate_em`: the
        responsibility uses the geometric-mean intensity exp(E_q[log I_h]) =
        exp(psi(alpha_h) - log beta_h) (Bishop PRML 10.9), and the converged
        fixed point is differentiated exactly via 1/(1-K) rather than unrolled.
        Sufficient statistics are summed over each HKL's observations, so the
        batch must contain complete HKL groups (group_by_asu_id). This replaces
        the EMA-buffer E-step: I_h is solved fresh per batch, not carried across
        batches. Returns (alpha_h, beta_h, pi).
        """
        device = counts.device
        cm = counts * mask  # (B, P)
        bg = bg_mean.unsqueeze(-1)  # (B, 1)
        e = scale.unsqueeze(-1) * profile_mean  # (B, P) signal exposure / pixel

        def scatter(x: Tensor) -> Tensor:  # sum per HKL via precomputed inverse
            return torch.zeros(
                n_unique, device=device, dtype=x.dtype
            ).scatter_add_(0, inverse, x)

        # beta_h = tau_h + sum_{i,p} s_i prf (constant in I_h; carries gradient).
        beta_h = tau_h + scatter((e * mask).sum(dim=-1))
        log_beta_h = beta_h.clamp(min=1e-12).log()

        def em_map(I_h_tilde: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """One CAVI step from the per-HKL geometric-mean intensity."""
            signal_rate = e * I_h_tilde[inverse].unsqueeze(-1)  # (B, P)
            pi = signal_rate / (signal_rate + bg).clamp(min=1e-12)
            alpha_h = self.alpha_W + scatter((pi * cm).sum(dim=-1))
            i_new = torch.exp(
                torch.digamma(alpha_h.clamp(min=1e-6)) - log_beta_h
            )
            return alpha_h, pi, i_new

        # Phase 1: converge the geometric-mean fixed point (no gradient). The
        # merge Jacobian K is high (responsibilities summed over ~22 obs per HKL),
        # so weak/background-dominated HKLs need many iterations. Freeze each HKL
        # once it converges (PER-HKL early-stop) and break only when ALL have, so
        # a single worst HKL no longer gates the global torch.all stop while the
        # rest waste iterations or, at the cap, the worst stay under-iterated.
        with torch.no_grad():
            i_tilde = 1.0 / tau_h.clamp(min=1e-12)  # Wilson-mean cold start
            converged = torch.zeros(n_unique, dtype=torch.bool, device=device)
            iters_used = self.n_em_iters
            for it in range(self.n_em_iters):
                _, _, i_new = em_map(i_tilde)
                rel = (i_new - i_tilde).abs() / i_tilde.clamp(min=1e-12)
                # adopt the update only for not-yet-converged HKLs, then freeze.
                i_tilde = torch.where(converged, i_tilde, i_new)
                converged = converged | (rel < self.em_tol)
                if bool(converged.all()):
                    iters_used = it + 1
                    break
            # Diagnostics (logged in _step): is the cap under-iterating the merge?
            self._em_iters_used = iters_used
            self._em_frac_converged = float(converged.float().mean())
            alpha_h, pi, _ = em_map(i_tilde)
            trigamma = torch.special.polygamma(1, alpha_h.clamp(min=1e-6))
            k = (
                trigamma * scatter((cm * pi * (1.0 - pi)).sum(dim=-1))
            ).clamp(max=1.0 - 1e-3)

        # Phase 2: implicit-function gradient via a 1/(1-K)-corrected step.
        _, _, f = em_map(i_tilde)  # value ~ i_tilde*, carries d_theta f
        i_implicit = i_tilde + (f - i_tilde) / (1.0 - k)
        alpha_h, pi, _ = em_map(i_implicit)
        return alpha_h, beta_h, pi

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma posterior from the merge buffers (for MTZ output).

        Populated by `finalize_merge` (a clean pass over the data); without it
        the buffers hold the prior. With `calibrate=True` (default) it is the
        exact quadrature posterior, otherwise the mean-field.
        """
        return Gamma(
            self.alpha_buffer.clamp(min=1e-6),
            self.beta_buffer.clamp(min=1e-12),
        )

    @torch.no_grad()
    def finalize_merge(self, dataloader, calibrate: bool = True) -> None:
        """Compute the per-HKL merged posterior over the full dataset (no EMA).

        Pass 1 -- run the per-HKL geometric-CAVI EM on each batch's complete
        groups and store the mean-field (alpha_h, beta_h). Pass 2 (calibrate) --
        the exact collapsed-posterior quadrature (the per-HKL analogue of
        ConjugateIntegrator.exact_intensity_posterior), which uses the Pass-1
        mean-field for each HKL's grid range, then overwrites the buffers with the
        calibrated moment-matched Gamma. Requires complete HKL groups per batch
        (group_by_asu_id); `cli/pred.py` passes a grouped loader.
        `calibrate=False` stops at the mean-field (for an A/B).
        """
        self.eval()
        device = self.alpha_buffer.device
        self.alpha_buffer.fill_(self.alpha_W)
        self.beta_buffer.fill_(1.0)
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)

        # Pass 1: per-HKL mean-field EM (the merged estimate + the grid basis).
        logger.info(
            "finalize_merge Pass 1: per-HKL mean-field EM over the dataset "
            "(device=%s)",
            device,
        )
        n_batches = 0
        for batch in dataloader:
            counts, shoebox, mask, metadata = batch
            counts = counts.clamp(min=0).to(device)
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            b = shoebox.shape[0]
            sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
            position = _get_normalized_position(metadata, device)
            qbg = self.surrogates["qbg"](
                self.encoders["k_bg"](sr), self.encoders["r_bg"](sr)
            )
            prf_labels = metadata.get(
                "profile_group_label", metadata.get("group_label")
            )
            prf_labels = prf_labels.long() if prf_labels is not None else None
            qp = self.surrogates["qp"](
                self.encoders["profile"](sr, position=position),
                mc_samples=1,
                group_labels=prf_labels,
                metadata=metadata,
            )
            scale = self._get_scale(metadata, device)
            asu = metadata["asu_id"].long().to(device)
            d_obs = metadata["d"].to(device).float()
            d_sum, inverse, unique = _scatter_sum_compact(d_obs, asu)
            cnt, _, _ = _scatter_sum_compact(torch.ones_like(d_obs), asu)
            tau_h = self._wilson_tau(d_sum / cnt.clamp(min=1))
            alpha_h, beta_h, _ = self._conjugate_em_merged(
                counts, qp.mean_profile, qbg.mean, scale, tau_h, mask,
                inverse, len(unique),
            )
            if bool(seen[unique].any()):
                raise RuntimeError(
                    "finalize_merge requires a grouped (group_by_asu_id) loader "
                    "so each HKL is complete in one batch; found an HKL spanning "
                    "batches."
                )
            self.alpha_buffer[unique] = alpha_h
            self.beta_buffer[unique] = beta_h
            seen[unique] = True
            n_batches += 1
        self.buffer_seen.copy_(seen)
        logger.info(
            "finalize_merge Pass 1 done: %d/%d HKLs populated over %d batches "
            "(mean-field beta median=%.3g)",
            int(seen.sum()),
            self.n_hkl,
            n_batches,
            float(self.beta_buffer[seen].median()) if bool(seen.any()) else 0.0,
        )

        # Pass 2: calibrate via the exact collapsed-posterior quadrature.
        if calibrate:
            logger.info(
                "finalize_merge Pass 2: calibrated quadrature (n_grid=%d)",
                self.exact_posterior_n_grid,
            )
            post = self.exact_merged_posterior(
                dataloader, n_grid=self.exact_posterior_n_grid, n_nuisance=1
            )
            s = post["seen"]
            self.alpha_buffer[s] = post["alpha"][s]
            self.beta_buffer[s] = post["beta"][s]
            self.buffer_seen[s] = True
            logger.info(
                "finalize_merge Pass 2 done: %d HKLs calibrated", int(s.sum())
            )

    @torch.no_grad()
    def exact_merged_posterior(
        self,
        dataloader,
        *,
        n_grid: int = 1024,
        n_nuisance: int = 1,
        grid_chunk: int = 128,
    ) -> dict[str, Tensor]:
        """Calibrated per-HKL intensity posterior via collapsed-posterior quadrature.

        The mean-field Gamma(alpha_h, beta_h) fixes the signal/background
        allocation to a point and so under-disperses sigma(I_h). This integrates
        the exact collapsed per-HKL posterior

            log p(I_h|c) = (alpha_W - 1) log I_h
                           - (tau_h + sum_{i,p} e_{i,p}) I_h
                           + sum_{i,p} c_{i,p} log(e_{i,p} I_h + bg_i)

        (e_{i,p} = s_i * profile_{i,p}) over a per-HKL grid, aggregating every
        observation of each HKL across the whole dataset by scatter-add. With
        n_nuisance <= 1 the profile/background are taken at their q-means (Fix A);
        n_nuisance > 1 also propagates q(profile)/q(bg) uncertainty by Monte
        Carlo (law of total variance), at the cost of one extra dataset pass per
        sample.

        Run after training / `finalize` -- the mean-field buffer sets each HKL's
        grid range. Returns per-HKL tensors (n_hkl,): mean, var, std, alpha, beta
        (Gamma moment-matched to mean/var), and a boolean `seen` mask.
        """
        self.eval()
        device = self.alpha_buffer.device

        # Per-HKL grid from the mean-field buffer (the exact posterior is wider,
        # so the range is padded generously and skewed to the right tail).
        a_mf = self.alpha_buffer.clamp(min=1e-6)
        b_mf = self.beta_buffer.clamp(min=1e-12)
        mf_mean = a_mf / b_mf
        mf_std = a_mf.sqrt() / b_mf
        std_eff = 3.0 * mf_std
        lo = (mf_mean - 8.0 * std_eff).clamp(min=1e-8)
        hi = torch.maximum(mf_mean + 12.0 * std_eff, lo + 1e-3)
        steps = torch.linspace(0.0, 1.0, n_grid, device=device)
        grid = lo[:, None] + steps[None, :] * (hi - lo)[:, None]  # (n_hkl, G)
        log_grid = grid.clamp(min=1e-30).log()
        dw = torch.diff(grid, dim=1, prepend=grid[:, :1]).clamp(min=1e-30).log()

        n_samp = max(int(n_nuisance), 1)
        sum_mean = torch.zeros(self.n_hkl, device=device)
        sum_mean_sq = torch.zeros(self.n_hkl, device=device)
        sum_var = torch.zeros(self.n_hkl, device=device)
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)
        d_sum = torch.zeros(self.n_hkl, device=device)
        d_cnt = torch.zeros(self.n_hkl, device=device)

        for s in range(n_samp):
            data_term = torch.zeros(self.n_hkl, n_grid, device=device)
            lin_extra = torch.zeros(self.n_hkl, device=device)  # sum_{i,p} e
            for batch in dataloader:
                counts, shoebox, mask, metadata = batch
                counts = counts.clamp(min=0).to(device)
                shoebox = shoebox.to(device)
                mask = mask.to(device)
                b = shoebox.shape[0]
                shoebox_reshaped = (shoebox * mask).reshape(
                    b, 1, *self.shoebox_shape
                )
                position = _get_normalized_position(metadata, device)
                x_profile = self.encoders["profile"](
                    shoebox_reshaped, position=position
                )
                x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
                x_r_bg = self.encoders["r_bg"](shoebox_reshaped)
                qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
                prf_labels = metadata.get(
                    "profile_group_label", metadata.get("group_label")
                )
                prf_labels = (
                    prf_labels.long() if prf_labels is not None else None
                )
                qp = self.surrogates["qp"](
                    x_profile,
                    mc_samples=1,
                    group_labels=prf_labels,
                    metadata=metadata,
                )
                scale = self._get_scale(metadata, device)  # (B,)
                asu = metadata["asu_id"].long().to(device)

                if n_nuisance <= 1:
                    prof = qp.mean_profile  # (B, P)
                    bg = qbg.mean  # (B,)
                else:
                    prof = _sample_profile(qp, 1)[:, 0, :]  # (B, P)
                    bg = qbg.rsample([1])[0]  # (B,)

                cm = counts * mask  # (B, P)
                e = scale.unsqueeze(-1) * prof * mask  # (B, P) signal exposure
                lin_extra.index_add_(0, asu, e.sum(dim=-1))

                g_h = grid[asu]  # (B, G): each obs uses its own HKL's grid
                bg_u = bg[:, None, None]  # (B, 1, 1)
                for j in range(0, n_grid, grid_chunk):
                    gI = g_h[:, j : j + grid_chunk]  # (B, g)
                    rate = e[:, :, None] * gI[:, None, :] + bg_u  # (B, P, g)
                    dterm = (
                        cm[:, :, None] * rate.clamp(min=1e-30).log()
                    ).sum(dim=1)  # (B, g)
                    data_term[:, j : j + grid_chunk].index_add_(0, asu, dterm)

                if s == 0:
                    d_obs = metadata["d"].to(device).float()
                    d_sum.index_add_(0, asu, d_obs)
                    d_cnt.index_add_(0, asu, torch.ones_like(d_obs))
                seen[asu] = True

            d_per_hkl = (d_sum / d_cnt.clamp(min=1.0)).clamp(min=1e-6)
            lin_coef = self._wilson_tau(d_per_hkl) + lin_extra
            log_unnorm = (
                (self.alpha_W - 1.0) * log_grid
                - lin_coef[:, None] * grid
                + data_term
            )
            w = torch.softmax(log_unnorm + dw, dim=1)  # (n_hkl, G)
            m1 = (w * grid).sum(dim=-1)
            m2 = (w * grid.pow(2)).sum(dim=-1)
            var = (m2 - m1.pow(2)).clamp(min=0.0)
            sum_mean += m1
            sum_mean_sq += m1.pow(2)
            sum_var += var

        mean = sum_mean / n_samp
        # Law of total variance over the nuisance posterior.
        var_total = (
            sum_var / n_samp + (sum_mean_sq / n_samp - mean.pow(2))
        ).clamp(min=1e-12)
        return {
            "mean": mean,
            "var": var_total,
            "std": var_total.sqrt(),
            "alpha": mean.pow(2) / var_total,
            "beta": mean / var_total,
            "seen": seen,
        }

    # ------------------------------------------------------------------

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = counts.clamp(min=0)
        B = shoebox.shape[0]
        device = shoebox.device
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(B, 1, *self.shoebox_shape)

        # Encoders: profile + bg only (no intensity encoder)
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            group_labels=prf_labels,
            metadata=metadata,
        )

        profile_mean = qp.mean_profile  # (B, P)
        bg_mean = qbg.mean              # (B,)

        scale = self._get_scale(metadata, device)  # (B,)

        asu_ids = metadata["asu_id"].long().to(device)
        d_per_obs = metadata["d"].to(device).float()

        # Per-HKL grouping + tau (d is constant within an HKL).
        d_sum_h, inverse, unique_asu = _scatter_sum_compact(d_per_obs, asu_ids)
        count_h, _, _ = _scatter_sum_compact(torch.ones_like(d_per_obs), asu_ids)
        tau_per_hkl = self._wilson_tau(d_sum_h / count_h.clamp(min=1))
        n_unique = unique_asu.shape[0]

        # Per-HKL conjugate EM: the ConjugateIntegrator's geometric-CAVI fixed
        # point + implicit-function gradient, aggregated over each HKL's complete
        # group in the batch (requires group_by_asu_id). Solves I_h fresh per
        # batch instead of carrying an EMA estimate across batches.
        alpha_h, beta_h, pi = self._conjugate_em_merged(
            counts,
            profile_mean,
            bg_mean,
            scale,
            tau_per_hkl,
            mask,
            inverse,
            n_unique,
        )

        # No buffer update during training: the merged MTZ posterior is computed
        # by finalize_merge (a clean pass) at inference.
        q_I_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))

        # Reconstruction rates for ELBO NLL term
        if self.sample_I_h:
            zI_h = q_I_h.rsample([self.mc_samples])  # (S, n_unique)
            zI_h = zI_h.clamp(min=1e-10)
            zI_per_obs = zI_h[:, inverse]  # (S, B)
        else:
            zI_per_obs = (alpha_h / beta_h)[inverse].expand(
                self.mc_samples, B
            )  # (S, B)

        zp = _sample_profile(qp, self.mc_samples)
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        zI_scaled = (scale.unsqueeze(0) * zI_per_obs).unsqueeze(-1).permute(
            1, 0, 2
        )  # (B, S, 1)
        rate = zI_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        # Per-obs qi for downstream metrics / loss interface
        qi_per_obs = Gamma(
            alpha_h[inverse].clamp(min=1e-6),
            beta_h[inverse].clamp(min=1e-12),
        )

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi_per_obs,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)
        out["asu_id"] = asu_ids
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi_per_obs,
            "qbg": qbg,
            "qi_h": q_I_h,
            "alpha_h": alpha_h,
            "beta_h": beta_h,
            "tau_h": tau_per_hkl,
            "inverse": inverse,
            "unique_asu": unique_asu,
            "scale": scale,
            "pi_mean": pi.mean().detach(),
            "scale_mean": scale.mean().detach(),
            "scale_std": scale.std().detach(),
            "scale_min": scale.min().detach(),
            "scale_max": scale.max().detach(),
            "bg_mean": bg_mean.mean().detach(),
            "profile_max_mean": profile_mean.max(dim=-1).values.mean().detach(),
        }

    # ------------------------------------------------------------------

    def _kl_I_h(
        self, alpha_h: Tensor, beta_h: Tensor, tau_h: Tensor
    ) -> Tensor:
        """KL(Gamma(alpha_h, beta_h) || Gamma(alpha_W, tau_h)) closed form."""
        q = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))
        p = Gamma(
            self.alpha_W * torch.ones_like(alpha_h),
            tau_h.clamp(min=1e-12),
        )
        return kl_divergence(q, p)

    def _consistency_loss(
        self,
        counts: Tensor,
        mask: Tensor,
        bg: Tensor,
        scale: Tensor,
        inverse: Tensor,
        n_unique: int,
    ) -> Tensor:
        """Cross-observation scaling consistency (DIALS-style, data-only).

        Symmetry-equivalent observations of an HKL must agree after scaling.
        With the per-observation measured intensity J_i = sum_p (counts - bg)
        (data, no profile / no I_h) and the model's per-obs scale s_i, the
        best merged value is the closed-form weighted least squares estimate
            I_hat_h = sum_i w_i J_i s_i / sum_i w_i s_i^2,   w_i = 1/var(J_i),
        and the loss penalizes the weighted residual (J_i - s_i I_hat_h)^2.

        Only s_i carries gradient (J, w, and I_hat are detached), so this is a
        clean per-observation scale signal that bypasses the trainable I_h --
        the term the ELBO under-identifies because I_h absorbs scale errors. It
        is gauge-invariant (s_i -> c s_i, I_hat -> I_hat/c leaves s_i*I_hat
        fixed). Singleton HKLs contribute exactly 0 (nothing to be consistent
        with). Group by the anomalous asu_id (Friedel mates separate) so it
        tightens the within-mate scale the anomalous signal needs.
        """
        device = scale.device
        cm = counts.clamp(min=0).to(device) * mask.to(device)
        J = (cm - bg.unsqueeze(-1) * mask.to(device)).sum(dim=-1).detach()
        var = cm.sum(dim=-1).clamp(min=1.0)
        w = (1.0 / var).detach()

        def scatter(x: Tensor) -> Tensor:
            return torch.zeros(
                n_unique, device=device, dtype=x.dtype
            ).scatter_add_(0, inverse, x)

        num = scatter(w * J * scale)
        den = scatter(w * scale * scale).clamp(min=1e-12)
        i_hat = (num / den).detach()  # stop-grad WLS target per HKL
        resid = J - scale * i_hat[inverse]
        return (w * resid.pow(2)).sum() / max(scale.shape[0], 1)

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

        # Standard loss: Poisson NLL + KL_prf + KL_bg
        # Set pi_weight=0 in YAML — the per-obs KL_i is overcounted and we
        # apply the per-HKL Gamma-Gamma KL below.
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
        )

        total_loss = loss_dict["loss"]

        # ELBO-consistent weighting. The intensity KL is one term per HKL, but
        # the NLL / profile / background terms are per observation (the loss
        # averages over observations). Put the intensity KL on the same
        # per-observation scale -- sum over HKLs / n_obs, which auto-scales by
        # obs-per-HKL (~ N_HKL/N_obs). A per-HKL *mean* would over-weight it by
        # ~N_obs/N_HKL (~22x for HEWL). merge_kl_weight = 1.0 is then the ELBO.
        kl_I_per_hkl = self._kl_I_h(
            outputs["alpha_h"], outputs["beta_h"], outputs["tau_h"]
        )
        kl_I = kl_I_per_hkl.sum() / counts.shape[0] * self.merge_kl_weight
        total_loss = total_loss + kl_I

        # Scaling-consistency loss: a direct, data-only gradient for the per-obs
        # scale (the ELBO under-identifies it). Grouped by the anomalous asu_id
        # (outputs["inverse"]), so it tightens the within-mate scale.
        if self.consistency_weight > 0.0:
            consist = self._consistency_loss(
                forward_out["counts"],
                forward_out["mask"],
                outputs["qbg"].mean,
                outputs["scale"],
                outputs["inverse"],
                outputs["unique_asu"].shape[0],
            )
            total_loss = total_loss + self.consistency_weight * consist
            self.log(
                f"{step} consistency",
                consist.detach(),
                on_step=False,
                on_epoch=True,
            )

        _log_loss(
            self,
            kl=loss_dict["kl_mean"] + kl_I,
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_bg": loss_dict["kl_bg_mean"],
                "kl_i_hkl": kl_I.detach(),
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        with torch.no_grad():
            alpha_h = outputs["alpha_h"]
            beta_h = outputs["beta_h"]
            I_h_mean = alpha_h / beta_h
            I_h_var = alpha_h / beta_h.pow(2)
            self.log(
                f"{step} qi_h_mean",
                I_h_mean.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} qi_h_var",
                I_h_var.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} alpha_h_mean",
                alpha_h.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} beta_h_mean",
                beta_h.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} pi_mean",
                outputs["pi_mean"],
                on_step=False,
                on_epoch=True,
            )
            # EM inner-solve diagnostics: if em_iters_used pins at n_em_iters and
            # frac_converged < 1, the merge fixed point is under-iterated (raise
            # n_em_iters). Stashed by _conjugate_em_merged on the last forward.
            self.log(
                f"{step} em_iters_used",
                torch.tensor(
                    float(getattr(self, "_em_iters_used", self.n_em_iters))
                ),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} em_frac_converged",
                torch.tensor(float(getattr(self, "_em_frac_converged", 1.0))),
                on_step=False,
                on_epoch=True,
            )
            for k in (
                "scale_mean",
                "scale_std",
                "scale_min",
                "scale_max",
                "bg_mean",
                "profile_max_mean",
            ):
                self.log(
                    f"{step} {k}", outputs[k], on_step=False, on_epoch=True
                )
            self.log(
                f"{step} n_unique_hkl",
                torch.tensor(len(outputs["unique_asu"]), dtype=torch.float),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} obs_per_hkl",
                torch.tensor(
                    counts.shape[0] / len(outputs["unique_asu"]),
                    dtype=torch.float,
                ),
                on_step=False,
                on_epoch=True,
            )

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": (loss_dict["kl_mean"] + kl_I).detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": kl_I.detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

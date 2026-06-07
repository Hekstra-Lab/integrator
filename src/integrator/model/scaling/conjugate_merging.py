"""Conjugate Bayesian merging via Poisson-Gamma sufficient statistics.

Generative model. For each shoebox i (HKL h, pixel p):

    counts_{i,p} ~ Poisson(rate_{i,p})
    rate_{i,p}   = s_i * I_h * profile_{i,p} + bg_i
    I_h          ~ Gamma(alpha_W, tau_h)         (Wilson prior approx)

With a Gamma prior on `I_h` and Poisson likelihood that is *linear* in
`I_h`, the conditional posterior `q(I_h | profile, bg, scale, counts)`
is Gamma in closed form via data augmentation (signal/background count
split):

    pi_{i,p} = s_i * I_h_hat * profile_{i,p} /
               (s_i * I_h_hat * profile_{i,p} + bg_i)

    alpha_h = alpha_W + sum_{i in h, p} pi_{i,p} * c_{i,p} * mask
    beta_h  = tau_h    + sum_{i in h, p} s_i  * profile_{i,p}  * mask

    q(I_h) = Gamma(alpha_h, beta_h)

`I_h_hat` for the E-step comes from an EMA buffer of previous batches'
posterior means (cold start uses Wilson prior mean 1/tau_h).

What the neural net does:
    1. Encoders predict q(profile_i), q(bg_i) per observation.
    2. Scale s_i from physics (LP, geometry) with optional learnable correction.
    3. Per-HKL I_h is *derived* from sufficient statistics — no encoder for I,
       no per-HKL embedding, no learned merging operator.

Pair with `GroupedAsuIdBatchSampler` so every HKL in a batch has all its
observations present (sufficient statistics need complete groups).
"""

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
    MLPScale,
    SpatialChebyshevScale,
)


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

        # EMA buffer of posterior (alpha_h, beta_h). The mean alpha/beta is
        # used as I_h_hat for the next batch's E-step.
        self.register_buffer(
            "alpha_buffer", torch.full((cfg.n_hkl,), self.alpha_W)
        )
        self.register_buffer("beta_buffer", torch.ones(cfg.n_hkl))
        self.register_buffer(
            "buffer_seen", torch.zeros(cfg.n_hkl, dtype=torch.bool)
        )
        self.ema_momentum = float(getattr(cfg, "ema_momentum", 0.9))

        # Closed-form per-HKL Wilson KL weight. ELBO-consistent scaling is
        # N_HKL/N_obs (~0.04 for HEWL); raise above for stronger regularization.
        self.merge_kl_weight = float(getattr(cfg, "merge_kl_weight", 1.0))

        # If true, sample I_h from q(I_h) for the reconstruction NLL (proper
        # MC ELBO). If false, use the posterior mean (point estimate, simpler).
        self.sample_I_h = bool(getattr(cfg, "sample_I_h", True))

        # Scale function
        if cfg.scale_mlp:
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=getattr(cfg, "dmin", 1.0),
                d_max=60.0,
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

    def _wilson_tau(self, d: Tensor) -> Tensor:
        """Wilson prior rate tau from resolution d (per-obs or per-HKL).

        No LP factor enters here: this model applies LP through the scale, so the
        prior stays on the corrected-intensity scale. lp_correction is forbidden
        in __init__, so _get_tau never takes its lp branch (and never needs an
        'lp' key in the metadata it is handed).
        """
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _get_I_h_for_estep(
        self, asu_ids: Tensor, tau_per_obs: Tensor
    ) -> Tensor:
        """Detached geometric-mean intensity exp(E_q[log I_h]) for the E-step pi.

        This is the exact mean-field (CAVI) update (Bishop PRML 10.9; Blei et al.
        2017): the responsibility uses exp(E[log I]) = exp(psi(alpha) - log beta),
        NOT the arithmetic mean alpha/beta, which over-weights the signal.

        Seen HKLs: exp(psi(alpha) - log beta) from the EMA buffer.
        Unseen HKLs: Wilson geometric mean exp(psi(alpha_W) - log tau_h).
        """
        a = self.alpha_buffer[asu_ids].clamp(min=1e-6)
        b = self.beta_buffer[asu_ids].clamp(min=1e-12)
        seen = self.buffer_seen[asu_ids]
        psi_aW = torch.digamma(
            torch.tensor(self.alpha_W, device=a.device, dtype=a.dtype)
        )
        geo_seen = torch.exp(torch.digamma(a) - b.log())
        geo_cold = torch.exp(psi_aW - tau_per_obs.clamp(min=1e-12).log())
        return torch.where(seen, geo_seen, geo_cold).detach()

    def _update_buffer(
        self, unique_asu: Tensor, alpha_h: Tensor, beta_h: Tensor
    ) -> None:
        """EMA update of per-HKL (alpha, beta) buffers."""
        was_seen = self.buffer_seen[unique_asu]
        old_a = self.alpha_buffer[unique_asu]
        old_b = self.beta_buffer[unique_asu]
        m = self.ema_momentum
        new_a = torch.where(was_seen, m * old_a + (1 - m) * alpha_h, alpha_h)
        new_b = torch.where(was_seen, m * old_b + (1 - m) * beta_h, beta_h)
        self.alpha_buffer[unique_asu] = new_a
        self.beta_buffer[unique_asu] = new_b
        self.buffer_seen[unique_asu] = True

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma posterior from the EMA buffer (for MTZ output)."""
        return Gamma(
            self.alpha_buffer.clamp(min=1e-6),
            self.beta_buffer.clamp(min=1e-12),
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
        tau_per_obs = self._wilson_tau(d_per_obs)

        # E-step: detached pi (signal probability per pixel)
        I_h_hat = self._get_I_h_for_estep(asu_ids, tau_per_obs)  # (B,)
        signal_rate_for_pi = (
            scale.unsqueeze(-1) * I_h_hat.unsqueeze(-1) * profile_mean
        )
        bg_rate_for_pi = bg_mean.unsqueeze(-1)
        pi = signal_rate_for_pi / (
            signal_rate_for_pi + bg_rate_for_pi
        ).clamp(min=1e-12)  # (B, P)

        # M-step sufficient statistics (with mask)
        signal_per_pixel = pi * counts * mask
        beta_term_per_pixel = scale.unsqueeze(-1) * profile_mean * mask

        signal_sum_per_obs = signal_per_pixel.sum(dim=-1)  # (B,)
        beta_sum_per_obs = beta_term_per_pixel.sum(dim=-1)  # (B,)

        alpha_signal_h, inverse, unique_asu = _scatter_sum_compact(
            signal_sum_per_obs, asu_ids
        )
        beta_sum_h, _, _ = _scatter_sum_compact(beta_sum_per_obs, asu_ids)

        # Per-HKL tau (d is constant per HKL; scatter_mean over d_per_obs)
        d_sum_h, _, _ = _scatter_sum_compact(d_per_obs, asu_ids)
        count_h, _, _ = _scatter_sum_compact(torch.ones_like(d_per_obs), asu_ids)
        d_per_hkl = d_sum_h / count_h.clamp(min=1)
        tau_per_hkl = self._wilson_tau(d_per_hkl)

        alpha_h = self.alpha_W + alpha_signal_h
        beta_h = tau_per_hkl + beta_sum_h

        if self.training:
            with torch.no_grad():
                self._update_buffer(
                    unique_asu, alpha_h.detach(), beta_h.detach()
                )

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
            "pi_mean": pi.mean().detach(),
            "I_h_hat_mean": I_h_hat.mean().detach(),
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

        kl_I_per_hkl = self._kl_I_h(
            outputs["alpha_h"], outputs["beta_h"], outputs["tau_h"]
        )
        kl_I = kl_I_per_hkl.mean() * self.merge_kl_weight
        total_loss = total_loss + kl_I

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
            self.log(
                f"{step} I_h_hat_mean",
                outputs["I_h_hat_mean"],
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} buffer_coverage",
                self.buffer_seen.float().mean(),
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

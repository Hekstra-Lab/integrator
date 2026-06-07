"""Amortized per-HKL intensity merging.

The amortized sibling of `ConjugateMergingIntegrator`: instead of deriving
`q(I_h)` in closed form, a recognition network learns it from the HKL's
observations. Two aggregation modes:

`mean` (legacy)
    DeepSets mean-pool of the `k_i`/`r_i` encoder features, fed to the `qi`
    Gamma surrogate. Mean-pooling erases multiplicity, so the posterior cannot
    scale precision with the number of observations.

`sum` (amortized conjugate update, recommended)
    Mirrors the conjugate sufficient statistics. Each observation emits two
    *positive* potentials and they are summed in natural-parameter space:

        alpha_h = alpha_W + sum_i  dalpha_i          (learned signal counts)
        beta_h  = tau_h   + sum_i  s_i * sum_p prf   (analytic exposure)
        q(I_h)  = Gamma(alpha_h, beta_h)

    Sum (not mean) is what gives the correct `~1/N` precision scaling, and the
    prior `(alpha_W, tau_h)` is the base term that regularizes weak HKLs. The
    potential heads are conditioned on per-observation `[log scale, log lp, d]`
    so the merge is scale-aware (the conjugate model gets this for free via the
    exposure term). Optional refinements: a self-attention trust gate
    (`merge_attention`, soft outlier rejection) and a learned variance inflation
    (`merge_overdispersion`, the between-observation error model the conjugate
    model cannot represent).

Pair with `GroupedAsuIdBatchSampler` (`group_by_asu_id: true`) so each batch
holds complete HKL groups; the sufficient-statistic sum is then complete within
the batch and no cross-batch buffer is needed during training.
"""

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma, kl_divergence

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
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
from integrator.model.scaling.conjugate_merging import _scatter_sum_compact
from integrator.model.scaling.deepsets_merging import _scatter_mean_compact

_N_COND = 3  # per-observation conditioning: [log scale, log lp, d]


class AmortizedMergingIntegrator(BaseIntegrator):
    """Per-HKL amortized variational intensity (sibling of conjugate merging).

    See the module docstring for the `mean` vs `sum` aggregation modes. Best
    paired with `GroupedAsuIdBatchSampler` (`group_by_asu_id: true`).
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_i": configs.IntensityEncoderArgs,
        "r_i": configs.IntensityEncoderArgs,
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
            raise ValueError("AmortizedMergingIntegrator requires n_hkl.")

        # LP is applied in the scale (scale = scale_fn / lp), so I_h is the
        # LP-corrected intensity; lp_correction would double-count it.
        if getattr(self.loss, "_apply_lp", False):
            raise ValueError(
                "AmortizedMergingIntegrator applies LP through the scale, so "
                "I_h is LP-corrected; enabling lp_correction would multiply the "
                "Wilson prior by LP too and double-count it. Set "
                "loss.args.lp_correction: false."
            )

        self.n_hkl = cfg.n_hkl
        self.alpha_W = float(getattr(cfg, "wilson_alpha", 1.0))
        self.merge_kl_weight = float(getattr(cfg, "merge_kl_weight", 1.0))
        self.merge_aggregation = getattr(cfg, "merge_aggregation", "mean")
        self.merge_attention = bool(getattr(cfg, "merge_attention", False))
        self.merge_overdispersion = bool(
            getattr(cfg, "merge_overdispersion", False)
        )
        d = cfg.encoder_out

        if self.merge_aggregation == "mean":
            if "qi" not in self.surrogates:
                raise ValueError(
                    "merge_aggregation='mean' requires a 'qi' surrogate."
                )
        elif self.merge_aggregation == "sum":
            in_dim = 2 * d + _N_COND
            self.alpha_head = nn.Sequential(
                nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, 1)
            )
            if self.merge_attention:
                self.merge_attn = nn.MultiheadAttention(
                    in_dim, num_heads=1, batch_first=True
                )
                # Zero the output projection so attention starts as a no-op
                # (feat_ctx == feat): the head begins life as the pure
                # sum-of-potentials backbone and learns the gating from there.
                nn.init.zeros_(self.merge_attn.out_proj.weight)
                nn.init.zeros_(self.merge_attn.out_proj.bias)
                self.gate_head = nn.Linear(in_dim, 1)
                nn.init.constant_(self.gate_head.bias, 5.0)  # gate ~ 1 at init
            if self.merge_overdispersion:
                self.disp_head = nn.Sequential(
                    nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1)
                )
                nn.init.constant_(
                    self.disp_head[-1].bias, -5.0
                )  # phi ~ 0 init
        else:
            raise ValueError(
                f"unknown merge_aggregation {self.merge_aggregation!r}"
            )

        # Final merged per-HKL posterior, populated by `finalize_merge` (a clean
        # full-dataset pass).
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

        # Scale function (identical to the other merging integrators)
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
        """Wilson prior rate tau from resolution d (lp lives in the scale)."""
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _attend(self, feat: Tensor, asu_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Self-attention over each HKL's observations -> (context, trust gate).

        Pads observations into (n_groups, max_obs, F) by HKL, runs masked
        single-head attention (so observations see only their group-mates), and
        gathers back to per-observation order. The gate in (0, 1) down-weights
        observations discordant with their group (soft outlier rejection).
        """
        unique, inverse, counts = torch.unique(
            asu_ids, return_inverse=True, return_counts=True
        )
        n_groups = len(unique)
        f_dim = feat.shape[-1]
        b = feat.shape[0]
        device = feat.device
        max_n = int(counts.max().item())

        # Position of each observation within its (sorted-by-group) group.
        order = torch.argsort(inverse, stable=True)
        offsets = (
            torch.cumsum(counts, 0) - counts
        )  # group start in sorted order
        pos = torch.empty_like(inverse)
        pos[order] = torch.arange(b, device=device) - offsets[inverse[order]]

        padded = feat.new_zeros(n_groups, max_n, f_dim)
        padded[inverse, pos] = feat
        key_pad = torch.ones(n_groups, max_n, dtype=torch.bool, device=device)
        key_pad[inverse, pos] = False  # False = real observation

        attended, _ = self.merge_attn(
            padded, padded, padded, key_padding_mask=key_pad
        )
        feat_ctx = (attended + padded)[inverse, pos]  # residual, gather valid
        gate = torch.sigmoid(self.gate_head(feat_ctx)).squeeze(-1)
        return feat_ctx, gate

    def _merge(
        self,
        x_k_i: Tensor,
        x_r_i: Tensor,
        scale: Tensor,
        profile_mean: Tensor | None,
        mask: Tensor,
        asu_ids: Tensor,
        d_per_obs: Tensor,
        lp: Tensor,
    ) -> tuple[Gamma, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Merge per-observation features into per-HKL q(I_h).

        Returns `(qi_h, alpha_h, beta_h, inverse, unique, tau_h)`, where
        `inverse` maps each observation to its HKL row and `unique` are the HKL
        ids present.
        """
        d_sum, inverse, unique = _scatter_sum_compact(d_per_obs, asu_ids)
        cnt, _, _ = _scatter_sum_compact(torch.ones_like(d_per_obs), asu_ids)
        tau_h = self._wilson_tau((d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6))

        if self.merge_aggregation == "mean":
            z_k, _, _ = _scatter_mean_compact(x_k_i, asu_ids)
            z_r, _, _ = _scatter_mean_compact(x_r_i, asu_ids)
            qi_h = self.surrogates["qi"](z_k, z_r)
            return (
                qi_h,
                qi_h.concentration,
                qi_h.rate,
                inverse,
                unique,
                tau_h,
            )

        # sum mode: per-observation positive potentials, summed.
        cond = torch.stack(
            [scale.clamp(min=1e-8).log(), lp.clamp(min=1e-8).log(), d_per_obs],
            dim=-1,
        )  # (B, 3)
        feat = torch.cat([x_k_i, x_r_i, cond], dim=-1)  # (B, 2d+3)

        gate = None
        if self.merge_attention:
            feat, gate = self._attend(feat, asu_ids)

        delta_alpha = F.softplus(self.alpha_head(feat)).squeeze(-1)  # (B,)
        delta_beta = scale * (profile_mean * mask).sum(
            dim=-1
        )  # analytic exposure
        if gate is not None:
            delta_alpha = delta_alpha * gate
            delta_beta = delta_beta * gate

        alpha_sig, _, _ = _scatter_sum_compact(delta_alpha, asu_ids)
        beta_sum, _, _ = _scatter_sum_compact(delta_beta, asu_ids)
        alpha_h = self.alpha_W + alpha_sig
        beta_h = tau_h + beta_sum

        if self.merge_overdispersion:
            # Per-HKL spread of the per-observation intensities -> a *detached*
            # calibration signal for phi. no_grad because (a) the spread must not
            # train the encoders to game it, and (b) it sidesteps the sqrt-at-0
            # gradient when an HKL has a single observation (var == 0), which
            # would otherwise be inf/NaN and crash the backward pass.
            with torch.no_grad():
                i_obs = delta_alpha / delta_beta.clamp(min=1e-6)
                sum_i, _, _ = _scatter_sum_compact(i_obs, asu_ids)
                sum_i2, _, _ = _scatter_sum_compact(i_obs * i_obs, asu_ids)
                mean_i = sum_i / cnt.clamp(min=1.0)
                var_i = (sum_i2 / cnt.clamp(min=1.0) - mean_i * mean_i).clamp(
                    min=0.0
                )
                cv = (var_i.sqrt() / mean_i.clamp(min=1e-6)).clamp(max=100.0)
                disp_in = torch.stack(
                    [mean_i.clamp(min=1e-6).log(), cv, cnt.log()], dim=-1
                )
            phi = F.softplus(self.disp_head(disp_in)).squeeze(-1)
            # Preserve the mean (alpha/beta), inflate the variance by (1+phi).
            alpha_h = alpha_h / (1.0 + phi)
            beta_h = beta_h / (1.0 + phi)

        qi_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))
        return qi_h, alpha_h, beta_h, inverse, unique, tau_h

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Recompute the per-HKL posterior over the full dataset.

        Requires a loader that yields COMPLETE HKL groups per batch -- i.e. a
        `group_by_asu_id` loader such as `predict_dataloader(grouped=True)`. With
        complete groups the per-batch `_merge` is the exact per-HKL posterior for
        every mode (sum/mean, attention, overdispersion). A guard raises if an
        HKL spans batches (an ungrouped loader), which would otherwise produce
        silent partial sums.
        """
        self.eval()
        device = self.alpha_buffer.device
        seen = torch.zeros(self.n_hkl, dtype=torch.bool, device=device)
        self.alpha_buffer.fill_(self.alpha_W)
        self.beta_buffer.fill_(1.0)

        for batch in dataloader:
            _, shoebox, mask, metadata = batch
            shoebox = shoebox.to(device)
            mask = mask.to(device)
            b = shoebox.shape[0]
            sr = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)
            x_k_i = self.encoders["k_i"](sr)
            x_r_i = self.encoders["r_i"](sr)
            scale = self._get_scale(metadata, device)
            asu = metadata["asu_id"].long().to(device)
            d_obs = metadata["d"].to(device).float()
            lp = metadata["lp"].to(device).float()
            profile_mean = None
            if self.merge_aggregation == "sum":
                profile_mean = self.surrogates["qp"](
                    self.encoders["profile"](sr), mc_samples=1
                ).mean_profile
            _, alpha_h, beta_h, _, unique, _ = self._merge(
                x_k_i, x_r_i, scale, profile_mean, mask, asu, d_obs, lp
            )
            if bool(seen[unique].any()):
                raise RuntimeError(
                    "finalize_merge requires a grouped (group_by_asu_id) loader "
                    "so each HKL is complete in one batch; found an HKL spanning "
                    "batches. Use predict_dataloader(grouped=True)."
                )
            self.alpha_buffer[unique] = alpha_h
            self.beta_buffer[unique] = beta_h
            seen[unique] = True
        self.buffer_seen.copy_(seen)

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma posterior from the merge buffers (for MTZ output)."""
        return Gamma(
            self.alpha_buffer.clamp(min=1e-6),
            self.beta_buffer.clamp(min=1e-12),
        )

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        b = shoebox.shape[0]
        device = shoebox.device
        shoebox_reshaped = (shoebox * mask).reshape(b, 1, *self.shoebox_shape)

        # Encoders (5): profile, k_i, r_i, k_bg, r_bg
        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)
        profile_mean = qp.mean_profile

        scale = self._get_scale(metadata, device)  # (B,)
        asu_ids = metadata["asu_id"].long().to(device)
        d_obs = metadata["d"].to(device).float()
        lp = metadata["lp"].to(device).float()

        qi_h, alpha_h, beta_h, inverse, unique_hkls, tau_h = self._merge(
            x_k_i, x_r_i, scale, profile_mean, mask, asu_ids, d_obs, lp
        )

        # Sample I_h once per HKL, broadcast to its observations.
        zI_h = qi_h.rsample([self.mc_samples]).clamp(
            min=1e-10
        )  # (S, n_unique)
        zI = zI_h[:, inverse]  # (S, B)
        zI_scaled = (scale.unsqueeze(0) * zI).unsqueeze(-1).permute(1, 0, 2)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)
        rate = zI_scaled * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        qi = Gamma(
            alpha_h[inverse].clamp(min=1e-6),
            beta_h[inverse].clamp(min=1e-12),
        )

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
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
            "qi": qi,
            "qbg": qbg,
            "qi_h": qi_h,
            "alpha_h": alpha_h,
            "beta_h": beta_h,
            "tau_h": tau_h,
            "inverse": inverse,
            "unique_hkls": unique_hkls,
        }

    def _wilson_kl_per_hkl(self, qi_h: Gamma, tau_h: Tensor) -> Tensor:
        """KL(q(I_h) || Gamma(alpha_W, tau_h)), counted once per HKL."""
        p_i = Gamma(
            self.alpha_W * torch.ones_like(tau_h), tau_h.clamp(min=1e-12)
        )
        return kl_divergence(qi_h, p_i)

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        qi_h = outputs["qi_h"]

        group_labels = metadata["group_label"].long()

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

        # ELBO-consistent weighting
        kl_i_per_hkl = self._wilson_kl_per_hkl(qi_h, outputs["tau_h"])
        kl_i = kl_i_per_hkl.sum() / counts.shape[0] * self.merge_kl_weight
        total_loss = total_loss + kl_i

        _log_loss(
            self,
            kl=loss_dict["kl_mean"] + kl_i,
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "kl_prf": loss_dict["kl_prf_mean"],
                "kl_bg": loss_dict["kl_bg_mean"],
                "kl_i_hkl": kl_i.detach(),
            },
        )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        with torch.no_grad():
            self.log(f"{step} qi_h_mean", qi_h.mean.mean(), on_epoch=True)
            self.log(f"{step} qi_h_var", qi_h.variance.mean(), on_epoch=True)
            self.log(
                f"{step} qi_h_k", qi_h.concentration.mean(), on_epoch=True
            )
            self.log(f"{step} qi_h_rate", qi_h.rate.mean(), on_epoch=True)
            n_unique = len(outputs["unique_hkls"])
            self.log(
                f"{step} n_unique_hkl",
                torch.tensor(float(n_unique)),
                on_epoch=True,
            )
            self.log(
                f"{step} obs_per_hkl",
                torch.tensor(counts.shape[0] / max(n_unique, 1)),
                on_epoch=True,
            )

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": (loss_dict["kl_mean"] + kl_i).detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": kl_i.detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

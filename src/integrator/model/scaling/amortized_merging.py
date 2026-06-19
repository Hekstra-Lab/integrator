"""Amortized per-HKL intensity merging (sum-mode structured VAE).

A recognition network emits a per-observation positive potential and they are
summed in natural-parameter space against the Wilson prior:

    alpha_h = alpha_W + sum_i  dalpha_i          (learned signal counts)
    beta_h  = tau_h   + sum_i  s_i * sum_p prf   (analytic exposure)
    q(I_h)  = Gamma(alpha_h, beta_h)

Sum (not mean) gives the correct `~1/N` precision scaling, and the prior
`(alpha_W, tau_h)` regularizes weak HKLs. The potential head is conditioned on
per-observation `[log scale, log lp, d]` so the merge is scale-aware. The
per-observation scale `s_i` is an MLP over `[frame, radius, d, lp]` (+ optional
even-l crystal-frame absorption); it carries the LP correction, so the Wilson
prior must not (`lp_correction: false`).

Pair with `GroupedAsuIdBatchSampler` (`group_by_asu_id: true`) so each batch
holds complete HKL groups; the sufficient-statistic sum is then complete within
the batch and no cross-batch buffer is needed during training. Optional
anomalous-preserving auxiliary losses (Friedel-pooled scaling consistency,
centric anchor, double-Wilson coupling) default off.
"""

import logging
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gamma, kl_divergence

from integrator import configs
from integrator.model.scaling.base import ScalingLightningModule
from integrator.model.scaling.config import MergingIntegratorCfg
from integrator.model.scaling.merge_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
    _log_loss,
    _sample_profile,
    _scatter_sum_compact,
)
from integrator.model.scaling.mlp_scale import ChebyshevScale, MLPScale

logger = logging.getLogger(__name__)

_N_COND = 3  # per-observation conditioning: [log scale, log lp, d]


class AmortizedMergingIntegrator(ScalingLightningModule):
    """Per-HKL amortized variational intensity (sum-mode structured VAE).

    See the module docstring. Best paired with `GroupedAsuIdBatchSampler`
    (`group_by_asu_id: true`). Standalone: its base is `ScalingLightningModule`
    and its loss is `MergingWilsonLoss` (no `BaseIntegrator`/`WilsonLoss`).
    """

    CFG_CLASS = MergingIntegratorCfg

    REQUIRED_ENCODERS = {
        "profile": ("profile_encoder", configs.ProfileEncoderArgs),
        "k_i": ("intensity_encoder", configs.IntensityEncoderArgs),
        "r_i": ("intensity_encoder", configs.IntensityEncoderArgs),
        "k_bg": ("intensity_encoder", configs.IntensityEncoderArgs),
        "r_bg": ("intensity_encoder", configs.IntensityEncoderArgs),
    }

    DEFAULT_SURROGATES = {
        "qp": {
            "name": "learned_basis_profile",
            "args": {"latent_dim": 12, "init_std": 0.5, "prior_scale": 3.0},
        },
        "qbg": {
            "name": "gamma",
            "args": {
                "reparameterization": "mean_fano",
                "eps": 1.0e-6,
                "k_min": 0.01,
            },
        },
    }

    def __init__(
        self,
        cfg: MergingIntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
        optimizer=None,
    ):
        super().__init__(cfg, loss, encoders, surrogates, optimizer)

        # LP is applied in the scale (MLP input / Chebyshev `/lp`), so I_h is the
        # LP-corrected intensity; lp_correction on the prior would double-count.
        if getattr(self.loss, "_apply_lp", False):
            raise ValueError(
                "AmortizedMergingIntegrator applies LP through the scale, so "
                "I_h is LP-corrected; enabling lp_correction would multiply the "
                "Wilson prior by LP too and double-count it. Set "
                "loss.args.lp_correction: false."
            )

        if cfg.n_hkl is None:
            raise ValueError(
                "n_hkl is required: set integrator.args.n_hkl, or ensure "
                "<data_dir>/dataset.yaml has an `n_hkl` block (re-run "
                "make_shoeboxes) so the factory can auto-fill it."
            )
        self.n_hkl = cfg.n_hkl
        self.alpha_W = float(cfg.wilson_alpha)
        self.merge_kl_weight = float(cfg.merge_kl_weight)
        self.consistency_weight = float(cfg.consistency_weight)
        self.consistency_pool_friedel = bool(cfg.consistency_pool_friedel)
        self.double_wilson_weight = float(cfg.double_wilson_weight)
        self.wilson_centric_prior = bool(cfg.wilson_centric_prior)

        # Anomalous run merges on the Friedel-SEPARATE id; non-anomalous on the
        # pooled id. The anomalous-preserving terms always pair on the pooled id.
        self.anomalous = bool(getattr(cfg, "anomalous", True))
        self.merge_key = (
            "miller_idx_unfriedelized"
            if self.anomalous
            else "miller_idx_friedelized"
        )
        self.friedel_key = "miller_idx_friedelized"

        d = cfg.encoder_out
        in_dim = 2 * d + _N_COND
        self.alpha_head = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, 1)
        )

        # Final merged per-HKL posterior, populated by `finalize_merge`.
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

        # Scale field: MLP (production) or a frame-only Chebyshev fallback.
        if cfg.scale_mlp:
            n_abs_sh = 0
            if cfg.scale_mlp_absorption:
                n_abs_sh = (int(cfg.scale_sh_lmax) + 1) ** 2 - 1
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=cfg.dmin,
                d_max=60.0,
                head_init_std=cfg.scale_head_init_std,
                n_abs_sh=n_abs_sh,
                absorption_even_only=cfg.scale_mlp_absorption_even_only,
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            a = None
            if self.scale_fn.n_abs_sh > 0:
                if "absorption_sh" not in metadata:
                    raise KeyError(
                        "MLPScale with scale_mlp_absorption needs "
                        "'absorption_sh' in metadata; point the loader's "
                        "reference at the SH-augmented metadata file."
                    )
                a = metadata["absorption_sh"].to(device).float()
            # The MLP owns the LP correction (lp is an input), so no `/lp` here.
            return self.scale_fn(frame, x_det, y_det, lp, d, a)
        return self.scale_fn(frame) / lp

    def _cond_mid(self, metadata: dict, device: torch.device) -> Tensor:
        """Middle conditioning feature for the merge: the LP factor (mono)."""
        return metadata["lp"].to(device).float()

    def _wilson_tau(self, d: Tensor) -> Tensor:
        """Wilson prior rate tau from resolution d (lp lives in the scale)."""
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _merge(
        self,
        x_k_i: Tensor,
        x_r_i: Tensor,
        scale: Tensor,
        profile_mean: Tensor,
        mask: Tensor,
        miller_idx: Tensor,
        d_per_obs: Tensor,
        cond_mid: Tensor,
    ) -> tuple[Gamma, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Merge per-observation features into per-HKL q(I_h).

        Returns `(qi_h, alpha_h, beta_h, inverse, unique, tau_h)`, where
        `inverse` maps each observation to its HKL row and `unique` are the HKL
        ids present.
        """
        d_sum, inverse, unique = _scatter_sum_compact(d_per_obs, miller_idx)
        cnt, _, _ = _scatter_sum_compact(torch.ones_like(d_per_obs), miller_idx)
        tau_h = self._wilson_tau((d_sum / cnt.clamp(min=1.0)).clamp(min=1e-6))

        cond = torch.stack(
            [
                scale.clamp(min=1e-8).log(),
                cond_mid.clamp(min=1e-8).log(),
                d_per_obs,
            ],
            dim=-1,
        )  # (B, 3): [log scale, log lp, d]
        feat = torch.cat([x_k_i, x_r_i, cond], dim=-1)  # (B, 2d+3)

        delta_alpha = F.softplus(self.alpha_head(feat)).squeeze(-1)  # (B,)
        delta_beta = scale * (profile_mean * mask).sum(dim=-1)  # exposure

        alpha_sig, _, _ = _scatter_sum_compact(delta_alpha, miller_idx)
        beta_sum, _, _ = _scatter_sum_compact(delta_beta, miller_idx)
        alpha_h = self.alpha_W + alpha_sig
        beta_h = tau_h + beta_sum

        qi_h = Gamma(alpha_h.clamp(min=1e-6), beta_h.clamp(min=1e-12))
        return qi_h, alpha_h, beta_h, inverse, unique, tau_h

    @torch.no_grad()
    def finalize_merge(self, dataloader) -> None:
        """Recompute the per-HKL posterior over the full dataset.

        Requires a loader that yields COMPLETE HKL groups per batch (a
        `group_by_asu_id` loader, e.g. `predict_dataloader(grouped=True)`); a
        guard raises if an HKL spans batches (which would give partial sums).
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
            miller_idx = metadata[self.merge_key].long().to(device)
            d_obs = metadata["d"].to(device).float()
            cond_mid = self._cond_mid(metadata, device)
            profile_mean = self.surrogates["qp"](
                self.encoders["profile"](sr), mc_samples=1
            ).mean_profile
            _, alpha_h, beta_h, _, unique, _ = self._merge(
                x_k_i, x_r_i, scale, profile_mean, mask, miller_idx, d_obs,
                cond_mid,
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

        # encode shoeboxes into representations
        x_profile = self.encoders["profile"](shoebox_reshaped)
        x_k_i = self.encoders["k_i"](shoebox_reshaped)
        x_r_i = self.encoders["r_i"](shoebox_reshaped)
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)

        # get bg/profile surrogates
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)
        qp = self.surrogates["qp"](x_profile, mc_samples=self.mc_samples)

        # get mean profile
        profile_mean = qp.mean_profile

        # get scale
        scale = self._get_scale(metadata, device)  # (B,)

        miller_idx = metadata[self.merge_key].long().to(device)
        d_obs = metadata["d"].to(device).float()
        cond_mid = self._cond_mid(metadata, device)

        qi_h, alpha_h, beta_h, inverse, unique_hkls, tau_h = self._merge(
            x_k_i, x_r_i, scale, profile_mean, mask, miller_idx, d_obs, cond_mid
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
        out[self.merge_key] = miller_idx

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
            "scale": scale,
        }

    # ------------------------------------------------------------------

    def _wilson_kl_per_hkl(
        self, qi_h: Gamma, tau_h: Tensor, centric: Tensor | None = None
    ) -> Tensor:
        """KL(q(I_h) || Wilson prior), counted once per HKL.

        Acentric reflections follow Gamma(alpha_W, tau_h) (mean 1/tau_h = Sigma);
        centric reflections follow the chi^2_1 form Gamma(alpha_W/2,
        (alpha_W/2)*tau_h) -- half the shape, mean-preserving rate. `centric=None`
        gives every HKL the acentric prior.
        """
        alpha = self.alpha_W * torch.ones_like(tau_h)
        if centric is not None:
            alpha = torch.where(centric, alpha * 0.5, alpha)
        p_i = Gamma(alpha, (alpha * tau_h).clamp(min=1e-12))
        return kl_divergence(qi_h, p_i)

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
        With the measured intensity J_i = sum_p (counts - bg) (data, no I_h) and
        the per-obs scale s_i, the closed-form WLS merge is
        I_hat_h = sum_i w_i J_i s_i / sum_i w_i s_i^2 (w_i = 1/var(J_i)), and the
        loss penalizes the weighted residual (J_i - s_i I_hat_h)^2. Only s_i
        carries gradient (J, w, I_hat detached) -> a clean per-obs scale signal
        that bypasses the learned I_h. Gauge-invariant; singleton HKLs give 0.
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
        i_hat = (num / den).detach()
        resid = J - scale * i_hat[inverse]
        return (w * resid.pow(2)).sum() / max(scale.shape[0], 1)

    def _centric_anchor_loss(
        self,
        counts: Tensor,
        mask: Tensor,
        bg: Tensor,
        scale: Tensor,
        metadata: dict,
        device: torch.device,
    ) -> Tensor:
        """Centric reflections as a zero-anomalous control on the scale.

        Centrics have I(+) == I(-) by symmetry, so any Bijvoet difference the
        scale produces on them is pure scale error. Penalize the mean squared
        fractional difference of the sign-split WLS intensity. J_i and w_i are
        detached, so only `scale` carries gradient. Needs both mates of a centric
        in the batch (`group_by_key: nonanom_id`).
        """
        if not {"centric", "friedel_plus", self.friedel_key} <= set(metadata):
            return scale.new_zeros(())
        centric = metadata["centric"].bool().to(device)
        if int(centric.sum()) < 2:
            return scale.new_zeros(())
        plus = metadata["friedel_plus"].bool().to(device)
        nonanom = metadata[self.friedel_key].long().to(device)

        cm = counts.clamp(min=0).to(device) * mask.to(device)
        J = (cm - bg.unsqueeze(-1) * mask.to(device)).sum(dim=-1).detach()
        w = (1.0 / cm.sum(dim=-1).clamp(min=1.0)).detach()

        sel = centric
        s = scale[sel]
        js, ws = J[sel], w[sel]
        # (group, sign) key -> sign-split WLS intensity (keeps the scale grad).
        key = nonanom[sel] * 2 + plus[sel].long()
        uniq_key, key_inv = torch.unique(key, return_inverse=True)
        k = uniq_key.numel()
        num = torch.zeros(k, device=device).scatter_add(
            0, key_inv, ws * js * s
        )
        den = (
            torch.zeros(k, device=device)
            .scatter_add(0, key_inv, ws * s * s)
            .clamp(min=1e-12)
        )
        i_hat = num / den  # (K,), differentiable in scale

        grp = uniq_key // 2
        is_plus = uniq_key % 2 == 1
        _, g_inv = torch.unique(grp, return_inverse=True)
        gn = int(g_inv.max().item()) + 1

        def _by_sign(vals: Tensor, m: Tensor) -> Tensor:
            return torch.zeros(gn, device=device).scatter_add(
                0, g_inv[m], vals[m]
            )

        ones = torch.ones_like(i_hat)
        ihat_p, ihat_m = _by_sign(i_hat, is_plus), _by_sign(i_hat, ~is_plus)
        cnt_p, cnt_m = _by_sign(ones, is_plus), _by_sign(ones, ~is_plus)
        valid = (cnt_p > 0) & (cnt_m > 0)
        if not bool(valid.any()):
            return scale.new_zeros(())
        ip, im = ihat_p[valid], ihat_m[valid]
        denom = (0.5 * (ip + im)).detach().clamp(min=1e-6)
        return ((ip - im) / denom).pow(2).mean()

    def _double_wilson_coupling(
        self,
        qi_h: Gamma,
        inverse: Tensor,
        metadata: dict,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        """Double-Wilson coupling of Friedel mates (tractable surrogate).

        Adds a zero-mean Normal prior on the log-ratio of paired mates' posterior
        means, `L = sum_pairs (log E[I_+] - log E[I_-])^2`, shrinking noise-driven
        Bijvoet differences toward zero while the likelihood keeps the real
        signal (Dalton, Greisman & Hekstra, Nat. Commun. 2024). Mates paired by
        `nonanom_id`. Returns `(sum_sq_log_ratio, n_pairs)`.
        """
        if not {self.friedel_key, "friedel_plus"} <= set(metadata):
            return qi_h.mean.new_zeros(()), 0
        n_rows = qi_h.mean.shape[0]
        nonanom = metadata[self.friedel_key].long().to(device)
        plus = metadata["friedel_plus"].bool().to(device)
        # nonanom id and sign are constant within an asu group (row), so an
        # index-assign from its observations is well-defined.
        row_nonanom = torch.empty(n_rows, dtype=torch.long, device=device)
        row_nonanom[inverse] = nonanom
        row_plus = torch.zeros(n_rows, dtype=torch.bool, device=device)
        row_plus[inverse] = plus

        logmu = qi_h.mean.clamp(min=1e-10).log()
        _, na_inv = torch.unique(row_nonanom, return_inverse=True)
        g = int(na_inv.max().item()) + 1

        def _agg(m: Tensor) -> tuple[Tensor, Tensor]:
            s = torch.zeros(g, device=device).scatter_add(
                0, na_inv[m], logmu[m]
            )
            c = torch.zeros(g, device=device).scatter_add(
                0, na_inv[m], torch.ones_like(logmu[m])
            )
            return s, c

        sum_p, cnt_p = _agg(row_plus)
        sum_m, cnt_m = _agg(~row_plus)
        valid = (cnt_p > 0) & (cnt_m > 0)
        if not bool(valid.any()):
            return qi_h.mean.new_zeros(()), 0
        diff = (sum_p / cnt_p.clamp(min=1.0))[valid] - (
            sum_m / cnt_m.clamp(min=1.0)
        )[valid]
        return diff.pow(2).sum(), int(valid.sum())

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        qi_h = outputs["qi_h"]

        group_labels = (
            metadata["group_label"].long()
            if "group_label" in metadata
            else None
        )

        # Per-observation terms: Poisson NLL + profile KL + background KL.
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
        )
        total_loss = loss_dict["loss"]

        # Per-HKL Wilson intensity KL (one term per unique reflection), put on
        # the per-observation scale by dividing the sum by the obs count.
        row_centric = None
        if self.wilson_centric_prior and "centric" in metadata:
            tau_h = outputs["tau_h"]
            centric_obs = metadata["centric"].bool().to(tau_h.device)
            row_centric = torch.zeros(
                tau_h.shape[0], dtype=torch.bool, device=tau_h.device
            )
            row_centric[outputs["inverse"]] = centric_obs
        kl_i_per_hkl = self._wilson_kl_per_hkl(
            qi_h, outputs["tau_h"], row_centric
        )
        kl_i = kl_i_per_hkl.sum() / counts.shape[0] * self.merge_kl_weight
        total_loss = total_loss + kl_i

        scale_dev = outputs["scale"].device
        if self.consistency_weight > 0.0:
            cons_inverse = outputs["inverse"]
            cons_n = outputs["unique_hkls"].shape[0]
            if self.consistency_pool_friedel and self.friedel_key in metadata:
                nonanom = metadata[self.friedel_key].long().to(scale_dev)
                _, cons_inverse = torch.unique(nonanom, return_inverse=True)
                cons_n = int(cons_inverse.max().item()) + 1
            consist = self._consistency_loss(
                forward_out["counts"],
                forward_out["mask"],
                outputs["qbg"].mean,
                outputs["scale"],
                cons_inverse,
                cons_n,
            )
            total_loss = total_loss + self.consistency_weight * consist
            self.log(f"{step} consistency", consist.detach(), on_epoch=True)

        if self.double_wilson_weight > 0.0:
            dw_sum, n_pairs = self._double_wilson_coupling(
                qi_h, outputs["inverse"], metadata, scale_dev
            )
            if n_pairs > 0:
                dw_mean = dw_sum / n_pairs
                total_loss = total_loss + self.double_wilson_weight * dw_mean
                self.log(
                    f"{step} double_wilson", dw_mean.detach(), on_epoch=True
                )

        _log_loss(
            self,
            kl=loss_dict["kl_mean"] + kl_i,
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                "prf": loss_dict["kl_prf_mean"],
                "bg": loss_dict["kl_bg_mean"],
                "i_hkl": kl_i.detach(),
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

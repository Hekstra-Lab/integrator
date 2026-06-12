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

import logging
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
    LaueMLPScale,
    MLPScale,
    PhysicalScale,
    SpatialChebyshevScale,
)
from integrator.model.scaling.conjugate_merging import _scatter_sum_compact
from integrator.model.scaling.deepsets_merging import _scatter_mean_compact

logger = logging.getLogger(__name__)

_N_COND = 3  # per-observation conditioning: [log scale, log lp|wavelength, d]


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
        # Optional decoupled LR for the scale field (its own Adam group); None=lr.
        self.scaling_lr = getattr(cfg, "scaling_lr", None)
        # Cross-observation scaling-consistency loss weight (0 = off). Mirrors
        # ConjugateMergingIntegrator: a direct, data-only gradient for the per-obs
        # scale via internal consistency of symmetry-equivalents.
        self.consistency_weight = float(getattr(cfg, "consistency_weight", 0.0))
        # Anomalous-preserving terms (see configs.IntegratorCfg):
        #   pool_friedel  - consistency WLS target over the Friedel-pooled group.
        #   centric_anchor- zero-anomalous control on the scale from centrics.
        #   double_wilson - log-ratio coupling of paired mates' merged means.
        self.consistency_pool_friedel = bool(
            getattr(cfg, "consistency_pool_friedel", False)
        )
        self.centric_anchor_weight = float(
            getattr(cfg, "centric_anchor_weight", 0.0)
        )
        self.double_wilson_weight = float(
            getattr(cfg, "double_wilson_weight", 0.0)
        )
        self.wilson_centric_prior = bool(
            getattr(cfg, "wilson_centric_prior", False)
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
        self.scale_restraint_weight = float(
            getattr(cfg, "scale_absorption_restraint", 0.0)
        )
        if getattr(cfg, "scale_laue_mlp", False):
            # Polychromatic (Laue stills): wavelength-aware MLP scale. Inputs
            # [lambda, x, y, d] (+ optional crystal-frame SH absorption when
            # scale_mlp_absorption); the MLP owns spectrum + geometry, no /lp by
            # the caller. Optional per-image log-scale. Precedence over rotation
            # scales.
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
                d_max=60.0,
                n_images=getattr(cfg, "scale_n_images", None),
                head_init_std=getattr(cfg, "scale_head_init_std", 0.0),
                n_abs_sh=n_abs_sh,
                absorption_even_only=getattr(
                    cfg, "scale_mlp_absorption_even_only", True
                ),
            )
        elif getattr(cfg, "scale_physical", False):
            # DIALS-style: smooth scale(phi) x decay(phi,d) x crystal-frame SH
            # absorption (precomputed `absorption_sh`, scripts/
            # extract_crystal_frame_sh.py). scale_sh_lmax MUST match --lmax. Here
            # the scale lives in the conjugate merge (no warm-start/freeze), so
            # the surface is the under-identified within-image band the MLP left
            # at r_detrended~0; restrain it (scale_absorption_restraint).
            self.scale_fn = PhysicalScale(
                n_sh=(int(cfg.scale_sh_lmax) + 1) ** 2 - 1,
                degree_scale=cfg.scale_degree,
                degree_decay=cfg.scale_degree_decay,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                absorption_init_std=cfg.scale_absorption_init_std,
            )
            if self.scale_restraint_weight == 0.0:
                logger.warning(
                    "scale_physical with scale_absorption_restraint=0: the "
                    "absorption surface is bounded only by its low dimension. "
                    "Set scale_absorption_restraint (~1e-2) to oppose run-away "
                    "and protect the anomalous (odd-l) band."
                )
        elif cfg.scale_mlp:
            n_abs_sh = 0
            if getattr(cfg, "scale_mlp_absorption", False):
                n_abs_sh = (int(cfg.scale_sh_lmax) + 1) ** 2 - 1
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
                n_abs_sh=n_abs_sh,
                absorption_even_only=getattr(
                    cfg, "scale_mlp_absorption_even_only", True
                ),
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

        # End-of-epoch model-vs-DIALS intensity scatter (merge-quality readout).
        self.log_intensity_scatter = bool(
            getattr(cfg, "log_intensity_scatter", False)
        )
        self._iscatter_model: list[Tensor] = []
        self._iscatter_dials: list[Tensor] = []

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam; with scaling_lr set, the scale field gets its own param group.

        Mirrors ConjugateMergingIntegrator: the per-frame scale is the slowest-
        identified field, so a decoupled scaling_lr lets it equilibrate without
        raising the encoder LR. scaling_lr=None defers to the base optimizer
        (byte-identical). Preserves the decoder_weight_decay group; any LambdaLR
        warmup scales all groups uniformly.
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

    # ------------------------------------------------------------------

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        if isinstance(self.scale_fn, LaueMLPScale):
            # Polychromatic stills: the MLP owns the whole scale (incl. spectrum
            # and LP/Lorentz), so no /lp here -- there is no lp column.
            wavelength = metadata["wavelength"].to(device).float()
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            image_num = None
            if self.scale_fn.n_images > 0:
                image_num = metadata["image_num"].to(device).long()
            a = None
            if self.scale_fn.n_abs_sh > 0:
                if "absorption_sh" not in metadata:
                    raise KeyError(
                        "LaueMLPScale with scale_mlp_absorption needs "
                        "'absorption_sh' in metadata; run a stills variant of "
                        "scripts/extract_crystal_frame_sh.py and point the "
                        "loader's reference at the augmented metadata file."
                    )
                a = metadata["absorption_sh"].to(device).float()
            return self.scale_fn(
                wavelength, x_det, y_det, d, image_num, a
            )
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, PhysicalScale):
            if "absorption_sh" not in metadata:
                raise KeyError(
                    "PhysicalScale needs 'absorption_sh' in metadata; run "
                    "scripts/extract_crystal_frame_sh.py and point the data "
                    "loader's reference at the augmented metadata file."
                )
            d = metadata["d"].to(device).float()
            a = metadata["absorption_sh"].to(device).float()
            return self.scale_fn(frame, d, a) / lp
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            a = None
            if self.scale_fn.n_abs_sh > 0:
                if "absorption_sh" not in metadata:
                    raise KeyError(
                        "MLPScale with scale_mlp_absorption needs 'absorption_sh' "
                        "in metadata; run scripts/extract_crystal_frame_sh.py and "
                        "point the data loader's reference at metadata_sh.pt."
                    )
                a = metadata["absorption_sh"].to(device).float()
            return self.scale_fn(frame, x_det, y_det, lp, d, a)
        elif isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            return self.scale_fn(frame, x_det, y_det) / lp
        else:
            return self.scale_fn(frame) / lp

    def _cond_mid(self, metadata: dict, device: torch.device) -> Tensor:
        """Middle per-observation conditioning feature for the merge.

        The LP factor for monochromatic data; the wavelength for polychromatic
        (Laue) data, which has no `lp` column and whose scale already carries the
        LP/Lorentz correction.
        """
        if isinstance(self.scale_fn, LaueMLPScale):
            return metadata["wavelength"].to(device).float()
        return metadata["lp"].to(device).float()

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
        cond_mid: Tensor,
    ) -> tuple[Gamma, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Merge per-observation features into per-HKL q(I_h).

        Returns `(qi_h, alpha_h, beta_h, inverse, unique, tau_h)`, where
        `inverse` maps each observation to its HKL row and `unique` are the HKL
        ids present. `cond_mid` is the middle per-observation conditioning
        feature, logged into `cond`: the LP factor for monochromatic data, the
        wavelength for polychromatic (Laue) data.
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
            [
                scale.clamp(min=1e-8).log(),
                cond_mid.clamp(min=1e-8).log(),
                d_per_obs,
            ],
            dim=-1,
        )  # (B, 3): [log scale, log lp|wavelength, d]
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
            cond_mid = self._cond_mid(metadata, device)
            profile_mean = None
            if self.merge_aggregation == "sum":
                profile_mean = self.surrogates["qp"](
                    self.encoders["profile"](sr), mc_samples=1
                ).mean_profile
            _, alpha_h, beta_h, _, unique, _ = self._merge(
                x_k_i, x_r_i, scale, profile_mean, mask, asu, d_obs, cond_mid
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
        cond_mid = self._cond_mid(metadata, device)

        qi_h, alpha_h, beta_h, inverse, unique_hkls, tau_h = self._merge(
            x_k_i, x_r_i, scale, profile_mean, mask, asu_ids, d_obs, cond_mid
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
            "scale": scale,
        }

    def _wilson_kl_per_hkl(
        self, qi_h: Gamma, tau_h: Tensor, centric: Tensor | None = None
    ) -> Tensor:
        """KL(q(I_h) || Wilson prior), counted once per HKL.

        Acentric reflections follow the exponential Wilson distribution
        Gamma(alpha_W, tau_h) (mean 1/tau_h = Sigma); centric reflections follow
        the chi^2_1 form Gamma(alpha_W/2, (alpha_W/2)*tau_h) -- half the shape and
        a rate scaled to hold the same mean, so they get the correct heavier tail
        (twice the normalized variance) instead of the acentric exponential. The
        rate is alpha*tau_h (mean-preserving), identical to the legacy
        `Gamma(alpha_W, tau_h)` at the default alpha_W=1. `centric=None` gives
        every HKL the acentric prior (legacy behavior).
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

        Identical to ConjugateMergingIntegrator._consistency_loss: symmetry-
        equivalent observations of an HKL must agree after scaling. With the
        measured intensity J_i = sum_p (counts - bg) (data, no I_h) and the
        per-obs scale s_i, the closed-form WLS merge is
        I_hat_h = sum_i w_i J_i s_i / sum_i w_i s_i^2 (w_i = 1/var(J_i)), and the
        loss penalizes the weighted residual (J_i - s_i I_hat_h)^2. Only s_i
        carries gradient (J, w, I_hat detached) -> a clean per-obs scale signal
        that bypasses the learned I_h. Gauge-invariant; singleton HKLs give 0.
        Group by the anomalous asu_id (inverse) so it tightens the within-mate
        scale the anomalous signal needs.
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

        Centric reflections have I(+) == I(-) by symmetry, so any Bijvoet
        difference the scale produces on them is pure scale error. For each
        centric `nonanom_id` the sign-split data-only WLS merged intensity is
        I_hat_s = sum_{i in s} w_i J_i s_i / sum_{i in s} w_i s_i^2 for the two
        signs s in {+, -}, and we penalize the mean squared FRACTIONAL
        difference `((I_hat_+ - I_hat_-) / mean)^2`. J_i and w_i are detached
        (data), so only the per-obs `scale` carries gradient: the scale is
        pushed to reproduce equal mates where the truth is known to be zero. The
        logged value is the RMS fake-anomalous-on-centrics. Needs both mates of a
        centric in the batch (`group_by_key: nonanom_id`); single-sign groups are
        skipped, as is a batch with < 2 centric observations.
        """
        if not {"centric", "friedel_plus", "nonanom_id"} <= set(metadata):
            return scale.new_zeros(())
        centric = metadata["centric"].bool().to(device)
        if int(centric.sum()) < 2:
            return scale.new_zeros(())
        plus = metadata["friedel_plus"].bool().to(device)
        nonanom = metadata["nonanom_id"].long().to(device)

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
        num = torch.zeros(k, device=device).scatter_add(0, key_inv, ws * js * s)
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

        The double-Wilson prior (Dalton, Greisman & Hekstra, Nat. Commun. 2024)
        models F(+) and F(-) as bivariate-normal with correlation r -> 1, so the
        anomalous difference is small. On top of the per-HKL marginal Wilson KL
        this adds the conditional factor: a zero-mean Normal prior on the
        log-ratio of paired mates' posterior means,

            L_couple = sum_pairs (log E[I_+] - log E[I_-])^2,

        a log-normal coupling with anomalous variance sigma^2 = 1 / (2 w). It
        shrinks noise-driven Bijvoet differences toward zero while the likelihood
        keeps the real signal. Mates are paired by `nonanom_id`; both share a
        batch under a `group_by_key: nonanom_id` loader. Returns
        `(sum_sq_log_ratio, n_pairs)` so the caller can take the mean.
        """
        if not {"nonanom_id", "friedel_plus"} <= set(metadata):
            return qi_h.mean.new_zeros(()), 0
        n_rows = qi_h.mean.shape[0]
        nonanom = metadata["nonanom_id"].long().to(device)
        plus = metadata["friedel_plus"].bool().to(device)
        # Per-row (asu group) nonanom id and sign are constant within the group,
        # so an index-assign from its observations is well-defined.
        row_nonanom = torch.empty(n_rows, dtype=torch.long, device=device)
        row_nonanom[inverse] = nonanom
        row_plus = torch.zeros(n_rows, dtype=torch.bool, device=device)
        row_plus[inverse] = plus

        logmu = qi_h.mean.clamp(min=1e-10).log()
        _, na_inv = torch.unique(row_nonanom, return_inverse=True)
        g = int(na_inv.max().item()) + 1

        def _agg(m: Tensor) -> tuple[Tensor, Tensor]:
            s = torch.zeros(g, device=device).scatter_add(0, na_inv[m], logmu[m])
            c = torch.zeros(g, device=device).scatter_add(
                0, na_inv[m], torch.ones_like(logmu[m])
            )
            return s, c

        sum_p, cnt_p = _agg(row_plus)
        sum_m, cnt_m = _agg(~row_plus)
        valid = (cnt_p > 0) & (cnt_m > 0)
        if not bool(valid.any()):
            return qi_h.mean.new_zeros(()), 0
        # An asu group is purely + or -, so cnt is 0/1; the divide just recovers
        # the single paired row's log-mean.
        diff = (sum_p / cnt_p.clamp(min=1.0))[valid] - (
            sum_m / cnt_m.clamp(min=1.0)
        )[valid]
        return diff.pow(2).sum(), int(valid.sum())

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

        # ELBO-consistent weighting. With the centric Wilson prior on, reduce the
        # per-obs `centric` flag to a per-HKL-row flag (constant within an asu
        # group) so centric reflections get the chi^2_1 prior.
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

        # Scaling-consistency loss: a direct, data-only gradient for the per-obs
        # scale. Grouped by the anomalous asu_id (`inverse`) by default, or by
        # the Friedel-POOLED group when consistency_pool_friedel is set, which
        # identifies the scale against the +/- pooled mean so it cannot absorb
        # the Bijvoet difference.
        scale_dev = outputs["scale"].device
        if self.consistency_weight > 0.0:
            cons_inverse = outputs["inverse"]
            cons_n = outputs["unique_hkls"].shape[0]
            if self.consistency_pool_friedel and "nonanom_id" in metadata:
                nonanom = metadata["nonanom_id"].long().to(scale_dev)
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
            self.log(
                f"{step} consistency",
                consist.detach(),
                on_step=False,
                on_epoch=True,
            )

        # Centric anchoring (1): pin the per-obs scale on centrics, where the
        # true anomalous is zero. Logs the RMS fake-anomalous-on-centrics.
        if self.centric_anchor_weight > 0.0:
            anchor = self._centric_anchor_loss(
                forward_out["counts"],
                forward_out["mask"],
                outputs["qbg"].mean,
                outputs["scale"],
                metadata,
                scale_dev,
            )
            total_loss = total_loss + self.centric_anchor_weight * anchor
            self.log(
                f"{step} centric_anchor",
                anchor.detach(),
                on_step=False,
                on_epoch=True,
            )

        # Double-Wilson coupling (4): shrink the anomalous log-difference of
        # paired mates. Logs mean (log E[I_+] - log E[I_-])^2 (anomalous spread).
        if self.double_wilson_weight > 0.0:
            dw_sum, n_pairs = self._double_wilson_coupling(
                qi_h, outputs["inverse"], metadata, scale_dev
            )
            if n_pairs > 0:
                dw_mean = dw_sum / n_pairs
                total_loss = total_loss + self.double_wilson_weight * dw_mean
                self.log(
                    f"{step} double_wilson",
                    dw_mean.detach(),
                    on_step=False,
                    on_epoch=True,
                )

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

        # PhysicalScale absorption/decay restraint: opposes the merge/consistency
        # pulling the surface to absorb the anomalous signal (and run away).
        if (
            isinstance(self.scale_fn, PhysicalScale)
            and self.scale_restraint_weight > 0.0
        ):
            restraint = self.scale_restraint_weight * self.scale_fn.restraint_penalty()
            total_loss = total_loss + restraint
            self.log(
                f"{step} scale_restraint", restraint.detach(), on_epoch=True
            )

        with torch.no_grad():
            if isinstance(self.scale_fn, PhysicalScale):
                self.log(
                    f"{step} abs_c_rms",
                    self.scale_fn.absorption_c.detach().pow(2).mean().sqrt(),
                    on_epoch=True,
                )
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

        if (
            step == "train"
            and self.log_intensity_scatter
            and "intensity.sum.value" in metadata
        ):
            self._collect_intensity_scatter(outputs, metadata)

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

    # ------------------------------------------------------------------
    # End-of-epoch model-vs-DIALS intensity scatter
    # ------------------------------------------------------------------

    def _collect_intensity_scatter(
        self, outputs: dict, metadata: dict, max_per_batch: int = 64
    ) -> None:
        """Stash a per-batch subsample of (model scale*I_h, DIALS intensity)."""
        with torch.no_grad():
            model_i = (
                outputs["scale"] * outputs["qi"].mean
            ).detach().float().cpu()
            dials_i = metadata["intensity.sum.value"].detach().float().cpu()
            n = model_i.shape[0]
            k = min(max_per_batch, n)
            if k <= 0:
                return
            idx = torch.randperm(n)[:k]
            self._iscatter_model.append(model_i[idx])
            self._iscatter_dials.append(dials_i[idx])

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if not (self.log_intensity_scatter and self._iscatter_model):
            return
        import numpy as np

        model_i = torch.cat(self._iscatter_model).numpy()
        dials_i = torch.cat(self._iscatter_dials).numpy()
        self._iscatter_model.clear()
        self._iscatter_dials.clear()
        if len(model_i) > 5000:
            sel = np.random.choice(len(model_i), 5000, replace=False)
            model_i, dials_i = model_i[sel], dials_i[sel]
        self._plot_intensity_scatter(model_i, dials_i)

    def _plot_intensity_scatter(self, model_i, dials_i) -> None:
        """Log-log scatter of model scale*I_h vs DIALS intensity.sum.value."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception:
            return

        keep = (
            (model_i > 0)
            & (dials_i > 0)
            & np.isfinite(model_i)
            & np.isfinite(dials_i)
        )
        mi, di = model_i[keep], dials_i[keep]
        if len(mi) < 10:
            return
        log_cc = float(np.corrcoef(np.log(mi), np.log(di))[0, 1])

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(di, mi, s=4, alpha=0.3, edgecolors="none")
        lo, hi = float(min(di.min(), mi.min())), float(max(di.max(), mi.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("DIALS intensity.sum.value")
        ax.set_ylabel(r"model  scale $\cdot$ $I_h$")
        ax.set_title(
            f"epoch {self.current_epoch}  log-CC={log_cc:.3f}  n={len(mi)}"
        )
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()

        try:
            import wandb

            if self.logger is not None and hasattr(self.logger, "experiment"):
                self.logger.experiment.log(
                    {
                        "intensity_vs_dials": wandb.Image(fig),
                        "intensity_logCC": log_cc,
                        "epoch": self.current_epoch,
                    }
                )
        except Exception:
            pass
        try:
            from pathlib import Path

            save_dir = getattr(self.logger, "save_dir", None)
            out = Path(save_dir or ".") / "intensity_scatter"
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"epoch{self.current_epoch:04d}.png", dpi=110)
        except Exception:
            pass
        plt.close(fig)

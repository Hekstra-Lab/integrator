import logging
import math
from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Gamma, kl_divergence

from integrator.configs import OptimizerConfig
from integrator.model.scaling.config import MergingIntegratorCfg
from integrator.model.scaling.merge_utils import _log_loss
from integrator.model.scaling.mlp_scale import (
    ChebyshevScale,
    CoarseScale,
    LinearScale,
    MLPScale,
    ResNetScale,
    SolvedScale,
)
from integrator.model.scaling.scatter_logger import ScatterLogger

_logger = logging.getLogger(__name__)


def _sh_n_cols(use_lmax: int, even_only: bool) -> int:
    """Number of `absorption_sh` columns kept for `l=1..use_lmax` (+even filter)."""
    return sum(
        2 * l + 1
        for l in range(1, use_lmax + 1)
        if (not even_only or l % 2 == 0)
    )


def _sh_select_idx(extract_lmax: int, use_lmax: int, even_only: bool) -> list[int]:
    idx, c = [], 0
    for l in range(1, extract_lmax + 1):
        for _m in range(-l, l + 1):
            if l <= use_lmax and (not even_only or l % 2 == 0):
                idx.append(c)
            c += 1
    return idx

# Per-obs columns written during prediction (model + passed-through DIALS)
DEFAULT_PREDICT_KEYS = [
    "refl_ids",
    "is_test",
    "qi_mean",
    "qi_var",
    "qbg_mean",
    "qbg_var",
    "scaled_intensity",
    "intensity.prf.value",
    "intensity.prf.variance",
    "background.mean",
    "d",
    "group_label",
    "miller_idx_friedelized",
    "miller_idx_unfriedelized",
]


class ScalingLightningModule(ScatterLogger, pl.LightningModule):
    """Lightning base for the merging integrators."""

    REQUIRED_ENCODERS: dict[str, tuple[str, type]] = {}
    DEFAULT_SURROGATES: dict[str, dict] = {}
    CFG_CLASS: type = MergingIntegratorCfg

    def __init__(
        self,
        cfg: MergingIntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
        optimizer: OptimizerConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg

        opt = optimizer or OptimizerConfig()
        self.optimizer_cfg = opt
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.decoder_weight_decay = opt.decoder_weight_decay
        self.lr_schedule = opt.lr_schedule
        self.warmup_epochs = opt.warmup_epochs
        self.warmup_steps = opt.warmup_steps
        self.lr_min = opt.lr_min

        # Decoupled scale-field LR lives on the merger's own cfg.
        self.scaling_lr = getattr(cfg, "scaling_lr", None)

        self.mc_samples = cfg.mc_samples
        if cfg.data_dim == "2d":
            self.shoebox_shape = (cfg.h, cfg.w)
        else:
            self.shoebox_shape = (cfg.d, cfg.h, cfg.w)

        self.predict_keys = (
            DEFAULT_PREDICT_KEYS
            if cfg.predict_keys == "default"
            else cfg.predict_keys
        )

        self.encoders = nn.ModuleDict(encoders)
        self.surrogates = nn.ModuleDict(surrogates)
        self.loss = loss

        self._init_scatter_logger(cfg)
        self.automatic_optimization = True

        self._init_merge(cfg)

    def _init_merge(self, cfg: MergingIntegratorCfg) -> None:
        """Shared merge setup: Wilson prior, merge buffers, and the scale field.

        The merging integrators apply LP through the scale, so I_h is the LP-corrected intensity.
        """
        if cfg.n_hkl is None:
            raise ValueError(
                "n_hkl is required: set integrator.args.n_hkl, or ensure "
                "<data_dir>/dataset.yaml has an `n_hkl` block."
            )
        self.n_hkl = cfg.n_hkl
        self.alpha_W = float(cfg.wilson_alpha)
        self.merge_kl_weight = float(cfg.merge_kl_weight)
        self.wilson_centric_prior = bool(cfg.wilson_centric_prior)

        # Anomalous run merges on the Friedel-SEPARATE id; non-anomalous on the
        # pooled id.
        self.anomalous = bool(getattr(cfg, "anomalous", True))
        self.merge_key = (
            "miller_idx_unfriedelized"
            if self.anomalous
            else "miller_idx_friedelized"
        )
        self.friedel_key = "miller_idx_friedelized"

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

        # Extra metadata columns fed to the MLP scale as additional inputs.
        self.scale_extra_features = list(cfg.scale_extra_features or [])

        scale_mode = cfg.scale_mode or ("mlp" if cfg.scale_mlp else "chebyshev")
        if scale_mode == "mlp":
            self.scale_fn = MLPScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_layers=cfg.scale_mlp_layers,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=cfg.d_min,
                d_max=60.0,
                head_init_std=cfg.scale_head_init_std,
                n_extra=len(self.scale_extra_features),
                extra_loc=cfg.scale_extra_loc,
                extra_scale=cfg.scale_extra_scale,
                friedel_safe=cfg.scale_mlp_friedel_safe,
            )
        elif scale_mode == "resnet":
            self._absorption_lmax = int(getattr(cfg, "scale_absorption_lmax", 0))
            self._absorption_even_only = bool(
                getattr(cfg, "scale_absorption_even_only", True)
            )
            n_abs = (
                _sh_n_cols(self._absorption_lmax, self._absorption_even_only)
                if self._absorption_lmax > 0
                else 0
            )
            self._absorption_idx: Tensor | None = None
            self.scale_fn = ResNetScale(
                hidden_dim=cfg.scale_mlp_hidden,
                n_blocks=int(getattr(cfg, "scale_resnet_blocks", 4)),
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_max=cfg.scale_r_max,
                d_min=cfg.d_min,
                d_max=60.0,
                n_absorption=n_abs,
                use_xy=bool(getattr(cfg, "scale_resnet_use_xy", True)),
                head_init_std=cfg.scale_head_init_std,
            )
        elif scale_mode == "coarse":
            self.scale_fn = CoarseScale(
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                k_degree=cfg.scale_degree,
                decay_degree=cfg.scale_decay_degree,
            )
        elif scale_mode == "chebyshev":
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )
        elif scale_mode == "solved":
            self._absorption_lmax = int(getattr(cfg, "scale_absorption_lmax", 0))
            self._absorption_even_only = bool(
                getattr(cfg, "scale_absorption_even_only", True)
            )
            n_abs = (
                _sh_n_cols(self._absorption_lmax, self._absorption_even_only)
                if self._absorption_lmax > 0
                else 0
            )
            self._absorption_idx: Tensor | None = None  # built on first batch
            self.scale_fn = SolvedScale(
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                k_degree=cfg.scale_degree,
                decay_degree=cfg.scale_decay_degree,
                ridge=getattr(cfg, "scale_ridge", 1e-3),
                n_absorption=n_abs,
            )
        elif scale_mode == "linear":
            self._absorption_lmax = int(getattr(cfg, "scale_absorption_lmax", 0))
            self._absorption_even_only = bool(
                getattr(cfg, "scale_absorption_even_only", True)
            )
            n_abs = (
                _sh_n_cols(self._absorption_lmax, self._absorption_even_only)
                if self._absorption_lmax > 0
                else 0
            )
            self._absorption_idx: Tensor | None = None
            self.scale_fn = LinearScale(
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                k_degree=cfg.scale_degree,
                decay_degree=cfg.scale_decay_degree,
                n_absorption=n_abs,
                hidden=int(getattr(cfg, "scale_linear_hidden", 0)),
                n_layers=int(getattr(cfg, "scale_linear_layers", 2)),
                n_images=int(getattr(cfg, "scale_n_images", 0)),
                lambda_modes=int(getattr(cfg, "scale_lambda_modes", 0)),
                lambda_min=float(getattr(cfg, "scale_lambda_min", 0.0)),
                lambda_max=float(getattr(cfg, "scale_lambda_max", 1.0)),
                head_init_std=cfg.scale_head_init_std,
            )
        else:
            raise ValueError(
                f"Unknown scale_mode {scale_mode!r}; expected 'mlp', 'coarse', "
                "'chebyshev', 'solved', 'linear', or 'resnet'."
            )
        self.scale_solve_warmup = int(getattr(cfg, "scale_solve_warmup", 2))

    def _get_scale(self, metadata: dict, device: torch.device) -> Tensor:
        frame = metadata["xyzcal.px.2"].to(device).float()
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        if isinstance(self.scale_fn, ResNetScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            absn = self._absorption(metadata, device)
            # lp is an input FEATURE here -- the net learns the full scale, no /lp.
            return self.scale_fn(frame, x_det, y_det, lp, d, absn)
        if isinstance(self.scale_fn, MLPScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            d = metadata["d"].to(device).float()
            extra = None
            if self.scale_extra_features:
                cols = []
                for key in self.scale_extra_features:
                    if key not in metadata:
                        raise KeyError(
                            f"scale_extra_features needs '{key}' in metadata; "
                            "not found in the loader's reference file."
                        )
                    cols.append(metadata[key].to(device).float().reshape(-1))
                extra = torch.stack(cols, dim=-1)  # (B, n_extra)
            scale = self.scale_fn(frame, x_det, y_det, lp, d, extra)
            if self.scale_fn.friedel_safe:
                scale = scale / lp  # LP as the known fixed factor, not learned
            return scale
        if isinstance(self.scale_fn, LinearScale):
            d = metadata["d"].to(device).float()
            s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
            absn = self._absorption(metadata, device)
            image = None
            if self.scale_fn.n_images > 0:
                if "imageset_id" not in metadata:
                    raise KeyError(
                        "scale_n_images>0 needs 'imageset_id' in the metadata."
                    )
                image = metadata["imageset_id"].to(device).long().reshape(-1)
            wavelength = None
            if self.scale_fn.lambda_modes > 0:
                if "wavelength" not in metadata:
                    raise KeyError(
                        "scale_lambda_modes>0 needs 'wavelength' in the metadata."
                    )
                wavelength = metadata["wavelength"].to(device).float().reshape(-1)
            return self.scale_fn(frame, s_sq, absn, image, wavelength) / lp
        if isinstance(self.scale_fn, SolvedScale):
            d = metadata["d"].to(device).float()
            s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
            absn = self._absorption(metadata, device)
            return self.scale_fn(frame, s_sq, absn) / lp
        if isinstance(self.scale_fn, CoarseScale):
            d = metadata["d"].to(device).float()
            s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
            return self.scale_fn(frame, s_sq) / lp
        return self.scale_fn(frame) / lp

    def _absorption(
        self, metadata: dict, device: torch.device
    ) -> Tensor | None:
        """Select the crystal-frame SH absorption columns for the solved scale.

        """
        if getattr(self, "_absorption_lmax", 0) <= 0:
            return None
        if "absorption_sh" not in metadata:
            raise KeyError(
                "scale_absorption_lmax > 0 needs 'absorption_sh' in the metadata; "
                "run scripts/extract_crystal_frame_sh.py and point the loader's "
                "reference: at the resulting metadata_sh file."
            )
        a = metadata["absorption_sh"].to(device).float()
        if a.ndim == 1:
            a = a.unsqueeze(-1)
        idx_t = self._absorption_idx
        if idx_t is None:
            width = a.shape[-1]
            extract_lmax = int(round((width + 1) ** 0.5)) - 1
            if (extract_lmax + 1) ** 2 - 1 != width:
                raise ValueError(
                    f"absorption_sh width {width} is not (lmax+1)^2-1 for any lmax"
                )
            if extract_lmax < self._absorption_lmax:
                raise ValueError(
                    f"absorption_sh has lmax={extract_lmax} < requested "
                    f"scale_absorption_lmax={self._absorption_lmax}; re-extract "
                    "with a higher --lmax."
                )
            idx = _sh_select_idx(
                extract_lmax, self._absorption_lmax, self._absorption_even_only
            )
            idx_t = torch.tensor(idx, dtype=torch.long)
            self._absorption_idx = idx_t
        return a.index_select(-1, idx_t.to(a.device))

    def _wilson_tau(self, d: Tensor) -> Tensor:
        """Wilson prior rate tau from resolution d (lp lives in the scale)."""
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        return self.loss._get_tau({"d": d}, s_sq, d.device)

    def _bg_prior(
        self, group_labels: Tensor | None, n: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Per-obs background Gamma prior (concentration, rate) from the loss."""
        conc = self.loss.bg_concentration.to(device)
        rate = self.loss.bg_rate.to(device)
        if conc.ndim == 1:  # per-resolution-bin priors
            g = (
                torch.zeros(n, dtype=torch.long, device=device)
                if group_labels is None
                else group_labels.to(device).long()
            )
            return conc[g], rate[g]
        return conc.expand(n), rate.expand(n)

    def get_merged_qi(self) -> Gamma:
        """Per-HKL Gamma posterior from the merge buffers (for MTZ output)."""
        return Gamma(
            self.alpha_buffer.clamp(min=1e-6),
            self.beta_buffer.clamp(min=1e-12),
        )

    def _wilson_kl_per_hkl(
        self, qi_h: Gamma, tau_h: Tensor, centric: Tensor | None = None
    ) -> Tensor:
        """KL(q(I_h) || Wilson prior), counted once per HKL.

        Acentric reflections follow Gamma(alpha_W, tau_h) (mean 1/tau_h = Sigma);
        centric reflections follow the chi^2_1 form Gamma(alpha_W/2,
        (alpha_W/2)*tau_h) -- half the shape, mean-preserving rate.
        """
        alpha = self.alpha_W * torch.ones_like(tau_h)
        if centric is not None:
            alpha = torch.where(centric, alpha * 0.5, alpha)
        p_i = Gamma(alpha, (alpha * tau_h).clamp(min=1e-12))
        return kl_divergence(qi_h, p_i)

    def _extra_loss_terms(
        self, outputs: dict, metadata: dict
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Extra ELBO terms added in `_step`"""
        return outputs["scale"].new_zeros(()), {}

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        return self._forward_impl(counts, shoebox, mask, metadata)

    @abstractmethod
    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]: ...

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        qi_h = outputs["qi_h"]

        # EM M-step accumulation for the solved (analytical) scale.
        if (
            step == "train"
            and isinstance(self.scale_fn, SolvedScale)
            and self.current_epoch >= self.scale_solve_warmup
        ):
            self._accumulate_scale(outputs, metadata)

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

        # Per-HKL Wilson intensity KL, put on the per-observation scale by dividing the sum by the obs count.
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

        extra_loss, extra_logs = self._extra_loss_terms(outputs, metadata)
        total_loss = total_loss + extra_loss
        for name, value in extra_logs.items():
            self.log(f"{step} {name}", value, on_epoch=True)

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

        if step == "train":
            self._collect_scatters(outputs, metadata, mask, counts)

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

    def training_step(self, batch, _batch_idx):
        return self._step(batch, step="train")

    def validation_step(self, batch, _batch_idx):
        return self._step(batch, step="val")

    def on_train_epoch_end(self) -> None:
        # EM: solve the scale from this epoch's accumulated normal equations.
        # (single-GPU; multi-GPU would need an all-reduce of the accumulators.)
        if (
            isinstance(self.scale_fn, SolvedScale)
            and self.current_epoch >= self.scale_solve_warmup
        ):
            self.scale_fn.solve()

    @torch.no_grad()
    def _accumulate_scale(self, outputs: dict, metadata: dict) -> None:
        """Add this batch's targets to the solved-scale normal equations.

        `J_o = sig_counts / exposure` is the per-shoebox intensity, `I_h` the
        current merged intensity. The total scale is `exp(Phi @ theta) / lp`, and
        at the EM fixed point `scale * I_h ~ J_o`, so the target for `Phi @ theta`
        is `log(J_o) - log(I_h) + log(lp)` -- the `+log(lp)` cancels the LP baked
        into `J` (LP is the known `/lp` factor, not fit), leaving the smooth
        `K(phi)*decay(s^2)` residual. Weighted by signal counts.
        """
        sig = outputs["signal_counts"]
        device = sig.device
        scale = outputs["scale"].clamp(min=1e-12)
        e_o = (outputs["exposure"] / scale).clamp(min=1e-6)  # = sum_p prf
        j = (sig / e_o).clamp(min=1e-12)
        i_h = outputs["alpha_h"] / outputs["beta_h"].clamp(min=1e-6)
        i_o = i_h[outputs["inverse"]].clamp(min=1e-12)
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        y = j.log() - i_o.log() + lp.log()
        d = metadata["d"].to(device).float()
        s_sq = 1.0 / (4.0 * d.clamp(min=1e-6).pow(2))
        frame = metadata["xyzcal.px.2"].to(device).float()
        absn = self._absorption(metadata, device)
        self.scale_fn.accumulate(frame, s_sq, y, sig.clamp(min=0.0), absn)

    def predict_step(self, batch, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]
        self._warn_unknown_predict_keys(forward_out)
        return {k: v for k, v in forward_out.items() if k in self.predict_keys}

    def _warn_unknown_predict_keys(self, forward_out: dict) -> None:
        if getattr(self, "_predict_keys_checked", False):
            return
        self._predict_keys_checked = True
        missing = [k for k in self.predict_keys if k not in forward_out]
        if missing:
            _logger.warning(
                "predict_keys not produced by this integrator (ignored): %s. "
                "Available outputs: %s",
                sorted(missing),
                sorted(forward_out),
            )

    def _profile_basis_penalty(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Spatial-smoothness penalty on the learned qp decoder weight W."""
        zero = torch.zeros((), device=self.device)
        qp = self.surrogates["qp"] if "qp" in self.surrogates else None
        decoder = getattr(qp, "decoder", None)
        W = getattr(decoder, "weight", None)
        smooth_w = float(getattr(qp, "smoothness_weight", 0.0))
        if W is None or W.dim() != 2:
            return zero, {}

        shape = self.shoebox_shape
        D, H, W_spatial = shape if len(shape) == 3 else (1, *shape)
        K, d = W.shape
        if K != D * H * W_spatial:
            return zero, {}

        total = zero
        components: dict[str, Tensor] = {}
        if smooth_w > 0:
            vol = W.T.reshape(d, D, H, W_spatial)
            gx = vol[..., 1:] - vol[..., :-1]
            gy = vol[..., 1:, :] - vol[..., :-1, :]
            sq_sum = (gx.pow(2)).sum() + (gy.pow(2)).sum()
            n_terms = gx.numel() + gy.numel()
            if D > 1:
                gz = vol[..., 1:, :, :] - vol[..., :-1, :, :]
                sq_sum = sq_sum + gz.pow(2).sum()
                n_terms = n_terms + gz.numel()
            smooth = sq_sum / max(n_terms, 1)
            components["profile_smoothness"] = smooth.detach()
            total = total + smooth_w * smooth
        return total, components

    def on_after_backward(self) -> None:
        for sname in ("qbg",):
            if sname not in self.surrogates:
                continue
            for pname, p in self.surrogates[sname].named_parameters():
                if p.grad is not None:
                    self.log(
                        f"grad/{sname}.{pname}",
                        p.grad.norm(),
                        on_step=False,
                        on_epoch=True,
                    )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam; with `scaling_lr` set, the scale field gets its own group."""
        decoder_split = self.decoder_weight_decay is not None

        if self.scaling_lr is None and not decoder_split:
            return torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        scale_params: list[nn.Parameter] = []
        decoder_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self.scaling_lr is not None and name.startswith("scale_fn."):
                scale_params.append(param)
            elif decoder_split and name.endswith(
                "surrogates.qp.decoder.weight"
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
        if scale_params:
            groups.append(
                {
                    "params": scale_params,
                    "weight_decay": self.weight_decay,
                    "lr": self.scaling_lr,
                }
            )
        return torch.optim.Adam(groups, lr=self.lr)

    def _cosine_warmup_lambda(self, max_epochs: int):
        warmup = self.warmup_epochs
        lr_min_ratio = self.lr_min / max(self.lr, 1e-12)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            tail = max(max_epochs - warmup, 1)
            progress = min((epoch - warmup) / tail, 1.0)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cos_term

        return lr_lambda

    def _step_linear_warmup_lambda(self):
        warmup = max(int(self.warmup_steps), 0)

        def lr_lambda(step: int) -> float:
            if warmup == 0 or step >= warmup:
                return 1.0
            return float(step + 1) / float(warmup)

        return lr_lambda

    def configure_optimizers(self) -> Any:
        optimizer = self._build_optimizer()
        if self.lr_schedule is None:
            return optimizer

        if self.lr_schedule == "cosine_warmup":
            max_epochs = self.trainer.max_epochs
            if max_epochs is None or max_epochs <= 0:
                raise RuntimeError(
                    "cosine_warmup requires trainer.max_epochs to be set; "
                    f"got {max_epochs}."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._cosine_warmup_lambda(max_epochs)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        if self.lr_schedule == "step_linear_warmup":
            if self.warmup_steps <= 0:
                raise ValueError(
                    "step_linear_warmup requires warmup_steps > 0; "
                    f"got {self.warmup_steps}."
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self._step_linear_warmup_lambda()
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        raise ValueError(
            f"Unknown lr_schedule {self.lr_schedule!r}. "
            "Supported: 'cosine_warmup', 'step_linear_warmup', or null."
        )

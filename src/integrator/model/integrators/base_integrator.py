import logging
import math
from abc import abstractmethod
from typing import Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from integrator.configs import IntegratorCfg, OptimizerConfig
from integrator.model.integrators.integrator_utils import ScatterLoggerMixin

_logger = logging.getLogger(__name__)
# the merging branch's code in this module logs through the bare name
logger = _logger

_DECONVOLVED_KEYS = (
    "i_deconv",
    "var_deconv",
    "deconv_floored",
)

_RAO_BLACKWELL_KEYS = (
    "i_rb",
    "var_rb",
    "var_rb_cond",
    "var_rb_profile",
    "rb_pedestal",
    "rb_gate",
)

from integrator.model.integrators.integrator_utils import (
    _samples_first,
    rao_blackwellized_intensity,
)

logger = logging.getLogger(__name__)
logger = _logger  # the merging branch's modules log through this name

DEFAULT_PREDICT_KEYS = [
    "refl_ids",
    "is_test",
    "qi_mean",
    "qi_var",
    "qi_params",
    "qbg_mean",
    "qbg_var",
    "qbg_scale",
    "qbg_params",
    "intensity.prf.value",
    "intensity.prf.variance",
    "intensity.sum.value",
    "intensity.sum.variance",
    "background.mean",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "d",
    "H",
    "K",
    "L",
]


def _log_loss(
    self,
    kl,
    nll,
    total_loss,
    step: Literal["train", "val"],
    kl_components: dict[str, Tensor] | None = None,
):
    # progress bar: "train elbo" and "val elbo" (per epoch)
    if step == "train":
        self.log(
            "train elbo",
            total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
    else:
        self.log(
            "val elbo",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
    self.log(f"{step} kl", kl, on_step=False, on_epoch=True)
    self.log(f"{step} nll", nll, on_step=False, on_epoch=True)
    self.log("epoch", float(self.current_epoch), on_step=False, on_epoch=True)
    if kl_components is not None:
        for name, value in kl_components.items():
            self.log(
                f"{step} kl_{name}",
                value,
                on_step=False,
                on_epoch=True,
            )


class BaseIntegrator(ScatterLoggerMixin, pl.LightningModule):
    """Variational integrator base class."""

    REQUIRED_ENCODERS: dict[str, tuple[str, type]] = {}

    DEFAULT_SURROGATES: dict[str, dict] = {
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
        "qi": {
            "name": "gamma",
            "args": {
                "reparameterization": "mean_fano",
                "eps": 1.0e-6,
                "k_min": 0.01,
            },
        },
    }
    ARGS: type

    def __init__(
        self,
        cfg: IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
        optimizer: OptimizerConfig | None = None,
    ):
        """Build the integrator from its config and submodules.

        Args:
            cfg: Architecture/inference config (shoebox shape, encoder width, MC samples, predict keys).
            loss: ELBO module returning the NLL, KL, and per-component KL terms.
            encoders: Named encoder modules mapping shoeboxes to hidden representations.
            surrogates: Named variational surrogates (`qi`, `qbg`, `qp`) mapping representations to posterior distributions.
            optimizer: Optimizer/schedule settings; defaults are used when omitted.
        """
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
        self.loss_lr = getattr(opt, "loss_lr", None)
        self.mc_samples = cfg.mc_samples
        self.coset_mode = cfg.coset_mode
        if cfg.data_dim == "2d":
            self.shoebox_shape = (cfg.h, cfg.w)
        else:
            self.shoebox_shape = (cfg.d, cfg.h, cfg.w)

        self.predict_keys = (
            DEFAULT_PREDICT_KEYS
            if cfg.predict_keys == "default"
            else cfg.predict_keys
        )
        self.rb_gate_half_signal = cfg.rb_gate_half_signal
        self.rb_noise_a = cfg.rb_noise_a
        self.rb_noise_b = cfg.rb_noise_b
        self.rb_pedestal = cfg.rb_pedestal
        self.rb_background_term = cfg.rb_background_term
        self.rb_profile = getattr(cfg, "rb_profile", "samples")
        self.export_refine_steps = int(getattr(cfg, "export_refine_steps", 0))
        self.export_refine_lr = float(getattr(cfg, "export_refine_lr", 0.05))
        self.profile_coord_channels = getattr(
            cfg, "profile_coord_channels", False
        )
        self.beam_center = getattr(cfg, "beam_center", None)

        self.encoders = nn.ModuleDict(encoders)
        self.surrogates = nn.ModuleDict(surrogates)
        self.loss = loss

        self.automatic_optimization = True

        # End-of-epoch model-vs-DIALS scatters (intensity + background), available
        # to every integrator and gated by the log_*_scatter cfg flags.
        self._init_scatter_logger(cfg)

        # Optional: transplant (and freeze) the background modules from another
        # run's checkpoint. The background is the field the merging models keep
        # collapsing (it loses the race to the intensity); freezing a proven bg
        # from a good integration run removes it from the optimization so the
        # merge can only fit the peak.
        self._frozen_eval_modules: list[nn.Module] = []
        bg_ckpt = getattr(cfg, "bg_init_from_checkpoint", None)
        if bg_ckpt:
            self._init_bg_from_checkpoint(
                bg_ckpt, freeze=getattr(cfg, "bg_freeze", True)
            )
        prof_ckpt = getattr(cfg, "profile_init_from_checkpoint", None)
        if prof_ckpt:
            self._init_profile_from_checkpoint(
                prof_ckpt, freeze=getattr(cfg, "profile_freeze", True)
            )

    def _init_bg_from_checkpoint(
        self, ckpt_path: str, freeze: bool = True
    ) -> None:
        """Load (and optionally freeze) the background modules from a checkpoint.

        Copies the `k_bg`/`r_bg` background encoders and the `qbg` surrogate from
        another run's checkpoint (their architectures must match this model's).
        When `freeze`, their parameters are frozen and the modules are kept in
        eval mode (no dropout) even while the rest of the model trains -- see the
        `train` override. Raises if a targeted module is absent from the
        checkpoint (a silent partial load would be worse than failing).
        """
        from pathlib import Path

        if not Path(ckpt_path).exists():
            # At prediction the bg comes from the loaded checkpoint anyway, so a
            # missing source is non-fatal -- warn and skip.
            logger.warning(
                "bg_init_from_checkpoint: %s not found; skipping bg transplant.",
                ckpt_path,
            )
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        targets = [
            ("encoders.k_bg", self.encoders["k_bg"] if "k_bg" in self.encoders else None),
            ("encoders.r_bg", self.encoders["r_bg"] if "r_bg" in self.encoders else None),
            ("surrogates.qbg", self.surrogates["qbg"] if "qbg" in self.surrogates else None),
        ]
        loaded = []
        for name, module in targets:
            if module is None:
                continue
            prefix = name + "."
            sub = {
                k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
            }
            if not sub:
                raise KeyError(
                    f"bg_init_from_checkpoint: no '{name}.*' params in "
                    f"{ckpt_path} -- architecture/name mismatch."
                )
            module.load_state_dict(sub, strict=True)
            loaded.append(name)
            if freeze:
                for p in module.parameters():
                    p.requires_grad_(False)
                module.eval()
                self._frozen_eval_modules.append(module)
        logger.info(
            "bg_init_from_checkpoint: loaded %s from %s (freeze=%s)",
            loaded,
            ckpt_path,
            freeze,
        )

    def _init_profile_from_checkpoint(
        self, ckpt_path: str, freeze: bool = True
    ) -> None:
        """Load (and optionally freeze) the profile modules from a checkpoint.

        Copies the `profile` encoder and the `qp` surrogate from another run's
        checkpoint, so a profile learned where it can be learned -- a per-
        observation integration run -- can be carried into a merge that collapses
        its own. Shapes must match, which ties the merge's profile `encoder_out`
        to the source's. When `freeze`, the parameters are frozen and the modules
        stay in eval mode while the rest trains, as for the background.
        """
        from pathlib import Path

        if not Path(ckpt_path).exists():
            # At prediction the profile comes from the loaded checkpoint anyway.
            logger.warning(
                "profile_init_from_checkpoint: %s not found; skipping transplant.",
                ckpt_path,
            )
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        targets = [
            (
                "encoders.profile",
                self.encoders["profile"] if "profile" in self.encoders else None,
            ),
            (
                "surrogates.qp",
                self.surrogates["qp"] if "qp" in self.surrogates else None,
            ),
        ]
        loaded = []
        for name, module in targets:
            if module is None:
                continue
            prefix = name + "."
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if not sub:
                raise KeyError(
                    f"profile_init_from_checkpoint: no '{name}.*' params in "
                    f"{ckpt_path} -- architecture/name mismatch."
                )
            module.load_state_dict(sub, strict=True)
            loaded.append(name)
            if freeze:
                for p in module.parameters():
                    p.requires_grad_(False)
                module.eval()
                self._frozen_eval_modules.append(module)
        logger.info(
            "profile_init_from_checkpoint: loaded %s from %s (freeze=%s)",
            loaded,
            ckpt_path,
            freeze,
        )

    def train(self, mode: bool = True):
        """Keep frozen transplanted modules in eval mode (no dropout) when the
        rest of the model is switched to train."""
        super().train(mode)
        for module in getattr(self, "_frozen_eval_modules", []):
            module.eval()
        return self

    def _profile_encoder_input(self, shoebox: Tensor, metadata: dict) -> Tensor:
        """The profile encoder's input, with coordinate channels when enabled.

        Off by default, so the tensor passes through untouched and the encoder
        keeps its single channel.
        """
        if not getattr(self, "profile_coord_channels", False):
            return shoebox
        from integrator.model.integrators.integrator_utils import (
            coordinate_channels,
        )

        coords = coordinate_channels(
            metadata,
            tuple(shoebox.shape[2:]),
            self.beam_center,
            device=shoebox.device,
            dtype=shoebox.dtype,
        )
        return torch.cat((shoebox, coords), dim=1)

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        """Run the integrator forward pass, returning the surrogates and assembled outputs."""
        out = self._forward_impl(counts, shoebox, mask, metadata)
        return out

    @abstractmethod
    def _forward_impl(
        self,
        counts: torch.Tensor,
        shoebox: torch.Tensor,
        mask: torch.Tensor,
        metadata: dict,
    ) -> dict[str, Any]: ...

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

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
            qbg_plane=outputs.get("qbg_plane"),
            coset_mode=self.coset_mode,
        )

        total_loss = loss_dict["loss"]

        _log_loss(
            self,
            kl=loss_dict["kl_mean"],
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                k.removesuffix("_mean"): v
                for k, v in loss_dict.items()
                if k in ("kl_prf_mean", "kl_i_mean", "kl_bg_mean")
            },
        )

        if hasattr(self.loss, "diagnostics"):
            for name, value in self.loss.diagnostics().items():
                self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        if "coset_aux_mean" in loss_dict:
            self.log(
                f"{step} coset_aux",
                loss_dict["coset_aux_mean"],
                on_step=False,
                on_epoch=True,
            )

        # Track predicted intensity on coset (background-only) reflections.
        # I|coset should sit near the background floor (especially in supervised
        # mode); the gap to I|lattice is a direct false-positive / background-
        # calibration readout. Logged in either coset_mode for comparison.
        if "is_coset" in metadata:
            with torch.no_grad():
                qi_mean = outputs["qi"].mean.detach()
                coset = metadata["is_coset"].bool().to(qi_mean.device)
                if coset.any():
                    self.log(
                        f"{step} I_coset",
                        qi_mean[coset].mean(),
                        on_step=False,
                        on_epoch=True,
                    )
                if (~coset).any():
                    self.log(
                        f"{step} I_lattice",
                        qi_mean[~coset].mean(),
                        on_step=False,
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
                "kl": loss_dict["kl_mean"].detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": loss_dict["kl_i_mean"].detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

    def training_step(self, batch, _batch_idx):
        """Run one training step and log the ELBO and its components."""
        return self._step(batch, step="train")

    def validation_step(self, batch, _batch_idx):
        """Run one validation step and log the ELBO and its components."""
        return self._step(batch, step="val")

    def predict_step(self, batch, _batch_idx):
        """Run inference and return the `predict_keys` subset of the forward outputs."""
        if self.export_refine_steps > 0:
            return self._refined_predict_step(batch)
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        return self._predict_outputs(outputs, metadata)

    def _predict_outputs(self, outputs, metadata):
        forward_out = outputs["forward_out"]
        self._add_rao_blackwell(forward_out)
        if "qi" not in forward_out:
            forward_out["qi"] = outputs.get("qi")
        self._add_deconvolved(forward_out, metadata)
        self._warn_unknown_predict_keys(forward_out)
        return {k: v for k, v in forward_out.items() if k in self.predict_keys}


    def _batch_loss(self, outputs, metadata) -> torch.Tensor:
        """The training loss of one batch, as `_step` computes it, without logging."""
        forward_out = outputs["forward_out"]
        return self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=metadata["group_label"].long(),
            metadata=metadata,
            qbg_plane=outputs.get("qbg_plane"),
            coset_mode=self.coset_mode,
        )["loss"]

    def _refined_predict_step(self, batch):
        """`export_refine_steps` Adam steps on the profile's per-reflection variational parameters, then predict.

        Prediction runs under `torch.inference_mode`, whose tensors cannot enter
        autograd, so the batch is cloned with inference mode off. Gradients are
        taken with `torch.autograd.grad` with respect to the free tensors only,
        so no trained parameter accumulates a gradient or changes. The batch
        loss is a mean over reflections and each reflection's terms depend only
        on its own free parameters, so the loss is multiplied by the batch size
        to give each reflection the gradient of its own objective.
        """
        qp = self.surrogates["qp"]
        if not hasattr(qp, "refine_init"):
            raise RuntimeError(
                f"export_refine_steps needs a profile surrogate with per-reflection "
                f"variational parameters; {type(qp).__name__} has none"
            )

        def clone(v):
            return v.clone() if isinstance(v, torch.Tensor) else v

        with torch.inference_mode(False), torch.enable_grad():
            counts, shoebox, mask, metadata = batch
            counts, shoebox, mask = clone(counts), clone(shoebox), clone(mask)
            metadata = {k: clone(v) for k, v in metadata.items()}
            free = qp.refine_init(lambda: self(counts, shoebox, mask, metadata))
            free = [t.detach().clone().requires_grad_(True) for t in free]
            opt = torch.optim.Adam(free, lr=self.export_refine_lr)
            n = counts.shape[0]
            try:
                qp.refine_override = free
                for _ in range(self.export_refine_steps):
                    loss = self._batch_loss(self(counts, shoebox, mask, metadata), metadata) * n
                    grads = torch.autograd.grad(loss, free)
                    opt.zero_grad()
                    for t, g in zip(free, grads):
                        t.grad = g
                    opt.step()
                with torch.no_grad():
                    outputs = self(counts, shoebox, mask, metadata)
                    return self._predict_outputs(outputs, metadata)
            finally:
                qp.refine_override = None

    def _add_deconvolved(self, forward_out: dict, metadata: dict) -> None:
        """Add the intensity with our own Wilson prior divided back out.

        What we export is a posterior: `q(I) = Gamma(alpha, beta)` already shrunk
        toward the Wilson prior by the integrator's loss. careless then applies a
        Wilson prior of its own, so weak reflections are shrunk twice. Dividing a
        Gamma posterior by its Gamma prior is exact and stays in the family:

            Gamma(alpha, beta) / Gamma(c, lam)  ->  Gamma(alpha - c + 1, beta - lam)

        so this reports what the data alone said, which is what a likelihood term
        is supposed to carry.

        `beta <= lam` means the posterior came out narrower than the prior, which
        cannot happen if it really is likelihood times prior; those reflections
        keep their posterior values and are flagged in `deconv_floored` rather
        than being given a negative rate.
        """
        if not (set(_DECONVOLVED_KEYS) & set(self.predict_keys)):
            return
        qi = forward_out.get("qi")
        if qi is None or not hasattr(qi, "concentration"):
            raise RuntimeError(
                "deconvolved export needs a Gamma q(I) exposing concentration "
                "and rate; this integrator's qi does not"
            )
        loss = getattr(self, "loss", None)
        if loss is None or not hasattr(loss, "intensity_prior_params"):
            raise RuntimeError(
                "deconvolved export needs a Wilson loss exposing "
                "intensity_prior_params; got "
                f"{type(loss).__name__}"
            )

        alpha = qi.concentration.reshape(-1)
        beta = qi.rate.reshape(-1)
        c, lam = loss.intensity_prior_params(metadata, alpha.device)
        c = c.reshape(-1)
        lam = lam.reshape(-1)

        rate = beta - lam
        floored = rate <= 0
        conc_like = torch.clamp(alpha - c + 1.0, min=1e-3)
        rate_like = torch.where(floored, beta, rate)
        conc_like = torch.where(floored, alpha, conc_like)

        forward_out["i_deconv"] = conc_like / rate_like
        forward_out["var_deconv"] = conc_like / rate_like.pow(2)
        forward_out["deconv_floored"] = floored.to(alpha.dtype)

    def _add_rao_blackwell(self, forward_out: dict) -> None:
        """Add the profile-uncertainty-propagated intensity, if it was asked for.

        An export-time quantity, so it is not computed during training: the plug-in
        `qi_mean`/`qi_var` are what the ELBO uses, while this is what a downstream
        merge needs in order to weight correctly.
        """
        if not (set(_RAO_BLACKWELL_KEYS) & set(self.predict_keys)):
            return
        batch = forward_out["qp_mean"].shape[0]
        zp = _samples_first(forward_out["zp"], batch)
        zbg = _samples_first(forward_out["zbg"], batch)
        kw = dict(
            gate_half_signal=self.rb_gate_half_signal,
            noise_a=getattr(self, "rb_noise_a", 1.0),
            noise_b=getattr(self, "rb_noise_b", 0.0),
            pedestal=getattr(self, "rb_pedestal", "none"),
            background_term=getattr(self, "rb_background_term", "sample_bg"),
        )
        rb = rao_blackwellized_intensity(forward_out["counts"], forward_out["mask"], zp, zbg, **kw)
        if getattr(self, "rb_profile", "samples") == "mean":
            pm = forward_out["qp_mean"].reshape(batch, -1).unsqueeze(0)
            pm = pm.expand(zbg.shape[0], -1, -1)
            rm = rao_blackwellized_intensity(forward_out["counts"], forward_out["mask"], pm, zbg, **kw)
            rb["i_rb"] = rm["i_rb"]
            rb["var_rb_cond"] = rm["var_rb_cond"]
            rb["var_rb"] = rm["var_rb_cond"] + rb["rb_gate"] * rb["var_rb_profile"]
        forward_out.update(rb)

    def _warn_unknown_predict_keys(self, forward_out: dict) -> None:
        """Warn once if `predict_keys` requests columns the model never produces."""
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

    def on_after_backward(self) -> None:
        """Log gradient norms for intensity/background surrogate heads."""
        for sname in ("qi", "qbg"):
            if sname not in self.surrogates:
                continue
            surr = self.surrogates[sname]
            for pname, p in surr.named_parameters():
                if p.grad is not None:
                    self.log(
                        f"grad/{sname}.{pname}",
                        p.grad.norm(),
                        on_step=False,
                        on_epoch=True,
                    )

    def on_validation_epoch_end(self) -> None:
        """Log val - train generalization gaps for each ELBO component."""
        metrics = self.trainer.callback_metrics
        for key in list(metrics.keys()):
            if not key.startswith("val "):
                continue
            suffix = key[len("val ") :]
            train_key = f"train {suffix}"
            if train_key not in metrics:
                continue
            gap = metrics[key] - metrics[train_key]
            self.log(
                f"gap {suffix}",
                gap,
                on_step=False,
                on_epoch=True,
            )

    def on_train_epoch_end(self) -> None:
        """Re-estimate the Wilson prior from the epoch's raw counts.

        A module hook rather than a Callback because it mutates model state,
        it inherits into every integrator subclass, and Lightning runs it
        before the monitoring callbacks -- so a checkpoint saved this epoch
        holds the refitted prior rather than the one it replaced.
        """
        loss = getattr(self, "loss", None)
        # Self-calibrating per-bin intensity KL weight (independent of the G/B
        # refit below): drive each shell's posterior sd/mean to the Wilson
        # moment. Runs its own epoch-end M-step off sums gathered in forward.
        if getattr(loss, "intensity_kl_calibrate", False):
            for key, value in loss.calibrate_intensity_kl(self.current_epoch).items():
                if value == value:  # skip NaN
                    self.log(key, float(value), on_step=False, on_epoch=True)
        if not getattr(loss, "refit_enabled", False):
            return
        stats = loss.refit_prior(self.current_epoch)
        for key, value in stats.items():
            if value == value:  # skip NaN
                self.log(key, float(value), on_step=False, on_epoch=True)

    # the Wilson prior's scalars: physical quantities, not network weights
    _PRIOR_SCALARS = ("raw_G", "raw_B", "raw_scale")

    def _is_prior_scalar(self, name: str) -> bool:
        return name.startswith("loss.") and name.split(".")[-1] in self._PRIOR_SCALARS

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Adam, with the Wilson scalars held out of weight decay.

        G and B are physical quantities in counts and A^2, not network weights,
        and decaying them toward zero is not regularization -- it is a constant
        force on the prior. Under plain Adam the L2 term enters the gradient
        before the second-moment normalization, and for G it swamps the data
        term by ~4 orders of magnitude: dKL/draw_G is (1 - r)/G, order 1e-5,
        against wd * raw_G of order 0.4. The logged trace shows the result --
        G declines smoothly at a rate set by the learning-rate schedule and
        then stops dead when cosine decay reaches lr_min, never having been
        moved by the data at all.
        """
        decoder_params: list[nn.Parameter] = []
        prior_params: list[nn.Parameter] = []
        other_params: list[nn.Parameter] = []
        loss_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if self.loss_lr is not None and name.startswith("loss."):
                loss_params.append(param)
            elif self._is_prior_scalar(name):
                prior_params.append(param)
            # any surrogate decoder weight (e.g. surrogates.qp.decoder.weight)
            elif name.startswith("surrogates.") and name.endswith(
                ".decoder.weight"
            ):
                decoder_params.append(param)
            else:
                other_params.append(param)

        if self.decoder_weight_decay is not None and not decoder_params:
            raise RuntimeError(
                "decoder_weight_decay is set but no surrogate '.decoder.weight' parameter was found "
            )

        groups: list[dict] = [
            {"params": other_params, "weight_decay": self.weight_decay}
        ]
        if prior_params:
            groups.append({"params": prior_params, "weight_decay": 0.0})
        if decoder_params:
            decay = (
                self.weight_decay
                if self.decoder_weight_decay is None
                else self.decoder_weight_decay
            )
            groups.append({"params": decoder_params, "weight_decay": decay})
        if loss_params:
            groups.append({"params": loss_params, "weight_decay": 0.0, "lr": self.loss_lr})
        return torch.optim.Adam(groups, lr=self.lr)

    def _cosine_warmup_lambda(self, max_epochs: int):
        """Linear warmup for `warmup_epochs`, then cosine decay to lr_min."""
        warmup = self.warmup_epochs
        lr_min_ratio = self.lr_min / max(self.lr, 1e-12)

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            # Cosine decay from epoch == warmup (value 1) down to lr_min_ratio
            # at epoch == max_epochs.
            tail = max(max_epochs - warmup, 1)
            progress = min((epoch - warmup) / tail, 1.0)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cos_term

        return lr_lambda

    def _step_linear_warmup_lambda(self):
        """Linear ramp 0 -> 1 over the first `warmup_steps` optimizer steps."""
        warmup = max(int(self.warmup_steps), 0)

        def lr_lambda(step: int) -> float:
            if warmup == 0:
                return 1.0
            if step >= warmup:
                return 1.0
            return float(step + 1) / float(warmup)

        return lr_lambda

    def configure_optimizers(self) -> Any:
        """Build the Adam optimizer and the optional warmup/cosine or step-warmup scheduler."""
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
                optimizer,
                lr_lambda=self._cosine_warmup_lambda(max_epochs),
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
                optimizer,
                lr_lambda=self._step_linear_warmup_lambda(),
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

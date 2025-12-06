from dataclasses import dataclass
from typing import Any, Literal

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from integrator.model.distributions import (
    DirichletDistribution,
    FoldedNormalDistribution,
    GammaDistribution,
    LogNormalDistribution,
)
from integrator.model.encoders import (
    IntensityEncoder,
    MLPMetadataEncoder,
    ShoeboxEncoder,
)


def calculate_intensities(counts, qbg, qp, mask, cfg):
    with torch.no_grad():
        counts = counts * mask  # [B,P]
        zbg = qbg.rsample([cfg.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.mean.unsqueeze(1)

        vi = zbg + 1e-6
        intensity = torch.tensor([0.0])

        # kabsch sum
        for _ in range(cfg.max_iterations):
            num = (counts.unsqueeze(1) - zbg) * zp * mask.unsqueeze(1) / vi
            denom = zp.pow(2) / vi
            intensity = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
            vi = (intensity.unsqueeze(-1) * zp) + zbg
            vi = vi.mean(-1, keepdim=True)
        kabsch_sum_mean = intensity.mean(-1)
        kabsch_sum_var = intensity.var(-1)

        # profile masking
        zp = zp * mask.unsqueeze(1)  # profiles
        thresholds = torch.quantile(
            zp,
            0.99,
            dim=-1,
            keepdim=True,
        )  # threshold values
        profile_mask = zp > thresholds

        masked_counts = counts.unsqueeze(1) * profile_mask

        profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)

        profile_masking_mean = profile_masking_I.mean(-1)

        profile_masking_var = profile_masking_I.var(-1)

        intensities = {
            "profile_masking_mean": profile_masking_mean,
            "profile_masking_var": profile_masking_var,
            "kabsch_sum_mean": kabsch_sum_mean,
            "kabsch_sum_var": kabsch_sum_var,
        }

        return intensities


def _default_predict_keys() -> list["str"]:
    return [
        "qi_mean",
        "qi_var",
        "refl_ids",
        "dials_I_sum_value",
        "dials_I_sum_var",
        "dials_I_prf_value",
        "dials_I_prf_var",
        "dials_bg_mean",
        "qbg_mean",
        "qbg_scale",
        "x_c",
        "y_c",
        "z_c",
    ]


# config classes
@dataclass
class IntegratorHyperParameters:
    data_dim: Literal["2d", "3d"]
    d: int
    h: int
    w: int
    lr: float = 0.001
    encoder_out: int = 64
    weight_decay: float = 0.0
    mc_samples: int = 4
    renyi_scale: float = 0.0
    predict_keys: Literal["default"] | list[str] = "default"


@dataclass
class IntegratorBaseOutputs:
    rates: Tensor
    counts: Tensor
    mask: Tensor
    qbg: Any
    qp: Any
    qi: Any
    zp: Tensor
    concentration: Tensor | None = None
    reference: Tensor | None = None


def extract_reference_fields(
    ref: Tensor,
    data_dim: str,
) -> dict[str, Any]:
    if data_dim == "3d":
        return {
            "dials_I_sum_value": ref[:, 6],
            "dials_I_sum_var": ref[:, 7],
            "dials_I_prf_value": ref[:, 8],
            "dials_I_prf_var": ref[:, 9],
            "refl_ids": ref[:, -1].int().tolist(),
            "x_c": ref[:, 0],
            "y_c": ref[:, 1],
            "z_c": ref[:, 2],
            "x_c_mm": ref[:, 3],
            "y_c_mm": ref[:, 4],
            "z_c_mm": ref[:, 5],
            "dials_bg_mean": ref[:, 10],
            "dials_bg_sum_value": ref[:, 11],
            "dials_bg_sum_var": ref[:, 12],
            "d": ref[:, 13],
        }
    elif data_dim == "2d":
        return {
            "dials_I_sum_value": ref[:, 3],
            "dials_I_sum_var": ref[:, 4],
            "dials_I_prf_value": ref[:, 3],
            "dials_I_prf_var": ref[:, 4],
            "refl_ids": ref[:, -1].tolist(),
            "x_c": ref[:, 9],
            "y_c": ref[:, 10],
            "z_c": ref[:, 11],
            "dials_bg_mean": ref[:, 0],
            "dials_bg_sum_value": ref[:, 0],
            "dials_bg_sum_var": ref[:, 1],
            "wavelength": ref[:, 8],
            "batch": ref[:, 2],
            "h": ref[:, 5],
            "k": ref[:, 6],
            "l": ref[:, 7],
        }
    else:
        raise ValueError(f"Unsupported data_dim: {data_dim}")


def _assemble_outputs(
    out: IntegratorBaseOutputs,
    data_dim: Literal["2d", "3d"],
) -> dict[str, Any]:
    base = {
        "rates": out.rates,
        "counts": out.counts,
        "mask": out.mask,
        "zp": out.zp,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp_mean": out.qp.mean,
        "qi_mean": out.qi.mean,
        "qi_var": out.qi.variance,
        "profile": out.qp.mean,
        "concentration": out.concentration,
    }

    if out.reference is None:
        return base

    ref_fields = extract_reference_fields(out.reference, data_dim)

    base.update(ref_fields)
    return base


def _encode_shoebox(encoder1, encoder2, shoebox, shoebox_shape):
    if shoebox.dim() == 2:
        x_profile = encoder1(
            shoebox.reshape(shoebox.shape[0], 1, *(shoebox_shape))
        )
        x_intensity = encoder2(
            shoebox.reshape(shoebox.shape[0], 1, *(shoebox_shape))
        )

        return x_profile, x_intensity

    elif shoebox.dim() == 3:
        x_profile = encoder1(
            shoebox.reshape(shoebox.size(0), shoebox.size(1), *(shoebox_shape))
        )
        x_intensity = encoder2(
            shoebox[:, 0, :].reshape(shoebox.size(0), 1, *(shoebox_shape))
        )
        return x_profile, x_intensity
    else:
        raise ValueError(
            "Incorrect shoebox dimension. The shoebox should be 2 or 3 dimensional"
        )


@dataclass
class EncoderModules:
    encoder1: ShoeboxEncoder | IntensityEncoder
    encoder2: ShoeboxEncoder | IntensityEncoder
    encoder3: MLPMetadataEncoder | None = None


@dataclass
class SurrogateModules:
    qbg: LogNormalDistribution | GammaDistribution | FoldedNormalDistribution
    qi: LogNormalDistribution | GammaDistribution | FoldedNormalDistribution
    qp: DirichletDistribution


# %%
class Integrator(LightningModule):
    def __init__(
        self,
        cfg: IntegratorHyperParameters,
        loss: nn.Module,
        surrogates: SurrogateModules,
        encoders: EncoderModules,
    ):
        super().__init__()
        self.cfg = cfg

        # hyperparams
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.mc_samples = cfg.mc_samples
        self.renyi_scale = cfg.renyi_scale

        # posterior modules
        self.qbg = surrogates.qbg
        self.qp = surrogates.qp
        self.qi = surrogates.qi

        # encoder modules
        self.encoder1 = encoders.encoder1
        self.encoder2 = encoders.encoder2
        self.encoder3 = encoders.encoder3

        # loss module
        self.loss = loss

        # predict keys
        self.predict_keys = (
            _default_predict_keys()
            if cfg.predict_keys == "default"
            else cfg.predict_keys
        )

        if self.encoder3 is not None:
            self.linear = nn.Linear(cfg.encoder_out * 2, cfg.encoder_out)

        self.automatic_optimization = True

        if cfg.data_dim == "3d":
            self.shoebox_shape = (cfg.d, cfg.h, cfg.w)
        elif cfg.data_dim == "2d":
            self.shoebox_shape = (cfg.h, cfg.w)

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        reference: Tensor | None = None,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        x_profile, x_intensity = _encode_shoebox(
            self.encoder1, self.encoder2, shoebox, self.shoebox_shape
        )

        metadata = None

        if self.encoder3 is not None:
            if reference is None:
                raise ValueError(
                    "A metadata encoder (encoder 3) was provided, but no reference data was found. "
                    "Please provide a `reference.pt` dataset"
                )
            if self.cfg.data_dim == "2d":
                metadata = reference[:, [8, 9, 10]].float()
            else:
                metadata = reference[:, [0, 1, 2, 3, 4, 5, 13]]

            x_metadata = self.encoder3(metadata)

            # combining metadata and profile representation
            x_profile = torch.cat([x_profile, x_metadata], dim=-1)
            x_profile = self.linear(x_profile)

        qbg = self.qbg(x_intensity)
        qi = self.qi(x_intensity)
        qp = self.qp(x_profile)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        rate = zI * zp + zbg

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            concentration=qp.concentration,  # if using Dirichlet
            reference=reference,
        )
        out = _assemble_outputs(out, self.cfg.data_dim)
        return {
            "forward_base_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    def training_step(self, batch, _batch_idx):
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        # Calculate loss
        loss_dict = self.loss(
            rate=outputs["forward_base_out"]["rates"],
            counts=outputs["forward_base_out"]["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=outputs["forward_base_out"]["mask"],
        )

        self.log("Mean(qi.mean)", outputs["qi"].mean.mean())
        self.log("Min(qi.mean)", outputs["qi"].mean.min())
        self.log("Max(qi.mean)", outputs["qi"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        total_loss = loss_dict["loss"]
        kl = loss_dict["kl_mean"]
        nll = loss_dict["neg_ll_mean"]

        self.log(
            "train/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/kl", kl, on_step=False, on_epoch=True)
        self.log("train/nll", nll, on_step=False, on_epoch=True)

        outputs["loss"] = total_loss
        return {
            "loss": total_loss,
            "model_output": outputs["forward_base_out"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=3e-4,
            betas=(0.9, 0.95),
        )
        return optimizer
        # return torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        # )

    def validation_step(self, batch, _batch_idx):
        # Unpack batch
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        loss_dict = self.loss(
            rate=outputs["forward_base_out"]["rates"],
            counts=outputs["forward_base_out"]["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=outputs["forward_base_out"]["mask"],
        )

        total_loss = loss_dict["loss"]
        kl = loss_dict["kl_mean"]
        nll = loss_dict["neg_ll_mean"]

        self.log(
            "val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/kl", kl, on_step=False, on_epoch=True)
        self.log("val/nll", nll, on_step=False, on_epoch=True)

        outputs["loss"] = total_loss
        return {
            "loss": total_loss,
            "model_output": outputs["forward_base_out"],
        }

    def predict_step(self, batch: Tensor, _batch_idx):
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        return {k: v for k, v in outputs.items() if k in self.predict_keys}


if __name__ == "__main__":
    import torch

    from integrator.model.distributions import (
        DirichletDistribution,
        FoldedNormalDistribution,
    )
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        load_config,
    )
    from utils import CONFIGS

    cfg = list(CONFIGS.glob("*"))[0]
    cfg = load_config(cfg)

    integrator = create_integrator(cfg)
    data = create_data_loader(cfg)

    # hyperparameters
    mc_samples = 100

    # distributions
    qbg_ = FoldedNormalDistribution(in_features=64)
    qi_ = FoldedNormalDistribution(in_features=64)
    qp_ = DirichletDistribution(in_features=64, out_features=(3, 21, 21))

    # load a batch
    counts, sbox, mask, meta = next(iter(data.train_dataloader()))

    out = integrator.forward(counts, sbox, mask, meta)

    out = integrator.training_step((counts, sbox, mask, meta), 0)

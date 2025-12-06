from dataclasses import dataclass
from typing import Any, Literal

import polars as pl
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
from integrator.model.loss import BaseLoss


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
        "intensity_mean",
        "intensity_var",
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
        "qbg": out.qbg,
        "qbg_mean": out.qbg.mean,
        "qbg_var": out.qbg.variance,
        "qp": out.qp,
        "qp_mean": out.qp.mean,
        "qi": out.qi,
        "intensity_mean": out.qi.mean,
        "intensity_var": out.qi.variance,
        "profile": out.qp.mean,
        "zp": out.zp,
        "concentration": out.concentration,
    }
    if out.reference is None:
        return base

    base.update(extract_reference_fields(out.reference, data_dim))
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
        # Unpack batch
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

        # calculate profile renyi entropy
        avg_reynyi_entropy = (-(zp.pow(2).sum(-1).log())).mean(-1)

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
        return out

    def training_step(self, batch, _batch_idx):
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        # Calculate loss
        loss_dict = self.loss(
            rate=outputs["rates"],
            counts=outputs["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=outputs["mask"],
        )

        renyi_loss = (
            (
                -torch.log(
                    outputs["qp"]
                    .rsample([self.cfg.mc_samples])
                    .permute(1, 0, 2)
                    .pow(2)
                    .sum(-1)
                )
            )
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

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

        return total_loss + renyi_loss.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def validation_step(self, batch, _batch_idx):
        # Unpack batch
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        loss_dict = self.loss(
            rate=outputs["rates"],
            counts=outputs["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=outputs["mask"],
        )

        total_loss = loss_dict["loss"]
        kl = loss_dict["kl_mean"]
        nll = loss_dict["neg_ll_mean"]

        self.log(
            "val/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("val/kl", kl, on_step=False, on_epoch=True)
        self.log("val/nll", nll, on_step=False, on_epoch=True)

        return outputs

    def predict_step(self, batch: Tensor, _batch_idx):
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        return {k: v for k, v in outputs.items() if k in self.predict_keys}


# %%
class IntegratorNew(LightningModule):
    """Integrator class to infer intenities from raw X-ray diffraction images and experimental metadata."""

    encoder1: ShoeboxEncoder | IntensityEncoder
    """Encoder to get profile distribution"""
    encoder2: ShoeboxEncoder | IntensityEncoder
    """Encoder to get intensity & background distributions"""
    encoder3: MLPMetadataEncoder | None
    """Optional Encoder for experimental metadata"""
    qbg: nn.Module
    """Surrogate posterior shoebox Background"""
    qp: nn.Module
    """Surrogate posterior of spot Profile"""
    qi: nn.Module
    """Surrogate posterior of the spot Intensity"""
    data_dim: str
    """Dimensionality of diffraction data (2d or 3d)"""
    loss: BaseLoss
    """Loss function to optimize."""
    max_iterations: int
    mc_samples: int
    """Number of samples to use for Monte Carlo approximations"""
    d: int
    """Depth of input shoebox."""
    h: int
    """Height on input shoebox."""
    w: int
    """Width of input shoebox."""
    lr: float
    """Learning rate for `torch.optim.Adam`"""
    encoder_out: int
    """Dimension of the encoder codomain"""
    predict_keys: str | list[str]
    """List of keys to store during the `predict_step`. """
    renyi_scale: float
    schema: list[tuple]
    """A `polars.DataFrame` schema to define logged metrics"""
    train_df: pl.DataFrame
    """`DataFrame` with train and validation metrics"""
    weight_decay: float
    """Weight decay value for Adam optimizer."""

    avg_loss: list
    """List containing the average train loss per train epoch"""

    avg_kl: list
    """List containing the average Kullback-Leibler divergence of the train set"""

    avg_nll: list
    """List containing the average validation negative log-likelihood train epoch"""

    val_loss: list
    """List containing the average validation loss per validation epoch"""

    val_kl: list
    """List containing the average Kullback-Leibler divergence per validation epoch"""

    val_nll: list
    """List containing the average validation negative log-likelihood validation epoch"""

    def __init__(
        self,
        qbg: nn.Module,
        qp: nn.Module,
        qi: nn.Module,
        loss: BaseLoss,
        encoder_out: int,
        encoder1: ShoeboxEncoder | IntensityEncoder,
        encoder2: ShoeboxEncoder | IntensityEncoder,
        encoder3: MLPMetadataEncoder | None = None,
        data_dim: str = "3d",  # defaults to rotation data
        d: int = 3,
        h: int = 21,
        w: int = 21,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        mc_samples: int = 100,
        max_iterations: int = 4,
        renyi_scale: float = 0.00,
        predict_keys: list[str] | str = "default",
    ):
        super().__init__()
        self.qbg = qbg
        self.qp = qp
        self.qi = qi
        self.d = d
        self.h = h
        self.w = w
        self.loss = loss
        self.renyi_scale = renyi_scale
        self.data_dim = data_dim
        self.encoder_out = encoder_out

        # encoders
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

        if predict_keys == "default":
            self.predict_keys = [
                "intensity_mean",
                "intensity_var",
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
        elif isinstance(predict_keys, list):
            self.predict_keys = predict_keys

        if self.encoder3 is not None:
            self.linear = nn.Linear(self.encoder_out * 2, self.encoder_out)

        # lists to track avg traning metrics
        self.train_loss = []
        self.train_kl = []
        self.train_nll = []

        # lists to track avg validation metrics
        self.val_loss = []
        self.val_kl = []
        self.val_nll = []
        self.lr = lr
        self.automatic_optimization = True
        self.weight_decay = weight_decay
        self.mc_samples = mc_samples
        self.max_iterations = max_iterations
        self.predict_keys = predict_keys

        #
        if self.data_dim == "3d":
            self.shoebox_shape = (self.d, self.h, self.w)
        elif self.data_dim == "2d":
            self.shoebox_shape = (self.h, self.w)

        # dataframes to keep track of val/train epoch metrics
        self.schema = [
            ("epoch", int),
            ("avg_loss", float),
            ("avg_kl", float),
            ("avg_nll", float),
        ]
        self.train_df = pl.DataFrame(schema=self.schema)
        self.val_df = pl.DataFrame(schema=self.schema)

    def calculate_intensities(self, counts, qbg, qp, mask):
        with torch.no_grad():
            counts = counts * mask  # [B,P]
            zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
            zp = qp.mean.unsqueeze(1)

            vi = zbg + 1e-6
            intensity = torch.tensor([0.0])

            # kabsch sum
            for _ in range(self.max_iterations):
                num = (counts.unsqueeze(1) - zbg) * zp * mask.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                intensity = num.sum(-1) / denom.sum(
                    -1
                )  # [batch_size, mc_samples]
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

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        reference: Tensor | None = None,
    ) -> dict[str, Any]:
        # Unpack batch
        counts = torch.clamp(counts, min=0)

        if shoebox.dim() == 2:
            x_profile = self.encoder1(
                shoebox.reshape(shoebox.shape[0], 1, *(self.shoebox_shape))
            )

            x_intensity = self.encoder2(
                shoebox.reshape(shoebox.shape[0], 1, *(self.shoebox_shape))
            )

        elif shoebox.dim() == 3:
            x_profile = self.encoder1(
                shoebox.reshape(
                    shoebox.size(0), shoebox.size(1), *(self.shoebox_shape)
                )
            )
            x_intensity = self.encoder2(
                shoebox[:, 0, :].reshape(
                    shoebox.size(0), 1, *(self.shoebox_shape)
                )
            )
        else:
            assert ValueError(
                "Incorrect shoebox dimension. The shoebox should be 2 or 3 dimensional"
            )

        if self.encoder3 is not None and reference is None:
            assert ValueError(
                "A metadata encoder (encoder 3) was provided, but no reference data was found. Please provide a `reference.pt` dataset"
            )

        metadata = torch.nn.Identity()

        if self.encoder3 is not None and reference is not None:
            if self.data_dim == "2d" and reference is not None:
                # TODO: Change the datatypes in the DataLoader
                metadata = (
                    reference[:, [8, 9, 10]]
                ).float()  # [wavelength,xcal,ycal]

            elif self.data_dim == "3d" and reference is not None:
                metadata = reference[:, [0, 1, 2, 3, 4, 5, 13]]

            x_metadata = self.encoder3(metadata)

            # combining metadata and intensity representation
            # x_intensity = torch.concat([x_intensity, x_metadata], dim=-1)
            # x_intensity = self.linear(x_intensity)

            # combining metadata and profile representation
            x_profile = torch.concat([x_profile, x_metadata], dim=-1)
            x_profile = self.linear(x_profile)

        qbg = self.qbg(x_intensity)
        qi = self.qi(x_intensity)
        qp = self.qp(x_profile)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        rate = zI * zp + zbg

        # calculate profile renyi entropy
        avg_reynyi_entropy = (-(zp.pow(2).sum(-1).log())).mean(-1)
        out = get_outputs(locals(), self.data_dim)
        return out

    def on_train_epoch_end(self):
        # calculate epoch averages
        avg_train_loss = sum(self.train_loss) / len(self.train_loss)
        avg_kl = sum(self.train_kl) / len(self.train_kl)
        avg_nll = sum(self.train_nll) / len(self.train_nll)

        # log averages to weights & biases
        self.log("train_loss", avg_train_loss)
        self.log("avg_kl", avg_kl)
        self.log("avg_nll", avg_nll)

        # create epoch dataframe
        epoch_df = pl.DataFrame(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_train_loss,
                "avg_kl": avg_kl,
                "avg_nll": avg_nll,
            }
        )

        # udpate training dataframe
        self.train_df = pl.concat([self.train_df, epoch_df])

        # clear all lists
        self.train_loss = []
        self.train_kl = []
        self.train_nll = []

    def on_validation_epoch_end(self):
        avg_val_loss = sum(self.val_loss) / len(self.val_loss)
        avg_kl = sum(self.val_kl) / len(self.val_kl)
        avg_nll = sum(self.val_nll) / len(self.val_nll)

        self.log("validation_loss", avg_val_loss)
        self.log("validation_avg_kl", avg_kl)
        self.log("validation_avg_nll", avg_nll)

        epoch_df = pl.DataFrame(
            {
                "epoch": self.current_epoch,
                "avg_loss": avg_val_loss,
                "avg_kl": avg_kl,
                "avg_nll": avg_nll,
            }
        )
        self.val_df = pl.concat([self.val_df, epoch_df])

        self.val_loss = []
        self.avg_kl = []
        self.val_nll = []

    def training_step(self, batch, _batch_idx):
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        # Calculate loss
        loss_dict = self.loss(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_i=outputs["qi"],
            q_bg=outputs["qbg"],
            mask=outputs["mask"],
        )

        renyi_loss = (
            (
                -torch.log(
                    outputs["qp"]
                    .rsample([self.mc_samples])
                    .permute(1, 0, 2)
                    .pow(2)
                    .sum(-1)
                )
            )
            .mean(1)
            .sum()
        ) * self.renyi_scale
        self.log("renyi_loss", renyi_loss)

        for k, v in loss_dict.items():
            key = f"train_{k}"
            value = v.mean()
            self.log(key, value)

        self.log("Mean(qi.mean)", outputs["qi"].mean.mean())
        self.log("Min(qi.mean)", outputs["qi"].mean.min())
        self.log("Max(qi.mean)", outputs["qi"].mean.max())
        self.log("Mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("Min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("Max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("Mean(qbg.variance)", outputs["qbg"].variance.mean())

        self.train_loss.append(loss_dict["total_loss"].mean())
        self.train_kl.append(loss_dict["kl_mean"].mean())
        self.train_nll.append(loss_dict["neg_ll_mean"].mean())

        return loss_dict["total_loss"].mean() + renyi_loss.sum()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def validation_step(self, batch, _batch_idx):
        # Unpack batch
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        loss_dict = self.loss(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_i=outputs["qi"],
            q_bg=outputs["qbg"],
            mask=outputs["mask"],
        )

        for k, v in loss_dict.items():
            key = f"val_{k}"
            value = v.mean()
            self.log(key, value)

        self.val_loss.append(loss_dict["total_loss"].mean())
        self.val_kl.append(loss_dict["kl_mean"].mean())
        self.val_nll.append(loss_dict["neg_ll_mean"].mean())

        return outputs

    def predict_step(self, batch: Tensor, _batch_idx):
        counts, shoebox, mask, reference = batch
        outputs = self(counts, shoebox, mask, reference)

        return {k: v for k, v in outputs.items() if k in self.predict_keys}


if __name__ == "__main__":
    pass

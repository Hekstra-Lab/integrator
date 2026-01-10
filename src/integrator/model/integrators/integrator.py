from dataclasses import dataclass
from typing import Any, Literal

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from integrator import configs
from integrator.configs.integrator import IntegratorArgs
from integrator.model.distributions import (
    DirichletDistribution,
    FoldedNormalDistribution,
)
from integrator.model.encoders import (
    IntensityEncoder,
    ShoeboxEncoder,
)
from integrator.utils.refl_utils import DEFAULT_DS_COLS

# Default keys to return for prediction
DEFAULT_PREDICT_KEYS = [
    "qi_mean",
    "qi_var",
    "refl_ids",
    "qbg_mean",
    "qbg_var",
    "qbg_scale",
    "intensity.prf.value",
    "intensity.prf.variance",
    "intensity.sum.value",
    "intensity.sum.variance",
    "background.mean",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "H",
    "K",
    "L",
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
    metadata: dict[str, torch.Tensor]
    concentration: Tensor | None = None


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


def extract_metadata_fields(
    ref: Tensor,
    data_dim: str,
):
    if data_dim == "3d":
        return ref
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


class OutputHandler:
    def __init__(
        self,
    ):
        self.base = {
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

    if out.metadata is None:
        return base

    ref_fields = extract_metadata_fields(out.metadata, data_dim)

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
            shoebox.reshape(shoebox.size(0), 1, *(shoebox_shape))
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
    encoder3: ShoeboxEncoder | IntensityEncoder | None = None


@dataclass
class SurrogateModules:
    qbg: nn.Module
    qi: nn.Module
    qp: nn.Module


def stats(name, x):
    print(
        f"{name}:   min={x.min().item():.4e}, max={x.max().item():.4e}, "
        f"mean={x.mean().item():.4e}, std={x.std().item():.4e}"
    )


# %%
def mean_pool_by_image(emb: torch.Tensor, img_ids: torch.Tensor):
    device = emb.device
    B, F = emb.shape

    # 1. Get unique image IDs and mapping
    pooled_ids, per_ref_idx = torch.unique(img_ids, return_inverse=True)
    n_img = pooled_ids.size(0)

    # 2. Sum per image using index_add_
    sums = torch.zeros(n_img, F, device=device)
    counts = torch.zeros(n_img, 1, device=device)

    sums.index_add_(0, per_ref_idx, emb)  # sum embeddings per image
    ones = torch.ones(B, 1, device=device)
    counts.index_add_(0, per_ref_idx, ones)  # count per image

    pooled = sums / counts.clamp_min(1.0)  # mean embedding per image
    return pooled, pooled_ids, per_ref_idx


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
        metadata: Tensor | None = None,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        x_profile, x_intensity = _encode_shoebox(
            self.encoder1, self.encoder2, shoebox, self.shoebox_shape
        )

        # im_sbox, pooled_ids, per_image_idx = mean_pool_by_image(
        #     shoebox, metadata[:, 2].float()
        # )
        #
        # num_images = im_sbox.shape[0]
        #
        # self.log("Number of images per batch", num_images)

        # NOTE: temporarily commenting out
        # im_rep = self.encoder3(
        #     im_sbox.reshape(num_images, 1, *(self.shoebox_shape))
        # )

        # im_rep = self.encoder3(im_sbox)

        # if self.encoder3 is not None:
        #     if metadata is None:
        #         raise ValueError(
        #             "A metadata encoder (encoder 3) was provided, but no metadata data was found. "
        #             "Please provide a `metadata.pt` dataset"
        #         )
        #     if self.cfg.data_dim == "2d":
        #         metadata = metadata[:, [2, 8, 9, 10]].float()
        #         # %%
        #         max = torch.log1p(counts.max(-1)[0]).unsqueeze(-1)
        #         min = torch.log1p(counts.min(-1)[0]).unsqueeze(-1)
        #         mean = torch.log1p(counts.mean(-1)).unsqueeze(-1)
        #         std = torch.log1p(counts.std(-1)).unsqueeze(-1)
        #
        #         metadata = torch.stack([max, min, mean, std], -1).squeeze(1)
        #
        #     else:
        #         metadata = metadata[:, [0, 1, 2, 3, 4, 5, 13]]
        #
        #     x_metadata = self.encoder3(metadata)
        #
        #     # combining metadata and profile representation
        #     x_profile = torch.cat([x_profile, x_metadata], dim=-1)
        #     x_profile = self.linear(x_profile)
        #
        # qri = self.qri(x_intensity)
        # qrbg = self.qrbg(x_intensity)
        # rbg = qrbg.rsample([self.mc_samples]).mean(0)
        # ri = qri.rsample([self.mc_samples]).mean(0)

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
            metadata=metadata,
        )
        out = _assemble_outputs(out, self.cfg.data_dim)
        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    def training_step(self, batch, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)

        # Calculate loss
        loss_dict = self.loss(
            rate=outputs["forward_out"]["rates"],
            counts=outputs["forward_out"]["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=outputs["forward_out"]["mask"],
        )

        self.log("train: mean(qi.mean)", outputs["qi"].mean.mean())
        self.log("train: min(qi.mean)", outputs["qi"].mean.min())
        self.log("train: max(qi.mean)", outputs["qi"].mean.max())
        self.log("train: max(qi.variance)", outputs["qi"].variance.max())
        self.log("train: min(qi.variance)", outputs["qi"].variance.min())
        self.log("train: mean(qi.variance)", outputs["qi"].variance.mean())

        self.log("train: mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("train: min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("train: max(qbg.mean)", outputs["qbg"].mean.max())
        self.log("train: mean(qbg.variance)", outputs["qbg"].variance.mean())
        self.log("train: max(qbg.variance)", outputs["qbg"].variance.max())
        self.log("train: min(qbg.variance)", outputs["qbg"].variance.min())

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
            "forward_out": outputs["forward_out"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def validation_step(self, batch, _batch_idx):
        # Unpack batch
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)

        loss_dict = self.loss(
            rate=outputs["forward_out"]["rates"],
            counts=outputs["forward_out"]["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            # # qri=outputs["qri"],
            # qrbg=outputs["qrbg"],
            mask=outputs["forward_out"]["mask"],
        )

        self.log("validation: mean(qi.mean)", outputs["qi"].mean.mean())
        self.log("validation: min(qi.mean)", outputs["qi"].mean.min())
        self.log("validation: max(qi.mean)", outputs["qi"].mean.max())
        self.log("validation: max(qi.variance)", outputs["qi"].variance.max())
        self.log("validation: min(qi.variance)", outputs["qi"].variance.min())
        self.log(
            "validation: mean(qi.variance)", outputs["qi"].variance.mean()
        )

        self.log("validation: mean(qbg.mean)", outputs["qbg"].mean.mean())
        self.log("validation: min(qbg.mean)", outputs["qbg"].mean.min())
        self.log("validation: max(qbg.mean)", outputs["qbg"].mean.max())
        self.log(
            "validation: mean(qbg.variance)", outputs["qbg"].variance.mean()
        )
        self.log(
            "validation: max(qbg.variance)", outputs["qbg"].variance.max()
        )
        self.log(
            "validation: min(qbg.variance)", outputs["qbg"].variance.min()
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
            "forward_out": outputs["forward_out"],
        }

    def predict_step(self, batch: Tensor, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)

        return {
            k: v
            for k, v in outputs["forward_out"].items()
            if k in self.predict_keys
        }


# to use:
# log_validation_distributions()
def log_validation_distributions(
    logger,
    outputs: dict,
    stage: str = "validation",
):
    qi = outputs["qi"]
    qbg = outputs["qbg"]

    logger.log(f"{stage}/qi/mean", qi.mean.mean())
    logger.log(f"{stage}/qi/mean_min", qi.mean.min())
    logger.log(f"{stage}/qi/mean_max", qi.mean.max())

    logger.log(f"{stage}/qi/var_mean", qi.variance.mean())
    logger.log(f"{stage}/qi/var_min", qi.variance.min())
    logger.log(f"{stage}/qi/var_max", qi.variance.max())

    logger.log(f"{stage}/qbg/mean", qbg.mean.mean())
    logger.log(f"{stage}/qbg/mean_min", qbg.mean.min())
    logger.log(f"{stage}/qbg/mean_max", qbg.mean.max())

    logger.log(f"{stage}/qbg/var_mean", qbg.variance.mean())
    logger.log(f"{stage}/qbg/var_min", qbg.variance.min())
    logger.log(f"{stage}/qbg/var_max", qbg.variance.max())


@dataclass
class IntegratorModelBArgs:
    cfg: IntegratorArgs
    loss: nn.Module
    surrogates: dict[str, nn.Module]
    encoders: dict[str, nn.Module]


def _log_forward_out(
    self,
    forward_out: dict,
    step: Literal["train", "val"],
):
    self.log(f"{step}: mean(qi.mean)", forward_out["qi_mean"].mean())
    self.log(f"{step}: min(qi.mean)", forward_out["qi_mean"].min())
    self.log(f"{step}: max(qi.mean)", forward_out["qi_mean"].max())
    self.log(f"{step}: max(qi.variance)", forward_out["qi_var"].max())
    self.log(f"{step}: min(qi.variance)", forward_out["qi_var"].min())
    self.log(f"{step}: mean(qi.variance)", forward_out["qi_var"].mean())
    self.log(f"{step}: mean(qbg.mean)", forward_out["qbg_mean"].mean())
    self.log(f"{step}: min(qbg.mean)", forward_out["qbg_mean"].min())
    self.log(f"{step}: max(qbg.mean)", forward_out["qbg_mean"].max())
    self.log(f"{step}: mean(qbg.variance)", forward_out["qbg_var"].mean())
    self.log(f"{step}: max(qbg.variance)", forward_out["qbg_var"].max())
    self.log(f"{step}: min(qbg.variance)", forward_out["qbg_var"].min())


def _log_loss(
    self,
    kl,
    nll,
    total_loss,
    step: Literal["train", "val"],
):
    self.log(
        "train/loss",
        total_loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    self.log(f"{step} kl", kl, on_step=False, on_epoch=True)
    self.log(f"{step} nll", nll, on_step=False, on_epoch=True)


# %%
class IntegratorModelB(LightningModule):
    REQUIRED_ENCODERS = {
        "encoder1": configs.ShoeboxEncoderArgs,
        "encoder2": configs.IntensityEncoderArgs,
        "encoder3": configs.IntensityEncoderArgs,
    }
    ARGS = IntegratorModelBArgs

    def __init__(
        self,
        cfg: IntegratorArgs,
        loss: nn.Module,
        surrogates: dict[str, nn.Module],
        encoders: dict[str, nn.Module],
    ):
        super().__init__()
        self.cfg = cfg

        # hyperparams
        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay
        self.mc_samples = cfg.mc_samples
        self.renyi_scale = cfg.renyi_scale
        self.shoebox_shape = (cfg.d, cfg.h, cfg.w)

        # posterior modules
        self.qbg = surrogates["qbg"]
        self.qp = surrogates["qp"]
        self.qi = surrogates["qi"]

        # encoder modules
        self.encoder1 = encoders["encoder1"]
        self.encoder2 = encoders["encoder2"]
        self.encoder3 = encoders["encoder3"]

        # loss module
        self.loss = loss

        # predict keys
        self.predict_keys = (
            DEFAULT_PREDICT_KEYS
            if cfg.predict_keys == "default"
            else cfg.predict_keys
        )

        # additional hyperparameters
        self.automatic_optimization = True

    def forward(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        # removing negative valued pixels from raw counts
        counts = torch.clamp(counts, min=0)

        # representations
        x_profile = self.encoder1(
            shoebox.reshape(shoebox.shape[0], 1, *(self.shoebox_shape))
        )

        x_k = self.encoder2(
            shoebox.reshape(shoebox.shape[0], 1, *(self.shoebox_shape))
        )

        x_r = self.encoder3(
            shoebox.reshape(shoebox.shape[0], 1, *(self.shoebox_shape))
        )

        # surrogate distributions
        qbg = self.qbg(x_k, x_r)
        qi = self.qi(x_k, x_r)
        qp = self.qp(x_profile)

        # Monte carlo samples
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qi.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        # Poisson rate
        rate = zI * zp + zbg

        # outputs
        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            metadata=metadata,
            concentration=qp.concentration,
        )
        out = _assemble_outputs(out, self.cfg.data_dim)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    def training_step(self, batch, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        # Calculate loss
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
        )

        # Loss components
        total_loss = loss_dict["loss"]
        kl = loss_dict["kl_mean"]
        nll = loss_dict["neg_ll_mean"]
        outputs["loss"] = total_loss

        # Logging outputs to W&B
        _log_forward_out(self, forward_out=forward_out, step="train")
        _log_loss(self, kl=kl, nll=nll, total_loss=total_loss, step="train")

        return {
            "loss": total_loss,
            "forward_out": forward_out,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def validation_step(self, batch, _batch_idx):
        # Unpack batch
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        # Calculate loss
        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
        )

        total_loss = loss_dict["loss"]
        kl = loss_dict["kl_mean"]
        nll = loss_dict["neg_ll_mean"]
        outputs["loss"] = total_loss

        _log_forward_out(self, forward_out=forward_out, step="val")
        _log_loss(self, kl=kl, nll=nll, total_loss=total_loss, step="val")

        return {
            "loss": total_loss,
            "forward_out": forward_out,
        }

    def predict_step(self, batch: Tensor, _batch_idx):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)

        return {
            k: v
            for k, v in outputs["forward_out"].items()
            if k in self.predict_keys
        }


if __name__ == "__main__":
    # %%
    import tempfile
    from pathlib import Path

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

    # generating sample data
    dataset_size = 1000
    batch_size = 10

    # shoebox dimensions
    depth = 3
    height = 21
    width = 21
    n_pix = depth * height * width

    counts = torch.randint(0, 10, (dataset_size, n_pix), dtype=torch.float32)
    masks = torch.randint(0, 2, (dataset_size, n_pix))
    stats = torch.tensor([0.0, 1.0])
    concentration = counts.mean(0)

    data = {}
    for c in DEFAULT_DS_COLS:
        data[c] = torch.randn(dataset_size)

    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        cfg["data_dir"] = tdir
        cfg["data_loader"]["args"]["data_dir"] = tdir
        torch.save(counts, tdir / "counts_3d_subset.pt")
        torch.save(masks, tdir / "masks_3d_subset.pt")
        torch.save(data, tdir / "reference_3d_subset.pt")
        torch.save(stats, tdir / "stats_3d.pt")
        torch.save(stats, tdir / "concentration_3d.pt")

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
# %%

import atexit
import shutil
import tempfile
from dataclasses import fields
from typing import Literal

import torch

from .data_loader import DataFileNames, DataLoaderArgs, DataLoaderConfig
from .distributions import (
    DirichletArgs,
    SurrogateArgs,
    SurrogateConfig,
    Surrogates,
)
from .encoder import EncoderConfig, IntensityEncoderArgs, ShoeboxEncoderArgs
from .global_config import GlobalConfig
from .integrator import IntegratorCfg, IntegratorConfig
from .logger import LoggerConfig
from .loss import LossArgs, LossConfig
from .output import OutputConfig
from .priors import DirichletParams, GammaParams, PriorConfig
from .trainer import TrainerConfig
from .yaml_config import YAMLConfig


def shallow_dict(dc) -> dict:
    return {f.name: getattr(dc, f.name) for f in fields(dc)}


TUPLE_FIELDS = {
    "input_shape",
    "conv1_kernel_size",
    "conv1_padding",
    "pool_kernel_size",
    "pool_stride",
    "conv2_kernel_size",
    "conv2_padding",
    "conv3_kernel_size",
    "conv3_padding",
    "shape",
    "sbox_shape",
}


DEFAULT_DS_COLS = [
    "zeta",
    "xyzobs.px.variance.0",
    "xyzobs.px.variance.1",
    "xyzobs.px.variance.2",
    "xyzobs.px.value.0",
    "xyzobs.px.value.1",
    "xyzobs.px.value.2",
    "xyzobs.mm.variance.0",
    "xyzobs.mm.variance.1",
    "xyzobs.mm.variance.2",
    "xyzobs.mm.value.0",
    "xyzobs.mm.value.1",
    "xyzobs.mm.value.2",
    "xyzcal.mm.0",
    "xyzcal.mm.1",
    "xyzcal.mm.2",
    "refl_ids",
    "qe",
    "profile.correlation",
    "partiality",
    "partial_id",
    "panel",
    "num_pixels.valid",
    "num_pixels.foreground",
    "num_pixels.background_used",
    "num_pixels.background",
    "lp",
    "intensity.prf.variance",
    "intensity.prf.value",
    "imageset_id",
    "flags",
    "entering",
    "d",
    "bbox.0",
    "bbox.1",
    "bbox.2",
    "bbox.3",
    "bbox.4",
    "bbox.5",
    "background.sum.variance",
    "background.sum.value",
    "background.mean",
    "s1.0",
    "s1.1",
    "s1.2",
    "xyzcal.px.0",
    "xyzcal.px.1",
    "xyzcal.px.2",
    "intensity.sum.variance",
    "intensity.sum.value",
    "H",
    "K",
    "L",
]


def generate_data_files(
    data_dir: str,
    save_files: bool = True,
    depth: int = 3,
    height: int = 21,
    width: int = 21,
    dataset_size: int = 1000,
) -> dict[str, str]:
    # shoebox dimensions
    n_pix = depth * height * width

    counts = torch.randint(0, 10, (dataset_size, n_pix), dtype=torch.float32)
    masks = torch.randint(0, 2, (dataset_size, n_pix))
    stats = torch.tensor([0.0, 1.0])
    concentration = counts.mean(0)

    metadata = {}
    for c in DEFAULT_DS_COLS:
        metadata[c] = torch.randn(dataset_size)
    metadata["refl_ids"] = torch.randint(
        low=0, high=10000, size=(dataset_size,)
    )

    fnames = {
        "data_dir": data_dir,
        "masks": data_dir + "/masks.pt",
        "counts": data_dir + "/counts.pt",
        "reference": data_dir + "/reference.pt",
        "stats": data_dir + "/stats.pt",
        "concentration": data_dir + "/concentration.pt",
    }

    if save_files:
        torch.save(counts, fnames["counts"])
        torch.save(masks, fnames["masks"])
        torch.save(metadata, fnames["reference"])
        torch.save(stats, fnames["stats"])
        torch.save(concentration, fnames["concentration"])

    return fnames


def make_temp_data_dir():
    path = tempfile.mkdtemp()
    atexit.register(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


def construct_yaml_configuration(
    data_dir: str,
    batch_size: int = 100,
    encoder_out: int = 64,
    mc_samples: int = 100,
    data_dim: Literal["2d", "3d"] = "3d",
    d: int = 3,
    h: int = 21,
    w: int = 21,
) -> YAMLConfig:
    global_cfg = GlobalConfig(
        encoder_out=encoder_out,
        mc_samples=mc_samples,
        data_dir=data_dir,
        d=d,
        h=h,
        w=w,
    )
    integrator_args = IntegratorCfg(
        data_dim=data_dim,
        d=global_cfg.d,
        h=global_cfg.h,
        w=global_cfg.w,
    )

    integrator_cfg = IntegratorConfig(
        name="modelb",
        args=integrator_args,
    )
    # Defining encoder modules
    shoebox_encoder_args = ShoeboxEncoderArgs(
        data_dim=data_dim,
        in_channels=1,
        input_shape=(d, h, w),  # d,w,h
        encoder_out=global_cfg.encoder_out,
        conv1_out_channels=global_cfg.encoder_out,
        conv1_kernel_size=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=128,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
    )

    shoebox_encoder_cfg = EncoderConfig(
        name="shoebox_encoder",
        args=shoebox_encoder_args,
    )

    intensity_encoder1_args = IntensityEncoderArgs(
        data_dim=data_dim,
        in_channels=1,
        encoder_out=global_cfg.encoder_out,
        conv1_out_channels=global_cfg.encoder_out,
        conv1_kernel_size=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=128,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
        conv3_out_channels=256,
        conv3_kernel_size=(3, 3, 3),
        conv3_padding=(1, 1, 1),
        norm3_num_groups=8,
    )

    intensity_encoder1_cfg = EncoderConfig(
        name="intensity_encoder",
        args=intensity_encoder1_args,
    )

    intensity_encoder2_args = IntensityEncoderArgs(
        data_dim=data_dim,
        in_channels=1,
        encoder_out=global_cfg.encoder_out,
        conv1_out_channels=global_cfg.encoder_out,
        conv1_kernel_size=(1, 3, 3),
        conv1_padding=(0, 1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(1, 2, 2),
        pool_stride=(1, 2, 2),
        conv2_out_channels=128,
        conv2_kernel_size=(3, 3, 3),
        conv2_padding=(0, 0, 0),
        norm2_num_groups=4,
        conv3_out_channels=256,
        conv3_kernel_size=(3, 3, 3),
        conv3_padding=(1, 1, 1),
        norm3_num_groups=8,
    )

    intensity_encoder2_cfg = EncoderConfig(
        name="intensity_encoder",
        args=intensity_encoder2_args,
    )

    # Defining surrogate distributions
    qp_args = DirichletArgs(
        eps=1e-6,
        in_features=global_cfg.encoder_out,
        sbox_shape=(3, 21, 21),
    )
    qp = SurrogateConfig(
        name="dirichlet",
        args=qp_args,
    )

    qbg_args = SurrogateArgs(
        eps=1e-6,
        in_features=global_cfg.encoder_out,
    )
    qbg = SurrogateConfig(
        name="gammaA",
        args=qbg_args,
    )

    qi_args = SurrogateArgs(
        eps=1e-6,
        in_features=global_cfg.encoder_out,
    )
    qi = SurrogateConfig(
        name="gammaA",
        args=qi_args,
    )

    surrogates = Surrogates(
        qp=qp,
        qbg=qbg,
        qi=qi,
    )

    # defining loss module
    pbg_params = GammaParams(
        concentration=0.01,
        rate=0.001,
    )
    pbg = PriorConfig(
        name="gamma",
        params=pbg_params,
        weight=0.5,
    )

    pi_params = GammaParams(
        concentration=0.01,
        rate=0.001,
    )

    pi = PriorConfig(
        name="gamma",
        params=pi_params,
        weight=0.5,
    )

    pprf_params = DirichletParams(
        concentration=0.01,
        shape=(d, w, h),
    )

    pprf = PriorConfig(
        name="dirichlet",
        params=pprf_params,
        weight=0.001,
    )

    loss_args = LossArgs(
        mc_samples=100,
        eps=1e-6,
        pprf_cfg=pprf,
        pbg_cfg=pbg,
        pi_cfg=pi,
    )

    loss = LossConfig(
        name="default",
        args=loss_args,
    )

    # defining data_loader
    data_file_names = DataFileNames(
        data_dir=data_dir,
        counts="counts.pt",
        masks="masks.pt",
        stats="stats.pt",
        reference="reference.pt",
        standardized_counts=None,
    )

    data_loader_args = DataLoaderArgs(
        data_dir=global_cfg.data_dir,
        batch_size=batch_size,
        val_split=0.3,
        test_split=0.0,
        num_workers=0,
        include_test=False,
        subset_size=100,
        cutoff=None,
        use_metadata=True,
        shoebox_file_names=data_file_names,
        D=global_cfg.d,
        H=global_cfg.h,
        W=global_cfg.w,
        anscombe=True,
    )

    data_loader_cfg = DataLoaderConfig(
        name="default",
        args=data_loader_args,
    )

    trainer = TrainerConfig(
        max_epochs=10,
        accelerator="cpu",
        devices=1,
        logger=True,
        precision="32",
        check_val_every_n_epoch=2,
        log_every_n_steps=1,
        deterministic=False,
        enable_checkpointing=True,
    )
    logger = LoggerConfig(
        d=global_cfg.d,
        h=global_cfg.h,
        w=global_cfg.w,
    )
    output = OutputConfig(
        refl_file="/Users/luis/master/harvard_phd/dac/second_dac/analysis/reflections_.refl"
    )

    yaml_cfg = YAMLConfig(
        global_vars=global_cfg,
        integrator=integrator_cfg,
        encoders=[
            shoebox_encoder_cfg,
            intensity_encoder1_cfg,
            intensity_encoder2_cfg,
        ],
        surrogates=surrogates,
        loss=loss,
        data_loader=data_loader_cfg,
        trainer=trainer,
        logger=logger,
        output=output,
    )

    return yaml_cfg

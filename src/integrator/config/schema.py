from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredWriterCfg(BaseModel):
    output_dir: Path | None = None
    write_interval: Literal["batch", "epoch"] = "epoch"


class TrainerArgs(BaseModel):
    max_epochs: int = 600
    accelerator: str | Literal["auto"] = "auto"
    devices: int | Literal["auto"] = 1
    logger: bool = True
    precision: str = "32"
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1
    deterministic: bool = False
    callbacks: dict[str, PredWriterCfg] = Field(default_factory=dict)
    enable_checkpointing: bool = True


class TrainerCfg(BaseModel):
    """
    Accepts *either*:
      trainer:
        args: {...}
    or
      trainer:
        params: {...}
    and normalizes to .args
    """

    model_config = ConfigDict(populate_by_name=True)
    args: TrainerArgs = Field(default_factory=TrainerArgs, alias="args")


class DataLoaderParams(BaseModel):
    data_dir: Path
    batch_size: int = 10
    val_split: float = 0.3
    test_split: float = 0.0
    num_workers: int = 0
    include_test: bool = False
    subset_size: int | None = None
    cutoff: int | None = None
    use_metadata: bool = True
    anscombe: bool = True
    H: int = 21
    W: int = 21
    shoebox_file_names: dict[str, Path | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _resolve_relative_file_names(self) -> DataLoaderParams:
        """
        If shoebox_file_names contains relative paths, resolve them against data_dir.
        Leaves absolute paths and None as-is.
        """
        resolved: dict[str, Path | None] = {}
        for k, v in (self.shoebox_file_names or {}).items():
            if v is None:
                resolved[k] = None
            else:
                p = Path(v)
                resolved[k] = p if p.is_absolute() else (self.data_dir / p)
        self.shoebox_file_names = resolved
        return self


class DataLoaderCfg(BaseModel):
    name: str = "default"
    args: DataLoaderParams


class GlobalCfg(BaseModel):
    in_features: int = 64
    mc_samples: int = 100
    data_dir: Path


class IntegratorArgs(BaseModel):
    data_dim: str
    encoder_out: int
    lr: float = 0.001
    mc_samples: int = 100
    renyi_scale: float = 0.0
    d: int = 3
    h: int = 21
    w: int = 21
    weight_decay: float = 0.0
    predict_keys: str | list[str]


class IntegratorCfg(BaseModel):
    name: str = "integrator"
    args: IntegratorArgs


class Component(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ComponentsCfg(BaseModel):
    """
    YAML structure:
      components:
        encoders:
          - encoder1: { name: ..., args: {...} }
          - encoder2: { name: ..., args: {...} }
        qp:  { name: ..., args: {...} }
        qbg: { name: ..., args: {...} }
        qi:  { name: ..., args: {...} }
        loss:{ name: ..., args: {...} }

    encoders is a list of one-key dicts so you keep the encoder name.
    """

    encoders: list[dict[str, Component]] = Field(default_factory=list)
    qp: Component
    qbg: Component
    qi: Component
    loss: Component

    @model_validator(mode="after")
    def _must_have_at_least_one_encoder(self) -> ComponentsCfg:
        if not self.encoders:
            raise ValueError(
                "components.encoders must contain at least one encoder"
            )
        return self


class LoggerCfg(BaseModel):
    d: int = 3
    h: int = 21
    w: int = 21


class Cfg(BaseModel):
    global_vars: GlobalCfg
    integrator: IntegratorCfg
    trainer: TrainerCfg
    data_loader: DataLoaderCfg
    components: ComponentsCfg
    logger: LoggerCfg

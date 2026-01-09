from dataclasses import dataclass

from .data_loader import DataLoaderConfig
from .distributions import Surrogates
from .encoder import EncoderConfig
from .global_config import GlobalConfig
from .integrator import IntegratorConfig
from .logger import LoggerConfig
from .loss import LossConfig
from .output import OutputConfig
from .trainer import TrainerConfig


@dataclass
class YAMLConfig:
    global_vars: GlobalConfig
    integrator: IntegratorConfig
    encoders: list[EncoderConfig]
    surrogates: Surrogates
    loss: LossConfig
    data_loader: DataLoaderConfig
    trainer: TrainerConfig
    logger: LoggerConfig
    output: OutputConfig

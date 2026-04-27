from dataclasses import dataclass

from .data_loader import DataLoaderConfig
from .distributions import Surrogates
from .encoder import EncoderConfig
from .integrator import IntegratorConfig
from .logger import LoggerConfig
from .loss import LossConfig
from .output import OutputConfig
from .trainer import TrainerConfig


@dataclass
class YAMLConfig:
    integrator: IntegratorConfig
    encoders: list[EncoderConfig]
    surrogates: Surrogates
    loss: LossConfig
    data_loader: DataLoaderConfig
    trainer: TrainerConfig
    logger: LoggerConfig
    output: OutputConfig

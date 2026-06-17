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
    """Top-level run configuration parsed from a YAML file.

    Each field is a typed sub-config; together they fully specify a run.

    Attributes:
        integrator: Integrator model selection and hyperparameters.
        encoders: Encoder selections, one per named encoder.
        surrogates: The profile/background/intensity surrogate selections.
        loss: ELBO loss selection and priors.
        data_loader: Data-module selection and arguments.
        trainer: Lightning `Trainer` settings.
        logger: Logger panel dimensions.
        output: Output artifact paths.
    """

    integrator: IntegratorConfig
    encoders: list[EncoderConfig]
    surrogates: Surrogates
    loss: LossConfig
    data_loader: DataLoaderConfig
    trainer: TrainerConfig
    logger: LoggerConfig
    output: OutputConfig

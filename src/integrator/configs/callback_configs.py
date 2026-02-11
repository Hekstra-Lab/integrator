from dataclasses import dataclass


@dataclass
class PlotterCfg:
    n_profiles: int
    plot_every_n_epochs: int
    d: int
    h: int
    w: int


@dataclass
class EpochMetricRecorderCfg:
    split: str
    max_rows_per_epoch: int
    split: str


@dataclass
class ModelCheckpointCfg:
    every_n_epochs: int


@dataclass
class TrainCallbacks:
    callbacks: list

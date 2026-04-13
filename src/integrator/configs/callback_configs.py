from dataclasses import dataclass


@dataclass
class PlotterCfg:
    n_profiles: int
    plot_every_n_epochs: int
    d: int
    h: int
    w: int

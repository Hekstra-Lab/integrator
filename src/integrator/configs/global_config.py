from dataclasses import dataclass


@dataclass
class GlobalConfig:
    encoder_out: int
    mc_samples: int
    data_dir: str
    d: int
    h: int
    w: int

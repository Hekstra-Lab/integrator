from dataclasses import dataclass


@dataclass
class SurrogateArgs:
    in_features: int
    eps: float
    k_max: float | None = None
    k_min: float = 0.1


@dataclass
class DirichletArgs:
    in_features: int
    sbox_shape: tuple[int, ...]
    eps: float


@dataclass
class SurrogateConfig:
    name: str
    args: SurrogateArgs | DirichletArgs


@dataclass
class Surrogates:
    qp: SurrogateConfig
    qbg: SurrogateConfig
    qi: SurrogateConfig

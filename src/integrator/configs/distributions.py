from dataclasses import dataclass


@dataclass
class SurrogateArgs:
    in_features: int
    eps: float


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

from dataclasses import dataclass


@dataclass
class SurrogateArgs:
    in_features: int
    eps: float
    k_min: float = 0.1

    def __post_init__(self):
        if self.in_features < 1:
            raise ValueError(
                f"in_features must be >= 1, got {self.in_features}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")


@dataclass
class DirichletArgs:
    in_features: int
    sbox_shape: tuple[int, ...]
    eps: float

    def __post_init__(self):
        if self.in_features < 1:
            raise ValueError(
                f"in_features must be >= 1, got {self.in_features}"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")


@dataclass
class SurrogateConfig:
    name: str
    args: SurrogateArgs | DirichletArgs


@dataclass
class Surrogates:
    qp: SurrogateConfig
    qbg: SurrogateConfig
    qi: SurrogateConfig

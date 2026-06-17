from dataclasses import dataclass


@dataclass
class SurrogateArgs:
    """Constructor arguments for the intensity/background (`qi`/`qbg`) surrogates.

    Attributes:
        in_features: Width of the encoder embedding fed to the surrogate head; must be `>= 1`.
        eps: Numerical floor added for stability; must be positive.
        k_min: Lower clamp on the Gamma shape parameter, preventing the `rsample` overflow at small shape.
    """

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
    """Constructor arguments for the Dirichlet profile surrogate.

    Attributes:
        in_features: Width of the encoder embedding fed to the surrogate head; must be `>= 1`.
        sbox_shape: Shoebox shape `(depth, height, width)` defining the number of profile components.
        eps: Numerical floor added for stability; must be positive.
    """

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
    """Registry selection for a single surrogate: a `name` plus its typed `args`.

    Attributes:
        name: Registry key naming the surrogate class to construct.
        args: Constructor arguments, a `SurrogateArgs` or `DirichletArgs`.
    """

    name: str
    args: SurrogateArgs | DirichletArgs


@dataclass
class Surrogates:
    """The three variational surrogates of the ELBO: profile, background, intensity.

    Attributes:
        qp: Profile surrogate configuration.
        qbg: Background surrogate configuration.
        qi: Intensity surrogate configuration.
    """

    qp: SurrogateConfig
    qbg: SurrogateConfig
    qi: SurrogateConfig

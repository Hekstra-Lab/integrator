from dataclasses import dataclass


@dataclass
class LoggerConfig:
    """Shoebox dimensions the logger uses to render profile and image panels.

    Attributes:
        d: Shoebox depth in pixels.
        h: Shoebox height in pixels.
        w: Shoebox width in pixels.
    """

    d: int
    h: int
    w: int

import logging
import sys

from rich.logging import RichHandler


def is_tty():
    return sys.stderr.isatty()


def setup_logging(verbosity: int = 0):
    """
    verbosity = 0 -> WARNING
    verbosity = 1 -> INFO
    verbosity >= 2 -> DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handlers = []

    if sys.stderr.isatty():
        # Interactive terminal -> Rich
        handlers.append(
            RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
            )
        )
    else:
        # SLURM / redirected output -> plain text
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if not sys.stderr.isatty()
        else "%(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )

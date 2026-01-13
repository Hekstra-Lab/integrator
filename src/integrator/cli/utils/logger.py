import logging
import sys


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

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

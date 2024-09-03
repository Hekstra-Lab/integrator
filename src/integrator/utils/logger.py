import os
from pytorch_lightning.loggers import TensorBoardLogger


def init_tensorboard_logger(save_dir="logs", name="integrator_model"):
    """
    Initialize a TensorBoard logger.

    Args:
        save_dir (str): Directory where the logs will be saved.
        name (str): Name of the logger.

    Returns:
        TensorBoardLogger: Initialized logger.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the TensorBoard logger
    logger = TensorBoardLogger(save_dir=save_dir, name=name)
    return logger

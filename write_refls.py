# from lightning.pytorch.loggers import TensorBoardLogger
import argparse
import glob
from pathlib import Path

from integrator.utils import (
    reflection_file_writer,
)

if __name__ == "__main__":
    # load data
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--path",
        type=str,
    )

    args = argparser.parse_args()

    path = Path(args.path)
    pred_path = path.as_posix() + "/files/predictions"
    pred_dirs = glob.glob(pred_path + "/epoch*")
    prediction_files = glob.glob(pred_path + "/epoch*/*.pt")

    print(prediction_files)
    print(pred_dirs)

    reflection_file_writer(
        pred_dirs,
        prediction_files,
        "/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator/data/pass1/reflections_.refl"
    )



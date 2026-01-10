import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict from a set of pytorch.ckpt files"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="Path to dir containing pred.yaml file",
    )
    parser.add_argument(
        "--data-dim",
        type=str,
        default="3d",
        help="String denoting shoebox dimensions; e.g. 2d or 3d",
    )
    parser.add_argument(
        "--write-refl",
        action="store_true",
        help="Write predictions as a .refl file",
    )
    # parser.add_argument(
    #     "--write-as-mtz",
    #     action="store_true",
    #     help="Write predictions as a .mtz file",
    # )
    return parser.parse_args()


def write_refl_from_preds(
    ckpt_dir,
    refl_file,
    epoch: int,
):
    import pandas as pd
    import reciprocalspaceship as rs
    import torch

    from integrator.utils.refl_utils import (
        DEFAULT_REFL_COLS,
        unstack_preds,
        write_refl_from_ds,
    )

    pred_file = list(ckpt_dir.glob("preds.pt"))[0]
    data = torch.load(pred_file, weights_only=False)
    fname = ckpt_dir / f"preds_epoch_{epoch}.refl"

    ds = rs.io.read_dials_stills(refl_file, extra_cols=DEFAULT_REFL_COLS)
    unstacked_preds = unstack_preds(data)

    id_filter = ds["refl_ids"].isin(unstacked_preds["refl_ids"])
    ds_filtered = ds[id_filter].sort_values(by="refl_ids")

    pred_df = pd.DataFrame(unstacked_preds).sort_values(by="refl_ids")

    # Overwriting columns
    ds_filtered["intensity.prf.value"] = pred_df["qi_mean"]
    ds_filtered["intensity.prf.variance"] = pred_df["qi_var"]
    ds_filtered["intensity.sum.value"] = pred_df["qi_mean"]
    ds_filtered["intensity.sum.variance"] = pred_df["qi_var"]
    ds_filtered["background.mean"] = pred_df["qbg_mean"]

    write_refl_from_ds(ds_filtered, fname)


def main():
    from pathlib import Path

    import torch
    import yaml

    from integrator.callbacks import PredWriter
    from integrator.utils import (
        construct_data_loader,
        construct_integrator,
        construct_trainer,
        load_config,
    )

    # Get args
    args = parse_args()

    # run dir
    run_dir = Path(args.run_dir)

    # Reading in pred.yaml
    meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    config = load_config(meta["config"])
    wandb_info = meta["wandb"]
    slurm_info = meta["slurm"]

    # writing prediction directories
    log_dir = Path(wandb_info["log_dir"])
    wandb_dir = log_dir.parent
    pred_dir = wandb_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # list of .ckpt files
    checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))

    # load data
    data_loader = construct_data_loader(config)
    data_loader.setup()

    # path to input refl file
    refl_file = config["output"]["refl_file"]

    epoch_re = re.compile(r"epoch=(\d+)")
    for ckpt in checkpoints:
        # Finding checkpoint epoch
        m = epoch_re.search(ckpt.name)
        if not m:
            raise ValueError(f"Could not parse epoch from {ckpt.name}")
        epoch = int(m.group(1))

        # Writing epoch prediction dir
        ckpt_dir = pred_dir / f"epoch_{epoch:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Construct callbacks
        callbacks = []
        pred_writer = PredWriter(
            output_dir=ckpt_dir,
            write_interval="epoch",
        )
        callbacks.append(pred_writer)

        # Construct trainer
        trainer = construct_trainer(
            config,
            callbacks=callbacks,
            logger=None,
        )

        ckpt_ = torch.load(ckpt.as_posix())
        integrator = construct_integrator(config)
        integrator.load_state_dict(ckpt_["state_dict"])

        # Use gpu if available
        if torch.cuda.is_available():
            integrator.to(torch.device("cuda"))

        integrator.eval()
        trainer.predict(
            integrator,
            return_predictions=False,
            dataloaders=data_loader.predict_dataloader(),
        )

        if args.write_refl:
            write_refl_from_preds(
                ckpt_dir=ckpt_dir,
                refl_file=refl_file,
                epoch=epoch,
            )

    print("Prediction complete!")


if __name__ == "__main__":
    main()
    #
    # import argparse
    # import re
    # from pathlib import Path
    #
    # import torch
    # import yaml
    #
    # from integrator.callbacks import PredWriter
    # from integrator.utils import (
    #     construct_data_loader,
    #     construct_integrator,
    #     construct_trainer,
    #     load_config,
    # )
    #
    # # Path to run-dir
    # run_dir = Path("/Users/luis/integrator/temp_run_dir/")
    #
    # # Reading in pred.yaml
    # meta = yaml.safe_load((run_dir / "run_metadata.yaml").read_text())
    # config = load_config(meta["config"])
    # wandb_info = meta["wandb"]
    # slurm_info = meta["slurm"]
    #
    # # writing prediction directories
    # log_dir = Path(wandb_info["log_dir"])
    # wandb_dir = log_dir.parent
    # pred_dir = wandb_dir / "predictions"
    # pred_dir.mkdir(exist_ok=True)
    #
    # # list of .ckpt files
    # checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
    #
    # # load data
    # data_loader = construct_data_loader(config)
    # data_loader.setup()
    #
    # refl_file = config["output"]["refl_file"]
    #
    # epoch_re = re.compile(r"epoch=(\d+)")
    # for ckpt in checkpoints:
    #     # Finding checkpoint epoch
    #     m = epoch_re.search(ckpt.name)
    #     if not m:
    #         raise ValueError(f"Could not parse epoch from {ckpt.name}")
    #     epoch = int(m.group(1))
    #
    #     # Writing epoch prediction dir
    #     ckpt_dir = pred_dir / f"epoch_{epoch:04d}"
    #     ckpt_dir.mkdir(parents=True, exist_ok=True)
    #
    #     # Construct callbacks
    #     callbacks = []
    #     pred_writer = PredWriter(
    #         output_dir=ckpt_dir,
    #         write_interval="epoch",
    #     )
    #     callbacks.append(pred_writer)
    #
    #     # Construct trainer
    #     trainer = construct_trainer(
    #         config,
    #         callbacks=callbacks,
    #         logger=None,
    #     )
    #
    #     ckpt_ = torch.load(ckpt.as_posix())
    #     integrator = construct_integrator(config)
    #     integrator.load_state_dict(ckpt_["state_dict"])
    #
    #     # Use gpu if available
    #     if torch.cuda.is_available():
    #         integrator.to(torch.device("cuda"))
    #
    #     integrator.eval()
    #     data = trainer.predict(
    #         integrator,
    #         return_predictions=True,
    #         dataloaders=data_loader.predict_dataloader(),
    #     )

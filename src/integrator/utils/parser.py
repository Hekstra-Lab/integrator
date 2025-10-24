import argparse


class BaseParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_base_args()

    def _add_base_args(self):
        """Default arguments shared across all scripts."""
        self.add_argument(
            "--root",
            type=str,
            default="/n/netscratch/hekstra_lab/Lab/laldama/lightning_logs/wandb/",
            help="Root directory for wandb logs.",
        )
        self.add_argument(
            "--wandb-id",
            type=str,
            help="WandB run ID",
        )

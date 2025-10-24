from pathlib import Path

import polars as pl
import pytextable

from integrator.utils import BaseParser


def main(args):
    # loading arguments
    root = Path(args.root)
    id = Path(args.wandb_id)
    config = args.config

    path = list((root / id).glob(f"*{id}"))[0]

    # Get peak data
    epochs = []
    dfs = []
    for c in list(path.glob(f"**/scaling/{config}/peaks*")):
        epochs.append(int(c.parents[2].name.split("_")[-1]))
        dfs.append(pl.read_csv(c))

    # make polars dataframe
    df = (
        pl.concat(dfs, how="vertical")
        .with_columns(
            pl.Series(
                "epoch",
                epochs,
            )
        )
        .sort("epoch")
    )

    df = pl.concat(dfs, how="vertical")
    df_e = pl.DataFrame({"epoch": epochs})

    df = pl.concat([df_e, df], how="horizontal")

    data = df.to_numpy()
    header = tuple(df.columns)

    # # save table as a LaTeX booktabs table
    pytextable.write(
        data, "table.text", header=header, caption=f"Peak heights for {config}"
    )


if __name__ == "__main__":
    argparser = BaseParser()

    argparser.add_argument(
        "--precog-path",
        type=str,
        default="/n/holylabs/hekstra_lab/Users/laldama/laue_analysis/precognition/",
    )
    argparser.add_argument(
        "--ld-path",
        type=str,
        default="/n/holylabs/hekstra_lab/Users/laldama/laue_analysis/laue-dials/",
    )
    argparser.add_argument("--config", type=str)
    args = argparser.parse_args()

    main(args)

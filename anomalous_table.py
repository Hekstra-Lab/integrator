from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import pytextable

from integrator.utils import BaseParser


def plot_peaks(
    df: pl.DataFrame,
):
    pass


def plotly_table(
    df: pl.DataFrame,
    config: str,
    id: str,
):
    # Create Plotly table for this method
    header = df.columns
    data = df.to_numpy().T

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header,
                    line_color="darkslategray",
                    fill_color="#CDCDCD",
                    align="center",
                    font=dict(color="black", size=12),
                ),
                cells=dict(
                    values=[r for r in data],  # Transpose for Plotly format
                    line_color="darkslategray",
                    fill_color=[["white", "#F3F3F3"] * df.height],
                    align="center",
                    font=dict(color="black", size=10),
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Peaks heights for {config} (id:{id})",
        margin=dict(l=20, r=20, t=60, b=20),
        width=None,  # Full width
        height=None,  # Auto height
    )
    return fig


def main(args):
    # setting arguments
    root = Path(args.root)
    id = Path(args.wandb_id)
    config = args.config

    path = list((root).glob(f"*{id}"))[0]

    save_dir = path / f"figures/{config}/peaks"
    save_dir.mkdir(parents=True, exist_ok=True)

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

    # constructing dataframe
    df = pl.concat(dfs, how="vertical")
    df_e = pl.DataFrame({"epoch": epochs})
    df = pl.concat([df_e, df], how="horizontal").sort("epoch")

    # save table as a LaTeX booktabs table
    data = df.to_numpy()
    header = tuple(df.columns)

    pytextable.write(
        data,
        f"{save_dir}/table_{id}_{config}.tex",
        header=header,
        caption=f"Peak heights for {config}",
    )

    # write out a plotly image
    fig = plotly_table(df, config, id.as_posix())
    fig.write_image(f"{save_dir}/peak_heights_{id}_{config}.png")


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

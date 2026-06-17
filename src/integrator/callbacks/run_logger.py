from __future__ import annotations

import json
from pathlib import Path

try:
    import wandb
except ImportError:
    wandb = None


def wandb_active() -> bool:
    """True when wandb is importable and a run has been initialised."""
    return wandb is not None and getattr(wandb, "run", None) is not None


def _to_scalar(v):
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def _slug(name: str) -> str:
    out = name.strip().lower()
    for ch in " /:()[],":
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "fig"


def _resolve_plot_dir(trainer) -> Path:
    """Best-effort local directory for plot/table artifacts from a trainer."""
    base = None
    lg = getattr(trainer, "logger", None)
    if lg is not None:
        base = getattr(lg, "log_dir", None) or getattr(lg, "save_dir", None)
        if base is None:
            exp = getattr(lg, "experiment", None)
            base = getattr(exp, "dir", None) if exp is not None else None
    if base is None:
        base = (
            getattr(trainer, "log_dir", None)
            or getattr(trainer, "default_root_dir", None)
            or "."
        )
    return Path(base) / "plots"


class RunLogger:
    """Always write local files; additionally log to W&B when a run is active."""

    def __init__(self, out_dir, use_wandb: bool | None = None):
        self.use_wandb = wandb_active() if use_wandb is None else use_wandb
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _suffix(self, step) -> str:
        return "" if step is None else f"_step{int(step):04d}"

    def log_scalars(self, metrics: dict, step=None) -> None:
        if not metrics:
            return
        clean = {k: _to_scalar(v) for k, v in metrics.items()}
        with open(self.out_dir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps({"step": step, **clean}, default=str) + "\n")
        if self.use_wandb:
            wandb.log(clean)

    def log_figure(
        self, name: str, fig, step=None, close: bool = True
    ) -> None:
        fig.savefig(
            self.out_dir / f"{_slug(name)}{self._suffix(step)}.png",
            bbox_inches="tight",
        )
        if self.use_wandb:
            wandb.log({name: wandb.Image(fig)})
        if close:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def log_table(self, name: str, df, step=None) -> None:
        df.write_csv(self.out_dir / f"{_slug(name)}{self._suffix(step)}.csv")
        if self.use_wandb:
            table = wandb.Table(data=df.rows(), columns=df.columns)
            wandb.log({name: table})

    def log_scatter(
        self,
        name: str,
        df,
        x: str,
        y: str,
        step=None,
        figsize=(3.0, 3.0),
        dpi: int = 80,
        loglog: bool = False,
    ) -> None:
        import matplotlib.pyplot as plt

        sub = df.select([x, y])
        xv = df[x].to_numpy()
        yv = df[y].to_numpy()
        # small, low-res figure on purpose: cheap to write every epoch
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        if loglog:
            mask = (xv > 0) & (yv > 0)
            xv, yv = xv[mask], yv[mask]
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.scatter(xv, yv, s=3, alpha=0.4)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(name, fontsize=8)
        slug = _slug(name)
        fig.savefig(
            self.out_dir / f"{slug}{self._suffix(step)}.png",
            bbox_inches="tight",
            dpi=dpi,
        )
        plt.close(fig)
        sub.write_csv(self.out_dir / f"{slug}{self._suffix(step)}.csv")
        if self.use_wandb:
            table = wandb.Table(data=sub.rows(), columns=[x, y])
            wandb.log({name: wandb.plot.scatter(table, x, y, title=name)})


def get_run_logger(obj, trainer) -> RunLogger:
    """Return a cached RunLogger for a callback.

    Uses the callback's explicit out_dir when set (e.g. run-dir/plots), else
    falls back to a directory derived from the trainer's logger.
    """
    rl = getattr(obj, "_run_logger", None)
    if rl is None:
        out_dir = getattr(obj, "out_dir", None) or _resolve_plot_dir(trainer)
        rl = RunLogger(out_dir)
        obj._run_logger = rl
    return rl

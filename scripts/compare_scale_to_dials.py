"""Compare a merging model's learned per-observation scale s_i to DIALS's scale.

Works for any merging integrator that exposes `_get_scale` (ConjugateMerging or
AmortizedMerging) - the model is built from the run's config, so just point it at
the run dir. Direct test of "is the scale the gap?": compute the model's per-obs
scale `s_i` (= `_get_scale`) over the dataset and correlate it (in log space,
which is gauge-invariant to a global factor) against DIALS's per-obs
`inverse_scale_factor` from a scaled.refl.

  * High |r| (e.g. > 0.9)  -> the model already learned DIALS's scale structure;
    the 29-vs-32 gap is NOT the scale -> look at integration / the ELBO objective.
  * Low |r|, AND the regression showed DIALS's scale is learnable with more
    capacity -> the model's 2-layer scale is under-fit; bump scale_mlp_layers/hidden.
  * Also reports a per-image-detrended |r|: the within-image (fine / absorption-
    like) structure that the bulk rotation trend hides, and that anomalous needs.

Matching: by `refl_ids` if present in the scaled.refl, else by `xyzcal.px`
(rounded) - dials.scale drops the custom `refl_ids` column, but xyzcal.px is
shared with the integrated.refl the model was built from.

Usage (cluster, refltorch env):
    python scripts/compare_scale_to_dials.py RUN_DIR SCALED.refl [--checkpoint CKPT]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_merging import (  # noqa: E402
    find_last_checkpoint,
    load_integrator,
    load_run_metadata,
)

from integrator.utils import construct_data_loader  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_RND = 10.0  # match xyzcal.px at 0.1-px resolution


@torch.no_grad()
def model_scales(integrator, loader, device) -> pd.DataFrame:
    """Per-observation s_i + geometry (+ refl_ids if available)."""
    integrator.eval()
    cols: dict[str, list] = {
        "s": [], "x": [], "y": [], "frame": [], "lp": [], "refl_ids": []
    }
    have_ids = True
    for batch in loader:
        _, _, _, metadata = batch
        s = integrator._get_scale(metadata, device)
        cols["s"].append(s.detach().cpu().numpy())
        cols["x"].append(metadata["xyzcal.px.0"].cpu().numpy())
        cols["y"].append(metadata["xyzcal.px.1"].cpu().numpy())
        cols["frame"].append(metadata["xyzcal.px.2"].cpu().numpy())
        cols["lp"].append(metadata["lp"].cpu().numpy())
        rid = metadata.get("refl_ids")
        if rid is None:
            have_ids = False
        else:
            cols["refl_ids"].append(rid.cpu().numpy())
    out = {
        "s": np.concatenate(cols["s"]),
        "x": np.concatenate(cols["x"]),
        "y": np.concatenate(cols["y"]),
        "frame": np.concatenate(cols["frame"]),
        "lp": np.concatenate(cols["lp"]),
    }
    if have_ids:
        out["refl_ids"] = np.concatenate(cols["refl_ids"]).astype(np.int64)
    logger.info("Model: %d observations (refl_ids=%s)", len(out["s"]), have_ids)
    return pd.DataFrame(out)


def read_dials(path: str) -> pd.DataFrame:
    want = ["xyzcal.px", "inverse_scale_factor", "refl_ids"]
    for cols in (want, ["xyzcal.px", "inverse_scale_factor"]):
        try:
            ds = rs.io.read_dials_stills(path, extra_cols=cols)
            break
        except Exception as e:  # noqa: BLE001
            logger.warning("read with %s failed: %s", cols, e)
    else:
        raise RuntimeError(f"Could not read {path}")
    df = pd.DataFrame({
        "scale": ds["inverse_scale_factor"].to_numpy(),
        "x": ds["xyzcal.px.0"].to_numpy(),
        "y": ds["xyzcal.px.1"].to_numpy(),
        "frame": ds["xyzcal.px.2"].to_numpy(),
    })
    if "refl_ids" in ds.columns:
        df["refl_ids"] = ds["refl_ids"].to_numpy().astype(np.int64)
    logger.info("DIALS: %d rows (refl_ids=%s)", len(df), "refl_ids" in df)
    return df


def _geo_key(df: pd.DataFrame) -> np.ndarray:
    return (
        np.round(df["x"].to_numpy() * _RND).astype(np.int64) * 10_000_000_000
        + np.round(df["y"].to_numpy() * _RND).astype(np.int64) * 100_000
        + np.round(df["frame"].to_numpy() * _RND).astype(np.int64)
    )


def match(model_df: pd.DataFrame, dials_df: pd.DataFrame) -> pd.DataFrame:
    if "refl_ids" in model_df and "refl_ids" in dials_df:
        logger.info("Matching by refl_ids")
        return model_df.merge(
            dials_df[["refl_ids", "scale"]], on="refl_ids", how="inner"
        )
    logger.info("Matching by xyzcal.px geometry (0.1 px)")
    model_df = model_df.assign(_k=_geo_key(model_df))
    dials_df = dials_df.assign(_k=_geo_key(dials_df))
    dials_u = dials_df.drop_duplicates("_k", keep=False)  # drop collisions
    dropped = len(dials_df) - len(dials_u)
    if dropped:
        logger.warning("Dropped %d DIALS rows with colliding geometry keys", dropped)
    return model_df.merge(dials_u[["_k", "scale"]], on="_k", how="inner")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(
        description="Correlate the model's per-obs scale with DIALS's scale."
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("scaled_refl")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()

    cfg, meta = load_run_metadata(args.run_dir.resolve())
    ckpt = args.checkpoint or find_last_checkpoint(meta)
    integrator = load_integrator(cfg, ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrator.to(device)
    logger.info("Loaded %s; checkpoint %s", type(integrator).__name__, ckpt)

    dl = construct_data_loader(cfg)
    dl.setup()
    model_df = model_scales(integrator, dl.predict_dataloader(), device)
    dials_df = read_dials(args.scaled_refl)

    m = match(model_df, dials_df)
    m = m[(m["s"] > 0) & (m["scale"] > 0)]
    logger.info("Matched %d observations", len(m))
    if len(m) < 100:
        logger.error("Too few matches (%d) - check geometry/ids alignment", len(m))
        return

    ls_dials = np.log(m["scale"].to_numpy())
    fr = np.round(m["frame"].to_numpy()).astype(np.int64)
    # MLPScale folds LP into s_i; DIALS's inverse_scale_factor excludes it, so
    # log(s) - log(lp) is the fair comparison of the *fitted* scale structure.
    variants = {
        "raw  log(s)            ": np.log(m["s"].to_numpy()),
        "LP-removed log(s/lp)   ": np.log(m["s"].to_numpy()) - np.log(
            m["lp"].to_numpy().clip(min=1e-8)
        ),
    }

    print("\n=== model s_i  vs  DIALS inverse_scale_factor (log space) ===")
    print(f"matched observations : {len(m)}")
    for name, lm in variants.items():
        r = _corr(lm, ls_dials)
        slope = np.polyfit(ls_dials, lm, 1)[0]
        d = pd.DataFrame({"fr": fr, "lm": lm, "ld": ls_dials})
        d["lm"] = d["lm"] - d.groupby("fr")["lm"].transform("mean")
        d["ld"] = d["ld"] - d.groupby("fr")["ld"].transform("mean")
        r_det = _corr(d["lm"].to_numpy(), d["ld"].to_numpy())
        print(f"{name}: r_global={r:+.4f}  slope={slope:+.3f}  "
              f"r_detrended={r_det:+.4f}")
    print(
        "\n(LP-removed is the fair comparison.) High global & detrended |r| "
        "(>0.9) -> the scale already matches DIALS; the gap is integration/"
        "objective, not the scale. Low |r|, esp. detrended -> the 2-layer scale "
        "is under-fit; bump scale_mlp_layers/hidden and re-check.\n"
        "slope < 1 -> the model's scale spans a narrower range than DIALS's."
    )


if __name__ == "__main__":
    main()

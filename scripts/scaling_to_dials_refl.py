"""Write a scaling model's PER-OBSERVATION intensities to a DIALS .refl.

Companion to `diagnose_merging.py`, but instead of letting the model do the
scaling + merge and writing a merged MTZ, this writes per-observation
intensities into a reflection table so the **standard DIALS pipeline** does the
scaling/merging:

    scaling model -> per-obs intensities (this script) -> dials.scale
                  -> dials.merge -> phenix.refine + rs.find_peaks

Two intensity sources (`--intensity-source`), to compare experimentally what
DIALS scaling can recover:

  profile-fit (default): integrate the RAW counts with the model's profile +
    background, per observation, WITHOUT the learned scale:
        I_prf = sum_p prf_p (counts_p - bg) / sum_p prf_p^2
    These are independent per-observation MEASUREMENTS (real photon scatter
    preserved), the faithful analogue of "neural integrator + DIALS scaling".
    DIALS fits its OWN scale (incl. absorption) -> tests whether DIALS scaling
    reaches the integrator's peak heights.

  model-mc: the model's per-observation reconstructed intensity s_i * I_h, via
    Monte-Carlo over draws of I ~ q(I_h) and prf ~ q(prf):
        I_i^(s) = scale_i * I_h^(s) * sum_p mask_p prf_p^(s)
    mean/var over S draws. NOTE: within an HKL these differ only by the model's
    own scale s_i, so dials.scale tends to re-derive s_i and hand back I_h
    (reproducing the model's merge). It tests the MERGE given the model's scale,
    not whether DIALS's scale beats the model's. Included so the contrast with
    profile-fit can be shown empirically. Requires group_by_asu_id (the per-HKL
    fixed point needs complete groups per batch).

Usage:
    uv run python scripts/scaling_to_dials_refl.py RUN_DIR \
        [--checkpoint CKPT] [--intensity-source {profile-fit,model-mc,both}] \
        [--mc-samples 100] [--anomalous]

Then (on the cluster, in the DIALS env) run the commands it prints.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Reuse the run/checkpoint loaders from the sibling diagnostic (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_merging import (  # noqa: E402
    find_last_checkpoint,
    load_integrator,
    load_run_metadata,
)

from integrator.cli.utils.io import _read_reference_refl  # noqa: E402
from integrator.model.integrators.hierarchical_integrator import (  # noqa: E402
    _get_normalized_position,
    _sample_profile,
)
from integrator.utils import construct_data_loader, load_config  # noqa: E402
from integrator.utils.refl_utils import write_refl_from_ds  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _empty_cols() -> dict[str, list]:
    return {
        "refl_ids": [],
        "intensity.prf.value": [],
        "intensity.prf.variance": [],
        "intensity.sum.value": [],
        "intensity.sum.variance": [],
        "background.mean": [],
    }


@torch.no_grad()
def predict_profilefit(integrator, dataloader, device) -> dict:
    """Profile-fit + summation per-observation intensities from RAW counts.

    Uses the model's profile (`qp.mean_profile`) and background (`qbg.mean`) but
    NOT its learned scale, so the result is the measured per-observation
    intensity DIALS expects to scale. Independent per observation.
    """
    integrator.eval()
    out = _empty_cols()
    n_neg = n_total = 0
    for batch in dataloader:
        counts, shoebox, mask, metadata = batch
        counts = counts.clamp(min=0).to(device).float()
        shoebox = shoebox.to(device)
        mask = mask.to(device).float()
        b = shoebox.shape[0]
        sr = (shoebox * mask).reshape(b, 1, *integrator.shoebox_shape)

        position = _get_normalized_position(metadata, device)
        qbg = integrator.surrogates["qbg"](
            integrator.encoders["k_bg"](sr), integrator.encoders["r_bg"](sr)
        )
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = integrator.surrogates["qp"](
            integrator.encoders["profile"](sr, position=position),
            mc_samples=1,
            group_labels=prf_labels,
            metadata=metadata,
        )

        prof = qp.mean_profile  # (B, P)
        bg = qbg.mean  # (B,)
        bg_var = getattr(qbg, "variance", None)
        if bg_var is None:
            bg_var = torch.zeros_like(bg)

        cmb = (counts - bg[:, None]) * mask
        pm = prof * mask
        den = (pm * pm).sum(dim=-1).clamp(min=1e-10)
        i_prf = (pm * cmb).sum(dim=-1) / den
        var_prf = (
            (pm * pm * (counts + bg_var[:, None]) * mask).sum(dim=-1)
            + (pm.sum(dim=-1)) ** 2 * bg_var
        ) / den.pow(2)

        n_fg = mask.sum(dim=-1)
        i_sum = cmb.sum(dim=-1)
        var_sum = (counts * mask).sum(dim=-1) + n_fg.pow(2) * bg_var

        out["refl_ids"].append(metadata["refl_ids"].long().cpu().numpy())
        out["intensity.prf.value"].append(i_prf.cpu().numpy())
        out["intensity.prf.variance"].append(var_prf.cpu().numpy())
        out["intensity.sum.value"].append(i_sum.cpu().numpy())
        out["intensity.sum.variance"].append(var_sum.cpu().numpy())
        out["background.mean"].append(bg.cpu().numpy())
        n_neg += int((i_prf < 0).sum())
        n_total += b

    merged = {k: np.concatenate(v) for k, v in out.items()}
    logger.info(
        "[profile-fit] %d obs; I_prf median=%.3g mean=%.3g; %.1f%% negative",
        n_total,
        float(np.median(merged["intensity.prf.value"])),
        float(np.mean(merged["intensity.prf.value"])),
        100.0 * n_neg / max(n_total, 1),
    )
    return merged


@torch.no_grad()
def predict_modelmc(integrator, dataloader, device, n_mc: int) -> dict:
    """Per-observation s_i * I_h via MC over draws of I ~ q(I_h), prf ~ q(prf).

    Runs the full forward (the per-HKL conjugate merge -> qi per obs), then
    Monte-Carlos the per-observation reconstructed intensity
    I_i = scale_i * I_h * sum_p mask_p prf_p. Requires complete HKL groups per
    batch (grouped loader). Writes the same value to the prf and sum columns.
    """
    integrator.eval()
    out = _empty_cols()
    n_total = 0
    for batch in dataloader:
        counts, shoebox, mask, metadata = batch
        counts = counts.clamp(min=0).to(device).float()
        shoebox = shoebox.to(device)
        mask = mask.to(device).float()

        fwd = integrator(counts, shoebox, mask, metadata)
        qi = fwd["qi"]  # Gamma(alpha_h[inverse], beta_h[inverse]) over obs
        qp = fwd["qp"]
        scale = integrator._get_scale(metadata, device)  # (B,)
        bg = fwd["qbg"].mean  # (B,)

        i_s = qi.rsample([n_mc]).clamp(min=0.0).permute(1, 0)  # (B, S)
        prf_s = _sample_profile(qp, n_mc)  # (B, S, P)
        prf_mass = (prf_s * mask[:, None, :]).sum(dim=-1)  # (B, S)
        inten = scale[:, None] * i_s * prf_mass  # (B, S)
        i_mean = inten.mean(dim=1)
        i_var = inten.var(dim=1).clamp(min=1e-12)

        out["refl_ids"].append(metadata["refl_ids"].long().cpu().numpy())
        out["intensity.prf.value"].append(i_mean.cpu().numpy())
        out["intensity.prf.variance"].append(i_var.cpu().numpy())
        out["intensity.sum.value"].append(i_mean.cpu().numpy())
        out["intensity.sum.variance"].append(i_var.cpu().numpy())
        out["background.mean"].append(bg.cpu().numpy())
        n_total += counts.shape[0]

    merged = {k: np.concatenate(v) for k, v in out.items()}
    logger.info(
        "[model-mc] %d obs (S=%d); I median=%.3g mean=%.3g",
        n_total,
        n_mc,
        float(np.median(merged["intensity.prf.value"])),
        float(np.mean(merged["intensity.prf.value"])),
    )
    return merged


def write_refl(preds: dict, refl_file: Path, data_dir: Path, out_path: Path):
    """Overwrite the original .refl's intensity columns with model per-obs values.

    Mirrors `io.write_refl_from_preds`: read the reference .refl, keep the rows
    the model predicted on (match by `refl_ids`), overwrite the intensity +
    background columns, write a new .refl with the dataset identifiers. All
    geometry (s1, xyzcal, bbox, miller_index, ...) is carried over unchanged.
    """
    ds = _read_reference_refl(str(refl_file))
    pred_ids = set(preds["refl_ids"].tolist())
    ds_f = ds[ds["refl_ids"].isin(pred_ids)].sort_values("refl_ids")
    ds_f = ds_f.reset_index(drop=True)

    pred_df = pd.DataFrame(preds)
    pred_df = pred_df[pred_df["refl_ids"].isin(set(ds_f["refl_ids"]))]
    pred_df = pred_df.sort_values("refl_ids").reset_index(drop=True)

    if len(ds_f) != len(pred_df) or not np.array_equal(
        ds_f["refl_ids"].to_numpy(), pred_df["refl_ids"].to_numpy()
    ):
        raise RuntimeError(
            f"refl_ids did not align: {len(ds_f)} refl rows vs "
            f"{len(pred_df)} predictions after intersection/sort."
        )

    for col in (
        "intensity.prf.value",
        "intensity.prf.variance",
        "intensity.sum.value",
        "intensity.sum.variance",
        "background.mean",
    ):
        ds_f[col] = pred_df[col].to_numpy()

    identifiers_path = data_dir / "identifiers.yaml"
    if not identifiers_path.exists():
        raise RuntimeError(f"Missing identifiers.yaml at {identifiers_path}")
    identifiers = load_config(identifiers_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_refl_from_ds(ds_f, str(out_path), identifiers=identifiers)
    logger.info("Wrote %s: %d reflections", out_path, len(ds_f))


def _print_dials_commands(
    out_refl: Path,
    expt_file: str,
    phenix_eff: str,
    pdb_file: str,
    anomalous: bool,
):
    anom = "anomalous=True " if anomalous else ""
    stem = out_refl.stem
    scaled_refl = out_refl.with_name(f"{stem}_scaled.refl")
    scaled_expt = out_refl.with_name(f"{stem}_scaled.expt")
    merged_mtz = out_refl.with_name(f"{stem}_merged.mtz")
    refine_dir = out_refl.with_name(f"{stem}_phenix")
    print(f"""
# ---- {stem}: run in the DIALS/PHENIX env (cluster) ----
cd {out_refl.parent}
dials.scale {out_refl} {expt_file} {anom}\\
    output.reflections={scaled_refl} output.experiments={scaled_expt}
dials.merge {scaled_refl} {scaled_expt} {anom}output.mtz={merged_mtz}

mkdir -p {refine_dir} && cd {refine_dir}
phenix.refine {phenix_eff} \\
    refinement.input.xray_data.file_name={merged_mtz} \\
    refinement.input.pdb.file_name={pdb_file} \\
    refinement.output.prefix=refined \\
    --overwrite
rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p PANOM -z 5.0 -o peaks.csv
""")


def main():
    parser = argparse.ArgumentParser(
        description="Write a scaling model's per-observation intensities to a "
        "DIALS .refl for the standard dials.scale -> merge -> phenix pipeline."
    )
    parser.add_argument("run_dir", type=Path, help="Training run directory")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--intensity-source",
        choices=["profile-fit", "model-mc", "both"],
        default="both",
        help="profile-fit = integrate raw counts (DIALS fits its own scale); "
        "model-mc = per-obs s*I_h via MC (DIALS re-derives the model's scale).",
    )
    parser.add_argument("--mc-samples", type=int, default=100)
    parser.add_argument(
        "--anomalous", action="store_true",
        help="Print dials.scale/merge commands with anomalous=True (keep I+/I-)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    cfg, meta = load_run_metadata(run_dir)
    ckpt = args.checkpoint or find_last_checkpoint(meta)
    logger.info("Run dir:   %s", run_dir)
    logger.info("Checkpoint: %s", ckpt)

    integrator = load_integrator(cfg, ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrator.to(device)
    logger.info("Loaded %s on %s", type(integrator).__name__, device)

    data_loader = construct_data_loader(cfg)
    data_loader.setup()

    sources = (
        ["profile-fit", "model-mc"]
        if args.intensity_source == "both"
        else [args.intensity_source]
    )
    diag = run_dir / "diagnostics"
    data_dir = Path(cfg["data_loader"]["args"]["data_dir"])
    refl_file = Path(cfg["output"]["refl_file"])

    for src in sources:
        if src == "profile-fit":
            loader = data_loader.predict_dataloader()  # ungrouped is fine
            preds = predict_profilefit(integrator, loader, device)
            out_path = diag / "per_obs_profilefit.refl"
        else:  # model-mc needs complete HKL groups per batch
            try:
                loader = data_loader.predict_dataloader(grouped=True)
            except TypeError:
                loader = None
            if loader is None or not getattr(
                data_loader, "group_by_asu_id", False
            ):
                logger.error(
                    "model-mc requires a grouped (group_by_asu_id) loader; "
                    "skipping. Set group_by_asu_id: true in the data_loader."
                )
                continue
            preds = predict_modelmc(integrator, loader, device, args.mc_samples)
            out_path = diag / "per_obs_modelmc.refl"

        write_refl(preds, refl_file, data_dir, out_path)
        _print_dials_commands(
            out_path,
            cfg["output"].get("expt_file", "<integrated.expt>"),
            cfg["output"].get("phenix_eff", "<phenix.eff>"),
            cfg["output"].get("pdb", "<model.pdb>"),
            args.anomalous,
        )


if __name__ == "__main__":
    main()

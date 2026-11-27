"""Collect careless -> phenix -> peaks results for several integrators and careless seeds on a truth simulation.

Each arm is given as NAME=UNMERGED_MTZ, the file `run_pipeline.py --mtz` scaled.
Every merge of it sits beside that file: `scaling_seed<N>/` from
`--careless-seed N`, and `scaling/` from the pipeline config's own seed
(`--default-seed`, 1 in sim_mos010_pipeline_cfg.yaml). Per arm and seed:

    r_work, r_free     the final cycle of refine2 (`integrator.utils.metrics.refinement_from_phenix_log`)
    peaks              anomalous peak heights at the model's S and Mn sites (`peak_heights.csv`,
                       `metrics.peak_heights_from_csv`)
    merged vs truth    `score_merged_vs_truth.py` on the merged MTZ against the simulation's
                       ground_truth.mtz (cc_F, cc_F_kB, r_F, cc_I, cc_dano per shell and overall)
    careless stats     the pipeline's long `careless_stats.csv` (CChalf, CCanom, CCpred, ...)
    refinement shells  phenix's final per-shell tables (R-work, R-free, completeness, ML targets;
                       FOM, phase error, sigma-A alpha and beta), `metrics.refinement_shells_from_phenix_log`
    refinement steps   R-work, R-free, bond and angle rmsd and B per macro-cycle stage
    CC_work, CC_free   corr(F_obs^2, F_model^2) of refined_2.mtz on the work and the free set, on
                       phenix's own shell edges (Karplus & Diederichs 2012 compare these with
                       CC* = sqrt(2 CC_1/2 / (1 + CC_1/2)), computed in the figure from careless's CC_1/2)

Writes into --out-dir, each with `method`, `dataset`, `seed` key columns:
`downstream_refine.csv` (one row per arm and seed), `downstream_refine_shells.csv`,
`downstream_refine_steps.csv`, `downstream_ccwork.csv`, `downstream_peaks.csv`
(one row per site), `downstream_merged.csv` (per shell, `shell = all` the whole
domain) and `downstream_careless_stats.csv`. Missing pieces are reported and skipped.

usage: python collect_downstream.py --arm NAME=MTZ [--arm ...] --truth ground_truth.mtz
           --dataset mos010_pc_default --dmin 1.78 --out-dir DIR [--default-seed 1]

On real data, where there is no truth, leave out --truth; `--refine-tag d1.81` reads a
refinement written by `run_pipeline.py --refine-tag d1.81` instead of the full-range one.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

from integrator.utils import metrics

HERE = Path(__file__).resolve().parent


def merges(mtz: Path, default_seed: int) -> list[tuple[int, Path]]:
    out = []
    plain = mtz.parent / "scaling"
    if (plain / "config4_0.mtz").is_file():
        out.append((default_seed, plain))
    for d in sorted(mtz.parent.glob("scaling_seed*")):
        m = re.fullmatch(r"scaling_seed(\d+)", d.name)
        if m and (d / "config4_0.mtz").is_file():
            out.append((int(m.group(1)), d))
    return out


def cc_work_free(mtz: Path, shells: list[dict]) -> list[dict]:
    """corr(F_obs^2, F_model^2) per shell on the work (flag 0) and free (flag 1) reflections, Friedel mates as rows."""
    import numpy as np
    import reciprocalspaceship as rs

    m = rs.read_mtz(str(mtz))
    d = m.compute_dHKL()["dHKL"].to_numpy()
    rows = []
    obs, mod, flg, dd = [], [], [], []
    for sgn in ("+", "-"):
        fo, fm, fl = m[f"F-obs-filtered({sgn})"], m[f"F-model({sgn})"], m[f"R-free-flags({sgn})"]
        ok = fo.notna().to_numpy() & fm.notna().to_numpy() & fl.notna().to_numpy()
        obs.append(fo.to_numpy(dtype=float)[ok]), mod.append(fm.to_numpy(dtype=float)[ok])
        flg.append(fl.to_numpy(dtype=float)[ok]), dd.append(d[ok])
    obs, mod, flg, dd = (np.concatenate(x) for x in (obs, mod, flg, dd))
    for sh in shells:
        sel = (dd <= sh["d_max"] + 1e-3) & (dd >= sh["d_min"] - 1e-3)
        for name, fv in (("work", 0), ("free", 1)):
            s = sel & (flg == fv)
            cc = float(np.corrcoef(obs[s] ** 2, mod[s] ** 2)[0, 1]) if s.sum() > 2 else float("nan")
            rows.append({"shell": sh["shell"], "d_max": sh["d_max"], "d_min": sh["d_min"], "set": name,
                         "n": int(s.sum()), "cc_I": cc})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=UNMERGED_MTZ")
    ap.add_argument("--truth", type=Path, default=None,
                    help="the simulation's ground_truth.mtz. Omitted on real data: every table but "
                         "downstream_merged.csv is still written")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dmin", type=float, default=1.78)
    ap.add_argument("--default-seed", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--refine-tag", default=None,
                    help="read refinement and peaks from config4_refine_<TAG> (run_pipeline.py --refine-tag), "
                         "e.g. d1.81 for a capped resolution. Default config4_refine")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tmp = a.out_dir / "merged_parts"
    tmp.mkdir(exist_ok=True)
    ref, peaks, merged, cstats, shells_rows, steps_rows, cc_rows = [], [], [], [], [], [], []
    for spec in a.arm:
        name, mtz = spec.split("=", 1)
        mtz = Path(mtz)
        found = merges(mtz, a.default_seed)
        if not found:
            print(f"{name}: no merge beside {mtz}")
        for seed, d in found:
            key = dict(method=name, dataset=a.dataset, seed=seed)
            work = d / (f"config4_refine_{a.refine_tag}" if a.refine_tag else "config4_refine")
            r = metrics.refinement_from_phenix_log(sorted((work / "refine2").glob("*.log")))
            if r is None:
                print(f"{name} seed {seed}: no refine2 result")
            else:
                ref.append({**key, **r})
            logs = sorted((work / "refine2").glob("*.log"))
            if logs:
                tabs = metrics.refinement_shells_from_phenix_log(logs[-1])
                fom = {r["shell"]: r for r in tabs["fom"]}
                for r in tabs["rfactors"]:
                    f = fom.get(r["shell"], {})
                    shells_rows.append({**key, **r, **{k: f.get(k) for k in ("fom", "phase_error", "alpha", "beta")}})
                for i, st in enumerate(metrics.refinement_steps_from_phenix_log(logs[-1])):
                    steps_rows.append({**key, "step": i, **st})
                mtz2 = next(iter(sorted((work / "refine2").glob("*_2.mtz"))), None)
                if mtz2 is not None and tabs["rfactors"]:
                    cc_rows += [{**key, **r} for r in cc_work_free(mtz2, tabs["rfactors"])]
            ph = work / "peak_heights.csv"
            model = next(iter(sorted((work / "refine2").glob("*.pdb"))), None)
            if ph.is_file():
                for p in metrics.peak_heights_from_csv(ph, model):
                    peaks.append({**key, "site": p["site"], "residue": p["residue"], "seqid": p["seqid"],
                                  "element": p["element"], "peakz": p["peakz"]})
            cs = d / "stats" / "careless_stats.csv"
            if cs.is_file():
                cstats.append(pl.read_csv(cs, infer_schema_length=None).with_columns(
                    method=pl.lit(name), seed=pl.lit(seed)))
            if a.truth is None:
                print(f"{name} seed {seed}: {r}")
                continue
            part = tmp / f"{name}_s{seed}.csv"
            part.unlink(missing_ok=True)
            res = subprocess.run([sys.executable, str(HERE / "score_merged_vs_truth.py"), "--merged", str(d / "config4_0.mtz"),
                                  "--truth", str(a.truth), "--method", name, "--dataset", a.dataset,
                                  "--dmin", str(a.dmin), "--out", str(part)], capture_output=True, text=True)
            if res.returncode != 0 or not part.is_file():
                print(f"{name} seed {seed}: score_merged_vs_truth failed\n{res.stderr[-800:]}")
            else:
                merged.append(pl.read_csv(part, infer_schema_length=None).with_columns(seed=pl.lit(seed)))
            print(f"{name} seed {seed}: {r}")
    if ref:
        pl.DataFrame(ref).write_csv(a.out_dir / "downstream_refine.csv")
    for rows_, name in ((shells_rows, "downstream_refine_shells.csv"), (steps_rows, "downstream_refine_steps.csv"),
                        (cc_rows, "downstream_ccwork.csv")):
        if rows_:
            pl.DataFrame(rows_).write_csv(a.out_dir / name)
    if peaks:
        pl.DataFrame(peaks).write_csv(a.out_dir / "downstream_peaks.csv")
    if merged:
        pl.concat(merged, how="diagonal_relaxed").write_csv(a.out_dir / "downstream_merged.csv")
    if cstats:
        pl.concat(cstats, how="diagonal_relaxed").write_csv(a.out_dir / "downstream_careless_stats.csv")
    if ref:
        with pl.Config(tbl_rows=40, float_precision=4):
            print(pl.DataFrame(ref).group_by("method").agg(
                pl.len().alias("n"), pl.col("r_free").mean().alias("r_free"), pl.col("r_free").std().alias("sd_r_free"),
                pl.col("r_work").mean().alias("r_work")).sort("r_free"))
    print(f"wrote {a.out_dir}")


if __name__ == "__main__":
    main()

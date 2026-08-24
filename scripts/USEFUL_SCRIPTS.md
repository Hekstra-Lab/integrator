# Useful scripts

NB likelihood ablation + figure pipeline. Cluster: FASRC `cannon`. Env: micromamba `integrator-cuda-dev`. Run root (cluster): `/n/holylabs/hekstra_lab/Users/laldama/nb_ablation`.

## NB dispersion screen (train)

| script | purpose |
|---|---|
| `scripts/ablation_likelihood/dispersion_screen.slurm` | SLURM array. One arm per task: poisson, `nb_r{0.5..100}` (fixed `r`, `nb_learn_dispersion=false`), `nb_learned`. Patches base cfg per arm, trains. |
| `scripts/ablation_likelihood/make_screen_config.py` | Patch `configs/ablation_likelihood/hierarchical_nbinom.yaml` → one arm cfg (`--likelihood`, `--dispersion`, `--scope`, `--learn-dispersion`). Called by the slurm. |

Submit: `INTEGRATOR_ROOT=<repo> sbatch scripts/ablation_likelihood/dispersion_screen.slurm`
Env knobs: `EPOCHS`, `WB_PROJECT`, `DISPERSIONS` (edit `--array` to match), `SCOPE`, `DATA_DIR`, `OUT`.
Out: `<OUT>/<tag>/` (default `./nb_screen/<tag>/`). Resubmit resumes per arm from `last.ckpt` (non-W&B).

## NB dispersion screen (analyze)

| script | purpose |
|---|---|
| `scripts/ablation_likelihood/plot_dispersion_screen.py` | Aggregate arms → `dispersion_screen.png` (held-out NLL + ELBO vs `r`) + `dispersion_screen_summary.csv`. |

Run: `uv run python scripts/ablation_likelihood/plot_dispersion_screen.py --runs-dir <OUT>/nb_screen`
In: per arm `plots/loss_history.csv`; learned `r` from checkpoint `raw_dispersion`.
Metric: held-out reconstruction NLL, comparable across Poisson/NB, lower better.

## Downstream (predict → DIALS/phenix → peaks)

| script | purpose |
|---|---|
| `nb_ablation/postprocess_arms.sh` (cluster) | SLURM GPU array. `train.sh` minus train, over already-trained arms: `integrator.predict --write-refl` → `post_config.py` → `submit_jobs.py` (fans out DIALS/phenix). |

Submit: `sbatch postprocess_arms.sh` (arms `poisson nb_learned`; keep `ARMS`+`--array` in sync with cfg).
Env: `NB`, `SCRIPTS`, `PROCESS_CFG`.
Produces per arm: `predictions/` (parquet, `.refl`), `merged.html`, `peaks.csv`, `refine_*.log`.

## Figures (multi-model comparison)

| script | in | out |
|---|---|---|
| `scripts/make_figures.sh` (repo) / `nb_ablation/make_figures.sh` (cluster) | driver; one `--plot-cfg` → all 4 below | `--out-dir` |
| `plot_loss.py` | `loss_history.csv` | `loss_<label>.png`, `loss_compare_<term>.png` |
| `plot_compare.py` | pred parquet (`qi_mean`,`qbg_mean`), DIALS `.refl` | `compare_*`, `correlation_*` |
| `plot_merging.py` | `merged.html` | `merging_<label>[_ccanom/cchalf/isigi/rpim].png` |
| `plot_peaks.py` | `peaks.csv`, `refine_*.log` | `<RES>_<seqid>.png`, `refinement*.png` |
| `plot_profiles.py` | checkpoint + data (single-model) | `profiles_*.png` |
| `plot_basis.py` | checkpoint tensors (single-model) | `learned_basis.png` |

Driver: `./make_figures.sh --plot-cfg <cfg> --out-dir figures [--epoch N] [--with-single]`. RUNNER default `python` (cluster) / `uv run python` (repo). Missing input → step warns + skips.
`loss` needs only training. `compare/merging/peaks` need predictions (run `postprocess_arms.sh` first).
`plot_i.py`: scratch, hardcoded paths, not wired.

## Plot cfg schema

`--plot-cfg` file (schema `plot_peaks.py:parse_plot_cfg`):
```yaml
runs:
  <name>: {path: <run-dir with run_paths.yaml>, label: <display/slug>}
reference_data:            # optional
  refl: <DIALS .refl>      # plot_compare
  merge: <DIALS merged.html>  # plot_merging
  peaks: <reference_peaks.csv>  # plot_peaks
  refinement: <phenix refine log>  # plot_peaks
```
Examples: `scripts/plot_cfg.example.yaml` (repo), `nb_ablation/nb_screen_cfg.yaml` (cluster; poisson vs nb_learned).

## Notes

- All cluster paths on `/n/lab_storage` post-migration (was `/n/hekstra_lab`). Do not `mv` micromamba envs; recreate from `environment-*.yml`.
- `integrator.train`/`integrator.predict` are console scripts; use `python -m integrator.cli.{train,predict}` if a stale shebang breaks after env moves.

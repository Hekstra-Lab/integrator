# Installations on Cannon

Which version of each external program a result was made with, and where it lives.
Inventoried 2026-11-26 from the checkouts' `git log` and each environment's package metadata.
Update this file when a checkout moves or an environment is rebuilt, and name the checkout or environment in every job that runs one.

## laue-dials

Four checkouts.
Which one a job imports is set by the environment's editable install, or by `LAUE_DIALS_SRC` (a checkout's `src/` first on `PYTHONPATH`) in `scripts/sim_truth/paper_sims/sim_laue_process.sbatch` and `scripts/poly/laue_dials_rerun.sbatch`.

| checkout | commit, date | branch | relative to upstream | imported by |
|---|---|---|---|---|
| `/n/holylabs/hekstra_lab/Users/laldama/software/laue-dials` | 63d483b, 2025-08-15 | `dump-profiles` | v0.5.1 plus local commits; the old summation `laue.integrate` | env `ld` (editable) |
| `/n/lab_storage/hekstra_lab/people/aldama/software/laue-dials` | 31d6f10, 2026-06-18 | `integrator` | v0.5.1-35; the new integrator, before the multi-panel and window-centring fixes | env `integrator-cuda-dev` (editable) |
| `/n/lab_storage/hekstra_lab/people/aldama/software/laue-dials-eiger` | 5de60e4, 2026-11-21 | `eiger-row-size` | v0.5.1-36; fixes predict's mask indexing on a non-square panel | `LAUE_DIALS_SRC`, the Eiger runs |
| `/n/lab_storage/hekstra_lab/people/aldama/software/laue-dials-main` | d57f96e, 2026-09-20 | `datsimx-sim` (local name) | **upstream `main` exactly**, clean | `LAUE_DIALS_SRC`, the mos010 `laue_dials_main` run and the hewl_1118 rerun |

Upstream main (rs-station/laue-dials) merged the new integrator in PR #86 on 2026-09-18: profile fitting in place of v0.5.1's summation, Sivia's prior on the final intensities, an integration radius estimated with a KD-tree, windows centred on floor(centroid), multi-panel support.
PyPI's newest release is still 0.5.1 (0.6.dev0 is a 2025-02 pre-release), so "laue-dials" without a commit is ambiguous.

**What each existing result used:**

- hewl_1118 shoeboxes and the `laue.integrate` reference `integrated_HEWL_anom_{0999,3049}.mtz` (May 2026): env `ld`, i.e. the v0.5.1-based summation integrator (`integrator_data/hewl_1118/process.sh`, `pytorch_data/mksbox.sh`).
- `docs/ablation_1118_1000.md` baseline (R-free 0.1780): built from `integrated_HEWL_anom_0999.mtz`, the same v0.5.1 integrator.
- mos010 simulation: `laue_dials_pc` (v0.5.1-based) and `laue_dials_main` (upstream main via `LAUE_DIALS_SRC`).

## Environments (`/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/`)

| env | key versions | used for |
|---|---|---|
| `ld` | DIALS 3.22.dev0, numpy 1.26.4, laue-dials from the holylabs checkout | the original laue-dials processing; the only env whose numpy reads datsimx CBFs |
| `integrator-cuda-dev` | DIALS 3.26, torch 2.9.1, numpy 2.5.2, laue-dials from the lab_storage `integrator` checkout | training, prediction, export, analysis; laue-dials reruns with `LAUE_DIALS_SRC` |
| `crls` | careless 0.5.4, TensorFlow 2.18.0 | careless scaling and `careless.stats` |
| `dials` | DIALS 3.27 | monochromatic DIALS chain |
| `refltorch` | DIALS 3.26, torch 2.9.1+cu128 | `refltorch.mksbox-laue`, cutting shoeboxes |

phenix: `/n/lab_storage/hekstra_lab/garden_2/phenix-2.0-5729` (phenix 2.0), named as `phenix.env` in the pipeline configs.

## integrator checkouts

| path | role |
|---|---|
| `/n/netscratch/hekstra_lab/Lab/laldama/integrator_main` | jobs are submitted from it (`SLURM_SUBMIT_DIR`) |
| `/n/lab_storage/hekstra_lab/people/aldama/software/integrator` | what `integrator-cuda-dev` imports; pull it too before training new code |

## Paths that moved

The storage migration moved `/n/hekstra_lab/...` to `/n/lab_storage/hekstra_lab/...`.
Files written before it still name the old prefix: hewl_1118's `poly_refined_HEWL_anom_3049.expt` points at `/n/hekstra_lab/people/aldama/data/1118/*.mccd`, which now live at `/n/lab_storage/hekstra_lab/people/aldama/data/1118/`.
`scripts/poly/laue_dials_rerun.sbatch` writes a copy with the prefix rewritten rather than editing the original.

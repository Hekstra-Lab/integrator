# Simulated Laue DHFR with known truth (datsimx)

Each dataset is a datsimx rotation series of still Laue exposures, rendered twice: noiseless (the truth) and with background and detector noise (the data every integrator sees).

## Pipeline

1. `datsimx.sim OUT --noNoise --waveImg --writeMtz --static --cuda ...` writes, per shot, the noiseless Bragg image `sim_image/` in photons, the per-pixel maps `h_data`, `k_data`, `l_data`, `wave_data`, and once per run the noiseless background `background_<run>.h5`, `ground_truth.mtz` and the geometry `geom_run<run>.expt`.
2. The noisy copy holds the master, geometry, background and shots files with `sim_image` only; `braggaddnoise` (nanoBragg-noise 1.9) overwrites that `sim_image` with $\mathrm{gain}\cdot\mathrm{Poisson}((B+b)(1+\epsilon_\text{calib})) + \text{readout}$.
3. `truth_table.py` reduces the truth to one row per (shot, hkl).
4. laue-dials (`scripts/poly/laue_process.sh` with `IMG_FILE=noisy/run_1_master.h5 REF_GEOM=noisy/geom_run1.expt`) and our integrator run on the noisy copy only.

All four steps for one grid row are `scripts/sim_truth/simulate.sbatch`; the grid files in `scripts/sim_truth/grids/` are the record of what was simulated, and each row's `manifest.json` records the commits of datsimx, easyBragg and this repository, the package versions, the GPU and the wall time.

```bash
cd $S/integrator_main && git pull
GRID=scripts/sim_truth/grids/sampling_check.tsv sbatch -o logs/sim_truth/sim_truth_%j.out scripts/sim_truth/simulate.sbatch
```

Outputs go to `/n/netscratch/hekstra_lab/Lab/laldama/sims/<grid name>/<row name>/{truth,noisy,manifest.json}`.

## Truth definitions

A pixel $p$ belongs to reflection $hkl$ when $B_p > 0$ and $\mathrm{round}(h_p, k_p, l_p) = hkl$; the maps are NaN exactly where $B_p = 0$.

| column | computed as |
|---|---|
| `i_true` | $\sum_p B_p$, photons before noise and gain |
| `n_px` | number of pixels with $B_p > 0$ |
| `n_eff` | $1 / \sum_p (B_p / I_\text{true})^2$ |
| `lam_mean` | $\sum_p B_p \lambda_p / I_\text{true}$ |
| `n_comp` | 8-connected components of the pixels with $B_p \ge 10^{-3} \max_p B_p$ |

The truth profile is $B_p / I_\text{true}$; the truth background is `background_<run>.h5`, identical for every shot of a run.
Where two reflections land on one pixel the map names one of them and that one receives all of $B_p$.

## Fixed settings

| setting | value | where it came from |
|---|---|---|
| structure | 7LVC (datsimx default `7lvc.pdb`), P2₁2₁2₁, 34.297 × 45.552 × 99.035 Å | datsimx package |
| structure factors | mmtbx Fcalc to 1 Å, $k_\text{sol}$ 0.4, $B_\text{sol}$ 120 Å², Henke $f'$, $f''$ at 1.1195 Å | datsimx defaults |
| spectrum | `from_vukica.lam`, 1.020–1.240 Å on the detector | datsimx package |
| detector | Eiger 4148 × 4362, 0.075 mm, 200 mm, no panel mask | datsimx defaults |
| crystal | `--Nabc 100 100 100`, orientation `Rotation.random(random_state=0)`, rotated 1° about $+x$ per shot, `--static` | datsimx |
| divergence | 0 | 0.1 mrad is 0.23 px at 200 mm |
| noise | gain 0.7 ADU/photon, readout 3 ADU, calibration 3 %, no PSF, no ADC offset | laue-dials paper sims, less PSF and ADC, which `braggaddnoise` lacks |

## Brightness

The target is 1/10 of the laue-dials paper simulations, measured from their CBFs at $4.4 \times 10^5$ and $5.0 \times 10^5$ Bragg photons per shot (`hkl_data` shots 1 and 90), so about $4.7 \times 10^4$.
The background does not depend on `--xtalSize`; the Bragg scale is easyBragg's `determine_spot_scale` (`simtbx/nanoBragg/sim_data.py`), the illuminated volume over the mosaic-domain volume, with the datsimx beam diameter $s_b = 0.01$ mm:

$$\text{spot scale} \propto \begin{cases} t\, s_b^2 & t \ge s_b \\ t^3 & t < s_b \end{cases}$$

| `--xtalSize` $t$ (mm) | Bragg photons, shot 1 | regime |
|---|---|---|
| 10 | $1.578 \times 10^8$ | linear |
| 1 | $1.578 \times 10^7$ | linear |
| 0.003 | $4.3 \times 10^3$ (eight settings, $4186$–$4331$) | cubic; $1.578 \times 10^5 \cdot 0.3^3 = 4.26 \times 10^3$ |
| 0.0067 | $4.746 \times 10^4$ (mosaic 0.1°, (100, 50), job 62469685) | cubic; predicted $4.7 \times 10^4$ |

The laue-dials paper's Bragg brightness corresponds to $t \approx 0.031$ in the linear regime.

## Sampling check

Grid `sampling_check.tsv`: one shot at mosaic spread 0.05° and 0.5°, (enSteps, mosDoms) = (322, 100), (100, 100), (100, 50), (50, 50), `--xtalSize 0.003`, scored by `compare_sampling.py` against (322, 100) at the same mosaic spread.

Job 62448785, 15.6 min on one A100-SXM4-40GB; `compare_sampling.py --min-photons 0.1`.
Wall time per shot (datsimx only, including about 30 s of structure factors and background): 248 s at (322, 100), 78 s at (100, 100), 50 s at (100, 50), 36 s at (50, 50).

**Spectrum samples are averaged.**
Total Bragg photons vary by at most 1.0 % between 50, 100 and 322 samples at fixed domain count, so the image brightness does not depend on `--enSteps`, and the `* len(weights)` in datsimx's `fluxes` cancels datsimx's own average.

| mosaic | (enSteps, mosDoms) | $\lvert\mathrm{dev}\rvert$ median, $I_R \ge 0.1$ | $\lvert\mathrm{dev}\rvert$ p95, $I_R \ge 0.1$ | $\lvert\mathrm{dev}\rvert$ median, top 100 | L1 median, top 100 | split, top 100 (reference 0.14 / 0.93) |
|---|---|---|---|---|---|---|
| 0.05° | (100, 100) | 0.078 | 0.30 | 0.072 | 0.78 | 0.60 |
| 0.05° | (100, 50) | 0.086 | 0.31 | 0.071 | 0.85 | 0.68 |
| 0.05° | (50, 50) | 0.187 | 0.62 | 0.150 | 1.33 | 0.98 |
| 0.5° | (100, 100) | 0.049 | 0.20 | 0.024 | 0.81 | 0.96 |
| 0.5° | (100, 50) | 0.085 | 0.39 | 0.059 | 1.53 | 0.96 |
| 0.5° | (50, 50) | 0.135 | 0.54 | 0.081 | 1.72 | 0.97 |

$\mathrm{dev} = (I_X / I_R) / (T_X / T_R) - 1$ per reflection; L1 $= \sum_p \lvert B_{X,p}/I_X - B_{R,p}/I_R \rvert$; split = fraction with `n_comp` > 1; $R$ = (322, 100) at the same mosaic spread.
1,820 reflections at 0.05° and 2,146 at 0.5° have $I_R \ge 0.1$.

No reduced setting is within 2 % of the reference.
The crops (`docs/figures/sim/sim_sampling_mos0p05.png`, `..._mos0p5.png`, from `fig_sim_sampling.py`) show why:

- at 0.05° the bright spots are 1–3 px wide streaks and look alike at (100, 100), (100, 50) and (322, 100); at 50 samples the rank-30 streak breaks into dots and its sum falls from 21.2 to 12.6 photons;
- at 0.5° the reference itself is a set of separate lines, one per mosaic domain, from rank 30 down, and 100 spectrum samples break those lines into dots.

Each (energy, domain) pair renders a spot of sub-pixel width at `--Nabc 100 100 100`, so a profile is continuous only when the samples are closer than a pixel in both detector directions.
At 0.5° the reference (322, 100) does not meet that, which also holds for the laue-dials paper's mosaicity-0.5 run (same sampling, plus 16 divergence directions at 0.1 mrad).
The truth sums are still exact for the image that was rendered; the sampling decides whether the profiles look like a continuous crystal.

## laue-dials on one shot at $4.7 \times 10^4$ Bragg photons

Row `brightness_check.tsv` (mosaic 0.1°, (100, 50), `--xtalSize 0.0067`), noisy copy, `laue_process.sh` with `IMG_FILE`, `REF_GEOM`, `TRUSTED=-1000,1000000`, SG 19, the 7LVC cell, λ 1.02–1.24 Å.

- The datsimx NeXus master declares a trusted range of (0, 0); without `TRUSTED` every pixel is masked and `laue.find_spots` finds nothing. 12 % of noisy pixels are negative (readout noise, minimum −14 ADU).
- Background pixels have variance/mean $= (0.49\lambda_b + 9)/(0.7\lambda_b)$ ADU, measured 2.72 at the median background $\lambda_b = 6.4$ photons, and 4.1 to 2.0 between 3 and 12 photons; the readout term dominates, so the dispersion gain 0.7 flags 6.2 M pixels.

| dispersion gain | strong pixels | spots extracted | spots kept in `strong.refl` | kept within 3 px of a truth centroid with $I \ge 1$ | truth reflections with $I \ge 20$ found |
|---|---|---|---|---|---|
| 0.7 | 6,243,337 | run stops | — | — | — |
| 2.7 | 6,600 | 6,029 | 66 | 0.20 | 0.040 of 302 |
| 4 | 345 | 238 | 15 | 0.60 | 0.026 of 302 |

`laue.index` fails on one image at either gain (12–15 centroids after its d_min 2.51 Å cut); it indexes a rotation sweep, so indexing needs a multi-shot test.
Our earlier `laue.find_spots` run on the laue-dials paper's `reference` set (defaults, `laue_runs/dhfr_sim/reference`) kept 177–219 spots per image, at 10× this brightness, 88.6 µm pixels and a 0.1 mm PSF.

## Matching the laue-dials paper images

The paper's simulations (`/n/holylabs/hekstra_lab/Users/laldama/data/simdhfr`) were made by `sim_laue.py` with the arguments in each directory's `commandline*.txt`, and with nanoBragg's own `SIM.add_noise()` at `detector_psf_fwhm_mm = 0.1`, `detector_psf_kernel_radius_pixels = 5`, `quantum_gain = 0.7`, `adc_offset_adu = 10`, `readout_noise_adu = 3`, on a 3840 × 3840 detector of 88.54 µm pixels at 200.5 mm.
`braggaddnoise` re-implements that noise in numpy without the PSF; `add_noise_nanobragg.py` calls nanoBragg's `add_noise` itself on the noisy copy.

`paper_match.tsv` emulates that detector in pixel units on the datsimx Eiger: `--dist 169.8` (0.44 mrad per pixel, as the paper) and `NOISE_TOOL=nanobragg NOISE_PSF_MM=0.0847` (1.13 px FWHM, as the paper). `image_stats.py`, one estimator for both:

| statistic | paper `hkl_data` shot 1 | datsimx, `paper_match`, 0.0067 mm |
|---|---|---|
| Var(net) = a·bg + b, photons: a, b | 0.090, 18.4 | 0.091, 18.2 |
| background, photons / sr (median) | $6.3 \times 10^7$ | $6.8 \times 10^7$ |
| background by 2θ, per sr | — | within 0–15 % of the paper in every 2.5° bin (200 mm run) |
| net Bragg photons | $4.40 \times 10^5$ | $3.46 \times 10^4$ (truth $4.75 \times 10^4$) |
| reflections on the image | 4,916 | 8,053 (the Eiger at 169.8 mm reaches higher 2θ) |

`laue.find_spots` (gain 1, max separation 10, min spot size from DIALS) against brightness, `paper_match` rows, one shot each:

| `--xtalSize` | Bragg photons (truth) | spots kept |
|---|---|---|
| 0.0067 | $4.75 \times 10^4$ | 6 |
| 0.0145 | $2.38 \times 10^5$ | 29 |
| 0.031 | $5.10 \times 10^5$ | 69 |
| paper `hkl_data`, 180 shots (`sim/laue180`) | $4.4$–$5.0 \times 10^5$ | 70 per image (mean) |

At the paper's brightness the emulation gives the paper's spot count.

## How laue-dials was run on the paper simulations

`scripts/sim_truth/paper_sims/` holds the scripts, as run in October and November 2026 from scratch.
`laue.index` is never run: `sim_build_stills.py` attaches each shot's true crystal from the h5 `Umat`, `Bmat` (checked against the truth centroids to a median 1.07 px), `sim_add_s1.py` adds the `s1` column `laue.index` would have, and `sim_laue_process.sbatch` runs `laue.find_spots` (gain 1.0, max separation 10) → `laue.optimize_indexing` (λ 1.00–1.20 Å, $d_\text{min}$ 1.15 Å) → `laue.refine` → `laue.predict` → `laue.integrate` into `sim/laue180/integrated.mtz`.
`sim_laue_all.sbatch` repeats the prediction with `cutoff_log_probability=-20` into `integrated_all.mtz`, because the default keeps 392 of about 3,000 excited reflections per shot.
`mos_laue.sh` ran `laue_process.sh` on the paper's three mosaicity sets into `sim_mos/`.

## Divergence

`paper_closest.tsv`, one shot at the paper settings (`e080_2.lam`, 322 energies, 100 domains, mosaic 0.1°, `--xtalSize 0.031`, `--dist 169.8`), with and without `--div 0.1 --divSteps 4`.
`--divSteps 4` samples a 5 × 5 grid from −0.05 to +0.05 mrad and keeps the 13 directions inside 1.1 × the half-angle (`nanoBragg_beam.py`); without `--divSteps`, `--div` has no effect.

| quantity | value |
|---|---|
| Bragg photons, no divergence / divergence | $4.753 \times 10^5$ / $4.756 \times 10^5$ |
| per-reflection $\lvert\mathrm{dev}\rvert$, median / p95, 3,794 reflections with $I \ge 1$ | 0.0048 / 0.030 |
| per-reflection $\lvert\mathrm{dev}\rvert$, median / p95, top 100 | 0.0029 / 0.024 |
| profile L1, median, top 100 | 0.034 |
| datsimx wall time per shot | 179 s / 2,292 s |

The largest beam tilt, 0.05 mrad, moves a spot by 0.11 px at 169.8 mm, a tenth of the PSF FWHM.
The full datasets (`paper_full.tsv`) are simulated without divergence.
Figures: `docs/figures/sim/sim_image_paper_vs_datsimx.png`, `docs/figures/sim/sim_sampling_mos0p1_divergence.png`.

## Full datasets

`paper_full.tsv`, row `mos010` first (20 Nov 2026), on `gpu` with a 36 h limit because datsimx writes all 180 shots in one call and a killed row restarts from nothing:

```bash
SPEC=/n/holylabs/hekstra_lab/Users/laldama/data/spectra/e080_2.lam NOISE_TOOL=nanobragg NOISE_PSF_MM=0.0847 \
GRID=scripts/sim_truth/grids/paper_full.tsv \
sbatch -p gpu -t 1-12:00:00 --mem=48G -c 4 --array=0 -o logs/sim_truth/paper_full_%A_%a.out scripts/sim_truth/simulate.sbatch
```

The noiseless background is one image per run (`truth/background_1.h5`): datsimx computes water and air scatter once, independent of the crystal, so shots differ in background only through the noise.

Reference results on it, chained after the simulation: `paper_sims/sim_laue_process.sbatch` with `IMG_FILE`, `REF_GEOM`, `TRUTH_SHOTS`, `BUILD_STILLS`, `ADD_S1`, `LD_ENV=.../integrator-cuda-dev` (step 0 imports the master as stills and attaches each shot's true crystal; on a one-shot test the crystals sit a median 1.22 px from the truth centroids and the chain runs through `laue.refine`, while `laue.predict` needs more than one shot), then `paper_sims/sim_laue_all.sbatch` with the same `OUT`, then `score_against_truth.py` on `integrated.mtz` and `integrated_all.mtz` into `paper_full/mos010/scores_vs_truth.csv`.

### mos010, first pass (20–21 Nov)

Simulation: 180 shots in 4 h 16 min on one A100 (job 62513336).
Step 0 position check, true crystal against the truth centroids: median 1.18–1.31 px, median $|\Delta\lambda|$ 0.00037–0.00039 Å (shots 1, 90, 180).
`laue.find_spots`: 45–87 spots per image, median 67.
`laue.refine`: 178 of 180 experiments refined (112 and 167 failed).

`laue.predict` (laue-dials 0.5.1, branch `integrator`) returned no predictions on any of the 178 images, then raised `Unknown column 'rlp'` on the empty table.
`predict.py:183` takes the mask row length as `image_size[1]`; `image_size` is (fast, slow), so on a non-square panel the flat mask index $x + 4362\,y$ runs past the 4148 × 4362 mask (380 of 6611 on-detector predictions on experiment 0), and the bare `except:` around it skips the image.
On the square Rayonix of the paper sims the two lengths are equal.
The fix (`image_size[0]`) is commit `5de60e4` on branch `eiger-row-size` of a separate clone, `/n/lab_storage/hekstra_lab/people/aldama/software/laue-dials-eiger`; the lab checkout is not modified.
Both laue sbatches take `LAUE_DIALS_SRC=<clone>/src`, which puts the clone first on `PYTHONPATH` and logs the module path and commit.

After the fix, `laue.predict` ran on all 178 images (42 s), and integration and scoring finished in 11 min (jobs 62600223, 62600225, 62600227).
Join: 88.3 % of 59,788 MTZ rows (default predictions) and 88.5 % of 144,780 (permissive), sign-flipped key 0.000; median $|\lambda_\text{mtz} - \lambda_\text{mean}|$ 0.0022 and 0.0036 Å.
`laue.integrate`, per observation in photons, default set: dev median +39 at $I_\text{true} < 10$, $-71$ at 100–300, $-765$ at 1000–3000 (mean $(\hat I - I_\text{true}) / I_\text{true} = -0.43$), $-0.48$ above 3000.
These are not read as laue-dials' accuracy until the PSF result in `sim_truth_protocol.md` §9 is in the reference: 14.6 % of every spot's photons are spread over the detector by nanoBragg's truncated PSF kernel, and a further 9.5 % lie 3–5 px from the source.

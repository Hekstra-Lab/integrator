# Training the integrator on PDZ2 / 1116

PDZ2 is the only dataset where the endpoint is **CCsym** rather than anomalous peaks, and that changes what to train for and how to read the result.
This file describes the procedure end to end: what the data is, what the configuration does, how to launch, and what the evaluation traps are.
Companion to `results_by_dataset.md`, which records what has been achieved.

## What the dataset is

LNX2 PDZ2, the electric-field TR-Laue study (Nature 2016, reference structure 5E11 for the off state).
A field is applied to the crystal and the structural response is read out as the difference between field-on and field-off.
The signal is roughly **1% of F**, so this is the most demanding dataset we have, and it is not an anomalous problem at all.

The endpoint is the half-dataset correlation of `F(h) - F(2-fold h)`, which must be zero without a field.
Four conditions exist -- `off`, `50ns`, `100ns`, `200ns` -- across two crystals, `e35c` and `e35d`.

## Prerequisites, all of which already exist

| what | where |
|---|---|
| training shoeboxes | `/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/pooled` |
| laue-dials processing | `laue_runs/pdz2/e35{c,d}_{off,50ns,100ns,200ns}` |
| reference pipeline | `laue_runs/pdz2_ref/gain_0,3_from_stills` |
| configurations | `configs/poly/pdz2_1116*.yaml` |

The pooled dataset holds **993,143 reflections** as 21x21 single-slice images, space group `C 1 2 1`, cell 63.35 38.29 37.89 90 117.48 90, beam centre (1984.99, 1967.21).
It carries `group_labels_1.npy` and `sweep_ranges.yaml`, which is what lets a split be made by something other than image index -- see the splitting trap below.

## The configuration

`configs/poly/pdz2_1116.yaml` is the baseline.
Four variants exist and differ by exactly one knob each, so a sweep over them is interpretable.

| config | change | why |
|---|---|---|
| `pdz2_1116` | -- | baseline |
| `pdz2_1116_k7` | first conv kernel 3 -> 7 | kernel should track the spot, not the box; k=7 was the 1118 plateau |
| `pdz2_1116_w35_k3` | window 21 -> 35 px | tests whether 21 px clips the streak |
| `pdz2_1116_w35_k7` | both | the two above together |
| `pdz2_1116_kl0p1` | `intensity_kl_weight` 1.0 -> 0.1 | weakens the Wilson prior's pull on I |

Choices worth understanding rather than copying:

**`loss.name: polychromatic_wilson` with `degree: 40` over `lambda` 0.95-1.20 A.**
The spectrum `G(lambda)` is learned as a Chebyshev polynomial in log space.
The range comes from the laue-dials tutorial for this dataset and must match the data, since a spectrum fit outside the true support is unconstrained where it matters.

**`qp.latent_dim: 16`, larger than the monochromatic default.**
Laue profiles are elongated and vary with detector position, so the profile latent carries more.

**`polarization: false`, `lorentz: false`.**
Off by default; `polarization: true` additionally requires `beam_center`.
Whether these belong on is a real question for Laue and has not been settled here -- see the `lp_correction` convention note in `results_by_dataset.md`.

**`transform: anscombe`.**
The manifest statistics must match the data.
The `mksbox_laue` datasets had wrong Anscombe statistics in their manifests, fixed at source, and **checkpoints trained before that fix need the old statistics to reproduce**.

**`n_bins: 1` for the background prior.**
A single resolution shell.
On a dataset this background-dominated a per-bin background prior is worth testing.

## Launching a run

Training goes through the poly driver, which is the same chain used for 1118.

```bash
cd scripts/poly/cannon
./02_train.sh ../../configs/poly/pdz2_1116.yaml     # or 02c_train_gputest.sh to smoke it
```

The driver writes a run directory containing `run_paths.yaml`, which every downstream script reads to find `predictions_dir` and `plots_dir`.
Nothing downstream should ever be pointed at a hard-coded path; that indirection is what survives a relocation.

Settings that matter at launch: `max_epochs: 100` with cosine warmup over 5 epochs, early stopping on `val elbo` with patience 3, and `save_top_k: -1`, which keeps **every** checkpoint.
Keeping them all is what makes a per-epoch trajectory possible later, and it is the setting most easily lost by copying a different config.

## The per-epoch trajectory

This is the part that does not yet exist for any dataset, and it is what turns a run into R-value, peak and merging-statistic curves.

The downstream chain per checkpoint is:

```
predict  ->  careless (config 4)  ->  phenix (two-pass)  ->  peaks / CCsym
```

`scripts/poly/run_pipeline.py --run-dir <run> --steps predict,careless,phenix,peaks` runs it once.
A trajectory means running it for each checkpoint and writing the output under `<predictions_dir>/epoch_<NNNN>/`, because `plot_peaks.py` extracts the epoch from the path with the pattern `/epoch_(\d+)/`.
Given that layout, `scripts/make_figures.sh --plot-cfg <cfg>` produces the per-epoch loss, peak-height and R-factor figures with no further work: `plot_peaks.py` already parses R-work and R-free out of the phenix logs, and an epoch whose refinement failed leaves a gap rather than failing the figure.

The precedent to avoid repeating is `sbgrid/821/_incomplete_epochs_hold`, which holds 19 epochs of `pred.parquet` and nothing else -- the expensive half was done and then abandoned before anything interpretable came out of it.

## Reading the result, and three ways to get it wrong

**Raw CCsym is not a signal measurement.**
`laue.integrate` shows CCsym 0.10-0.20 in the **field-off** state, where the true answer is exactly zero.
So a substantial part of any raw CCsym is reproducible non-field asymmetry, and only the field *response* -- on versus off -- means anything.

**Splitting halves by image lets a method borrow across the split.**
Profile and centroid information shared between same-image neighbours appears as cross-half covariance and scores as signal.
This is what inflated earlier CCsym comparisons, and it rewards exactly the profile borrowing that `laue.integrate` does by construction.

**Use `CC_Delta`, and the right transform.**
The recommended estimator forms the per-reflection field response *inside* each half and correlates those, reporting `2CC/(1-CC) = S/N`; it is split-rule invariant and cancels bias common to on and off.
The odds transform `r = CC/(1-CC)` is correct where Fisher's z is not, but `r` is strained as CC approaches 1, which is the **Q4** regime where the headline sits.
Normal-theory standard errors understate by about 1.8x from heavy tails, so significance should come from a paired reflection bootstrap rather than a normal approximation.

## What is already known

Our prior-free ML arm beats `laue.integrate` on R-free by 0.0056 (about 3.8 sigma, replicated), on a refinement pipeline we built -- the deposited pipeline has none.
Likelihood-export scored best and posterior-export worst, which is the same export defect documented in `integrator_theory_unified.md`.
On the field response the quartile picture inverts relative to raw CCsym: Q2 and Q3 tie and Q4 leads by +0.241.

An earlier conclusion that PDZ2 hit a data wall was **wrong** and has been corrected: the reference pipeline had simply never been run verbatim.

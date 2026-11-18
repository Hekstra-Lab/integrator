# The figures

The figures this work is reported through, named by the user on 2026-11-17.
Everything else is a diagnostic and does not live here.

Each is drawn by one script in `scripts/figures/`, with figTeX and the
**witch** colour scheme, from a long-format table in `docs/figures/data/`.
Build on the scripts that are there; a new one-off plotting script beside
them is a defect.

The monochromatic figures are read **over training**, one line per checkpoint
with the epoch on a sequential colour ramp and the reference integrator as a
single line in a contrasting colour: scoring a monochromatic checkpoint is
cheap enough to do at every one.
`figstyle.epoch_ramp(n)` is that ramp and its colorbar, and
`figstyle.reference_style()` is the reference line, a categorical colour so it
cannot be read as a checkpoint.

**The polychromatic figures are drawn at one checkpoint, not over training**
(decided 2026-11-17): a polychromatic epoch costs too much to repeat ten
times.

## The set

The scripts are the ones that already drew these figures, converted to figTeX
in place; a figure is not given a new script because it is being restyled.
Several now split in two, because the measurement is on the cluster and the
fonts are here: a `--dump`/`--dump-dir` mode reduces the run to the small
file named below and draws nothing, and a `--data` mode draws from it.

| figure | script | data | state |
|---|---|---|---|
| input shoeboxes against the predicted profiles | `plot_profile_grid.py` | `shoebox_profiles.npz` | built |
| training loss: NLL and each KL term separately, train against validation against test | `plot_loss.py` | `training_loss.csv` | built; test split not logged |
| model intensity against DIALS's profile fit (`intensity.prf.value`) | `plot_compare.py` | `scatter_points.csv` | built |
| model intensity against DIALS's summation (`intensity.sum.value`) | `plot_compare.py` | `scatter_points.csv` | built; needs a redump, see below |
| model background against the reference integrator | `plot_compare.py` | `scatter_points.csv` | built |
| the same two correlations within resolution shells, over training | `plot_compare.py` | `scatter_cc.csv` | built |
| merging statistics by resolution shell: $R_\mathrm{pim}$, $CC_\mathrm{anom}$, $CC_{1/2}$, $I/\sigma_I$ | `plot_merging.py` | per-epoch `merging_stats.csv` | built |
| refinement: $R_\mathrm{free}$ and $R_\mathrm{work}$ against the DIALS reference | `plot_peaks.py` | per-epoch `refine_001.log` | built |
| anomalous peak height, one panel per site | `plot_peaks.py` | per-epoch `peaks.csv` | built; atom name needs `pull_figdata.sh peak_models` |
| mean anomalous peak height over the anomalous scatterers | `plot_peaks.py` | per-epoch `peaks.csv` | built |
| polychromatic: model background against laue-dials | `plot_compare.py` | `scatter_points.csv` | blocked, see below |
| polychromatic: the learned spectrum $G(\lambda)$ | `scripts/visualize_spectrum.py` | `spectrum.csv` | to convert |
| polychromatic: $CC_{1/2}$, $CC_\mathrm{anom}$, $CC_\mathrm{pred}$, $R_\mathrm{split}$, $I/\sigma_I$ and completeness from `careless.stats` | `fig_careless_stats.py` | `careless_stats.csv` | built; drawn on 1118 and on all six PDZ2 arms |
| the Wilson prior's learned scale and $B$ over training | `fig_wilson_params.py` | `wilson_params.csv` | built |
| every component of the learned profile basis, and how it settled | `fig_profile_basis.py` | `basis/` | built |
| PDZ2: $CC_\mathrm{sym}$ by resolution shell, one panel per pump delay | `fig_ccsym.py` | `pdz2_ccsym.csv` | built |

Three of those were added on 2026-11-18 and two of them needed no new
measurement, only a reduction that had never been written.
The Wilson scale and $B$-factor have been in every run's
`plots/wilson_params.csv` since the logger was added, and the profile basis
is a parameter, so every checkpoint holds it and `dump_basis.py` recovers
the whole trajectory without retraining -- `ProfileBasisLogger` would have
written it during training but is gated behind `integrator.train --figures`,
which no production run passed.

PDZ2 figures are written to `docs/figures/pdz2/` rather than `m9/`, and the
gallery is built over `docs/figures` so that it spans both.

`coverage.md` beside this file is the other half of the list: which of these
figures can be drawn on a monochromatic run, which on a polychromatic one,
and what is missing where one cannot -- kept apart into values that do not
exist upstream and code that is not written yet.

## Rendering the whole set

`bash scripts/figures/render_all.sh` draws every figure above from
`docs/figures/data/` and rebuilds the gallery, in about twelve minutes and
without touching the cluster.
Run it after any change to figTeX, a colour scheme or the mydesign LaTeX
styles: one figure rendering correctly says nothing about the other
forty-three.
Targets are selectable, `render_all.sh merging peaks`, and
`docs/figures/gallery.html` is every figure on one page in the order of the
list above.

### Two things the data on disk does not carry yet

`intensity_sum` draws nothing until `scatter_points.csv` is dumped again.
The file on disk was reduced when there was one DIALS intensity, so it has
`intensity.prf.value` and `background.mean` and no third quantity; those two
still draw, under their current names `intensity_prf` and `background_mean`.
Redump with `bash scripts/figures/pull_figdata.sh scatter`.

The atom carrying each peak is not on disk until the refined models are
beside the peak lists.
`rs.find_peaks` has a `name` column for the nearest atom and the fork in use
writes it empty, so the atom is resolved against the refined PDB the peak
search itself ran on -- a sulfur peak is SD on a methionine and SG on a
cysteine, and the residue name does not say which.
`PEAK_RUN=<cluster run directory> bash scripts/figures/pull_figdata.sh
peak_models` brings those models down.
Checked against one of them on the real peak list, 13 of 16 sites resolve;
the three that do not are left as the residue alone, because the model's
answer is taken only where its chain, residue number and residue name agree
with what the peak search reported.

A panel title does not show the atom even once it is known.
Every residue in this experiment holds exactly one atom that could carry an
anomalous peak -- one iodine in an iodide, one sulfur in a cysteine, a
methionine or a sulfate -- so `CYS 30 SG` says sulfur twice and the title is
`CYS 30`.
`metrics.panel_site_label` adds the atom only where
`metrics.heavy_atom_candidates` counts more than one in that residue, and
the `site` column of `anomalous_peaks.csv` keeps it unconditionally, because
a column is read by name and one that sometimes carried the atom could not
be grouped on.

Two figures are kept from the geometry investigation because they answer a
question none of the above does: `fig_profile_shape.py` (does a profile model
follow the spot shape beside it) and `fig_detector_patch.py` (what that looks
like on one patch of one image). `fig_arm_metrics.py` compares arms at a
single checkpoint and is a diagnostic, not a reported figure.

`scripts/figures/tidy_mono.py` reduces a monochromatic run to
`training_loss.csv`, `merging_shells.csv`, `refinement.csv` and
`anomalous_peaks.csv`. It is kept for the arm comparison, which reads many
runs at one checkpoint rather than one run over many.

## Where a number is read

Nothing here parses a tool's output for itself. `plot_merging.py`,
`plot_peaks.py`, `tidy_mono.py`, `make_tables.py` and
`fig_careless_stats.py` all read through `integrator.utils.metrics`, which
defines the formats and the chain's own emitter in one place and is pinned
by `tests/test_metrics_contract.py`, `tests/test_dials_log.py` and
`tests/test_careless_stats.py`.

The DIALS log tables are read by `integrator.utils.dials_log`, which
`metrics` re-exports, so a caller still has one import.

They used to hold a parser each, and the copies had drifted: three different
regular expressions for the one `RESULT` line, two column-matching rules for
the `dials.merge` table. `results_table.py` had been rendering an **empty
table** because its expression no longer matched anything, and reported that
as an empty table rather than as an error.

The three forms a run can be scored through are read by the same function,
so a figure does not care which:

| form | written by | read by |
|---|---|---|
| `merging_stats.csv` | `emit_merging_stats.py`, both arms | `metrics.shells_from_merging_stats_csv` |
| `merged.html` | `dials.merge` | `metrics.shells_from_merged_html` |
| `merge.log` | `dials.merge` | `metrics.shells_from_merge_log` |

`merged.html` used to be read with `pandas.read_html`, which needs `lxml`;
without it the call raised, the reader returned `None`, and the DIALS
reference line was missing from all five merging figures with only a
warning.
The six tests that would have caught it were `importorskip("lxml")` and
skipped silently.
It is read with the standard library now, and the tests run.

Two details the tests pin because they are easy to get wrong and silent when
wrong: the **last** `Final R-work` line wins, since phenix prints `Start
R-work` first and a rerun appends; and `dials.merge` marks a significant
correlation with a trailing `*`, which read as text sorts as text.

## The tables

`scripts/figures/make_tables.py` writes `docs/tables/`, each table as `.tex`
for the manuscript and `.md` so the numbers can be read without compiling
anything.
They come from the `scale.log` and `merge.log` each arm already wrote, which
carry a good deal that never reaches `merging_stats.csv`.

| table | what it holds |
|---|---|
| `merging_by_arm` | the overall merging statistics for every arm, both estimators |
| `r_factors` | $R_\mathrm{merge}$, $R_\mathrm{meas}$ and $R_\mathrm{pim}$, each both merged and with the Bijvoet mates held apart |
| `error_model` | the fitted $a$ and $b$ of $\sigma'^2 = a^2(\sigma^2 + (bI)^2)$, the parameter count, the outliers rejected |
| `wilson` | the maximum-likelihood Wilson $B$, the scale factor and the anisotropy spread |
| `merging_shells` | the per-shell table for one arm |
| `anomalous_differences` | $\langle|\Delta F|/\sigma(\Delta F)\rangle$ by shell, one column per arm |

The LaTeX is the mydesign dialect, settled with the figtex side on
2026-11-17 and implemented in `scripts/figures/tables.py`: booktabs rules
only, `siunitx` `S` columns so numbers align on the decimal point, units on
a second header line, the body at `\small` or at `\footnotesize` with
`tabcolsep` at 4 pt once the table is wide, and never `\resizebox`.
A best value is marked with Minion Semibold and only in a table that
compares arms.

## What is blocked

**The polychromatic background scatter.** `laue.integrate` reports no mean
background per reflection, so there is nothing to put on the reference axis.
It has to be added to laue-dials: its integrator fits a background plane
internally, and the mean of that plane over the shoebox is the quantity our
`qbg_mean` should be compared against.

**The test split in the training-loss figure.** `loss_history.csv` carries
`train_*` and `val_*` only. The test split is the one that gets exported, so
its loss needs an evaluation pass at each checkpoint rather than a column.

## The data

Long format, one file per kind, `method` and `dataset` key columns throughout,
plus `epoch` wherever the quantity is read over training. See
`docs/figures/data/README.md` for the columns of each file and
`scripts/figures/tidy_*.{py,sh}` for what produces them on the cluster.

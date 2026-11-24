# Plot-ready tables

One long-format CSV per result kind, in the shape figtex reads, and nothing
else.
Figures are rendered on the laptop, because the documents are set in Minion
Pro with mtpro2 maths through LuaLaTeX and those fonts live there; so each
measurement is reduced on the cluster to the columns a figure actually draws
and only those files cross the wire.
A table here is tens of rows where the measurement behind it ran over 1.94
million reflections.

Every file carries `method` and `dataset`, the key columns of
`figtex.project.schema`, so these drop into the paper project
(`~/master/notebooks/integrator_paper/data/results/`) unchanged.
`method` is the training arm (`m9_geom`, `m9_geomshift`, ...); `dataset` is
`hewl_9b7c`.

Colour comes from the witch scheme, one colour per arm, assigned in
`scripts/figures/arms.py` so an arm keeps its colour across figures.

## The files

`profile_shape.csv`
: One row per (`method`, `dataset`, `reference`, `metric`).
  `reference` is `local` (the $k$ bright reflections nearest the weak one),
  `far` (bright reflections at least `far_px` away, which a model carrying no
  position information must match) or `ceiling` (a second disjoint draw of
  local bright reflections, so it bounds what any model could reach).
  `metric` is `dtheta_deg` (axial orientation difference from the central
  second-moment tensor; the null is exactly 45 degrees because the axis is
  defined modulo 180), `fft_cos` (cosine of the modulus of the Fourier
  transform with the DC term dropped) or `dlogratio`.
  `value` is the median, with `p25` and `p75` beside it, over `n` weak
  reflections.
  Both metrics are translation invariant: an earlier version compared
  profiles by cosine similarity and read back 0.42 to 0.65, which is the
  chance level for nonnegative vectors rather than a result.
  The orientation index $(\mathrm{local} - \mathrm{far}) /
  (\mathrm{ceiling} - \mathrm{far})$ is not a column; the figure computes it
  from three rows, so its definition lives in one place.

`profile_width.csv`
: One row per (`method`, `dataset`, `population`), `population` being `weak`
  or `local_bright`.
  `axis_ratio` is $\sqrt{\lambda_1/\lambda_2}$ of the central second-moment
  tensor.
  An axis ratio is a property of a set of profiles, not a comparison against
  a reference, which is why it is a separate file.

`scatter_points.csv`
: One row per reflection drawn, keyed by (`method`, `dataset`, `split`,
  `epoch`, `quantity`), with `model`, `dials` and `d`.
  `quantity` is `intensity_prf` (the model's $\langle I\rangle$ against
  `intensity.prf.value`, DIALS's profile fit), `intensity_sum` (the same
  model column against `intensity.sum.value`, its flat sum over the mask) or
  `background_mean` ($\langle b\rangle$ against `background.mean`).
  The model has one intensity and DIALS has two, and they are different
  integrators, so the same model column is compared against each.
  Points are subsampled to `--max-points` with a fixed seed; a file dumped
  before `intensity_sum` existed calls its two quantities `intensity` and
  `background` and is read under the current names.

`scatter_cc.csv`
: The same keys plus `bin`, `d_max`, `d_min`, `n` and `cc`: the correlation
  between the model and the reference within resolution bins, at every
  checkpoint rather than only the one the scatter is drawn at.
  The bin edges are quantiles of $d$ taken once, from the first epoch read,
  because $d$ is a property of the reflection and re-quantiling per epoch
  would make the lines incomparable.

`anomalous_peaks.csv`
: One row per (`method`, `dataset`, `estimator`, `variant`, `epoch`, site),
  with `site`, `chain`, `residue`, `seqid`, `atom`, `element`,
  `peak_height`, `dist` and `atom_from_position`.
  `site` is the name a panel is titled with, `IOD 204 I` or `MET 105 SD`:
  the residue alone does not say which atom carries the peak, since a
  sulfur peak is SD on a methionine and SG on a cysteine.
  `atom` is resolved against the refined PDB beside the peak search's own
  output, because the fork of `rs.find_peaks` in use writes its `name`
  column empty, and the model's answer is taken only where its chain,
  residue number and residue name agree with what the peak search reported.
  `atom_from_position` is True where the atom is the one nearest the peak
  and False where it is the heaviest atom of the named residue, which is
  all that can be said for the older `peak_heights.csv` -- that file
  carries no positions.
  `dist` is the peak-to-atom distance the peak search reported, in
  angstroms.

`wilson_params.csv`
: One row per (`method`, `dataset`, `epoch`, `parameter`), with `value`.
  `parameter` is `B`, and `G` or `G_mean` depending on whether the run had
  one scale group or several, plus `G_min` and `G_max` where there were
  several.
  Long format because a B-factor in square angstroms and a scale in counts
  do not share an axis.
  From each run's `plots/wilson_params.csv`, which `WilsonParamLogger`
  writes every epoch from the loss's own `get_G()` and `get_B()` rather
  than from a stored copy.

`basis/<arm>/basis_snapshots.npz` and `basis/basis_components.csv`
: The learned profile basis per epoch.
  The npz holds `epochs`, `weights` of shape (epochs, $d\,h\,w$,
  components), `biases` and `shape`; images stay there because an image is
  not a table.
  The CSV is one row per (`method`, `dataset`, `epoch`, `component`) with
  `norm` and `cos_to_final`: the norm says whether a component carries any
  amplitude and the cosine says when it stopped moving, and both are per
  component because the failure that matters is one collapsing while the
  rest look settled.
  Written by `dump_basis.py` from the checkpoints, not during training:
  `ProfileBasisLogger` would have written the same files but is gated
  behind `integrator.train --figures`, which no production run passed.
  The geometric surrogate decodes onto a square canvas larger than the
  shoebox, so the row count is not always $d\,h\,w$ and the figure
  recovers the canvas from it.

`pdz2_ccsym.csv`
: One row per (`method`, `dataset`, `delay`, `bin`, `repeat`) with `d_max`,
  `d_min` and `ccsym`.
  `delay` is `off`, `50ns`, `100ns` or `200ns`, and the off rows are the
  control rather than a baseline to subtract: the true value there is
  exactly zero.
  The correlation is **Spearman**, which is what `custom_ccsym.py` defaults
  to and what the deposited production script therefore computed.
  Two things are undone on the way in: that script doubles its own bin
  index before saving, and it saves no resolution at all, so the shell
  edges are recomputed by calling its own `make_halves_ccsym`.

`pdz2_refinement.csv`
: One row per (`method`, `dataset`, `replicate`) with `r_work` and
  `r_free`, from the field-off refinement.
  `replicate` is the careless merging seed, which dominates the refinement
  seed about ninefold on Laue data.
  The deposited pipeline computes no refinement for PDZ2; this chain is
  ours, with one R-free set of 404 reflections shared across every arm so
  the arms are compared on the same held-out reflections.

`careless_stats_pdz2.csv`
: The same long format as `careless_stats.csv`, one file for every PDZ2
  arm.
  `ccanom` is absent and not missing: the PDZ2 merge omits `--anomalous`,
  so there are no Friedel mates for it to correlate.
  The rows are from file index 0 of the eight-file merge, which is the
  field-off C2 merge -- the file the refinement endpoint uses.

`hparams.csv` and `hparam_labels.csv`
: One row per (`method`, `dataset`, `group`, `parameter`), with `value`,
  `unit` and `varies`.
  What every model was configured with, read from each run's **own**
  `config_log.yaml` rather than from the repository config -- they are not
  the same file, and `configs/poly/pdz2_1116.yaml` today says
  `hierarchical_3enc` while every PDZ2 run on disk trained `hierarchical`.
  Nothing new is logged during training to produce this; the settings were
  always recorded and only the reading was missing.
  `varies` is 1 where a parameter differs across the arms in the table, and
  that is a property of the **set**: a table is built by passing every arm
  that belongs in it to one invocation, and concatenating two halves gives
  a `varies` column that is wrong in both. A parameter absent from one arm
  counts as varying, because an arm without a setting differs from one that
  has it.
  Repeated encoders collapse: the hierarchical integrator takes four
  identical intensity encoders through a YAML anchor, and they are reported
  once with `encoder.intensity_encoder.n`, since four identical rows in a
  paper table are noise.
  Output directories, checkpoint paths and logger settings are dropped --
  they record where output went, not what was configured.
  The reading is in `integrator.utils.hparams` and pinned by
  `tests/test_hparams.py`; `scripts/figures/tidy_hparams.py` only locates
  the runs.

`cmp_mono.csv`, `cmp_1118.csv`
: One row per (`method`, `dataset`, `integrator`, `qp`, `quantity`), with
  `value` and `unit`.
  `quantity` is `n_params_<module>` and `n_params_total`, summed from the
  checkpoint's state dict; `n_epochs`; `seconds_per_epoch`; and
  `timing_intervals`.
  Seconds per epoch is the **median** over the gaps between consecutive
  checkpoints divided by the checkpoint stride -- median because a requeued
  job leaves one enormous interval that a mean would report as a slow
  model. `timing_intervals` is how many gaps that median was taken over, so
  a run with too few to mean anything is visible.
  It compares only runs on the same hardware and the same dataset.
  Inference time is **not** here and is recorded nowhere: it needs a timed
  predict pass on fixed hardware.

`arm_metrics.csv`
: One row per (`method`, `dataset`, `estimator`, `metric`).
  `estimator` is `rb` (the Rao-Blackwellized profile fit computed at export,
  from `i_rb`/`var_rb`) or `model` (the integrator's posterior, from
  `qi_mean`/`qi_var`); both are always present, because one training gives
  two sets of intensities and sigmas, `dials.merge` weights by $1/\sigma^2$,
  and which one wins is not monotone across arms.
  `metric` is `i_over_sigma`, `cc_anom`, `r_free_f`, `r_free_i` or
  `max_peak`.
  Long format rather than a column per metric, because the metrics have
  different units and are drawn in different panels.

`sbc/<run>/sbc_ecdf.csv`, `sbc/<run>/sbc_summary.csv`
: Simulation-based calibration of one trained run, from `scripts/sim_truth/sbc_ranks.py` on a dataset drawn by `scripts/sim_truth/sbc_simulate.py`.
  One row per (`method`, `dataset`, `quantity`, `binning`, `lo`, `hi`) and, in `sbc_ecdf.csv`, per grid point `z`.
  `quantity` is `I`, `b`, `loglik`, `n_eff`, `i_rb` or `eta_<k>`; `binning` is `all`, `qi_mean` (the posterior mean of I, a function of the counts), `i_true` or `d`, with the bin edges in `lo`, `hi`.
  An exact posterior is uniform within a `qi_mean` or `d` bin and not within an `i_true` bin, where its shrinkage towards the prior shows; `i_true` is for `i_rb`.
  `ecdf_minus_z` is ECDF(z) - z of the PIT values; `fig_sim_sbc.py` computes the simultaneous band from `n`.
  `sbc_summary.csv` carries `n`, `mean_minus_half`, `var12` (12 var(u)) and `ks` (the Kolmogorov statistic).

`prior_check/prior_check_summary.csv`, `prior_check/prior_check_ecdf.csv`, `prior_check/prior_predictive.csv`
: Prior checks of trained runs' Wilson intensity prior, from `scripts/sim_truth/prior_check.py`.
  The summary has one row per (`method`, `dataset`, `binning` in `all`, `d`, `wavelength`; `group` in `all`, `acentric`, `centric`; `lo`, `hi`), with `ratio` (mean true I over mean prior mean), `n`, `mean_minus_half`, `var12` and `ks` of u = F_prior(I_true).
  `prior_predictive.csv` holds per (`method` = the prior draw, `statistic` in `total`, `max`, `median`, d bin `lo`, `hi`, quantile `q`) the `real` and `sim` quantiles of that shoebox statistic.

## Rebuilding

```bash
bash scripts/figures/pull_figdata.sh                 # cluster -> here
uv run --with-editable /Users/luis/master/software/figtex \
    python scripts/figures/fig_profile_shape.py      # here -> docs/figures/m9/
```

The cluster side is `tidy_mono.py` (over the run directories and the
refinements), `tidy_pdz2.py` (over the PDZ2 arms), `dump_basis.py` (over the
checkpoints), `tidy_shape.py` (over the measurement's JSON) and
`tidy_xtal.sh` (over the DIALS logs), all kept in `scripts/figures/` and
copied to the job directory.
`plot_compare.py --dump-dir` does its own reduction, where the 1.94 million
reflections are.

The numbers behind the LaTeX tables are not here but in
`docs/tables/data/`, written by `scripts/figures/make_tables.py` from the
pulled DIALS logs rather than on the cluster.

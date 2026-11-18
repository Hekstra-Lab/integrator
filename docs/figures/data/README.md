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

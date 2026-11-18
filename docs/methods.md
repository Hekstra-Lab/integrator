# How each dataset was processed

What actually ran, with the script and the parameter, for every dataset the project reports a number on.

This is a methods record, not a plan.
Where the chain on disk differs from what a previous note claimed, the chain wins and the difference is stated.
Where a step's record exists only in the session notes and was not re-read from the scripts, it says so, because an unverified methods line is worse than an absent one.

Companions, which this file links to rather than restates: `exporting.md` for the export paths and column modes, `downstream_stats.md` for the shared per-shell contract, `results_by_dataset.md` for what the numbers came out as, `pitfalls.md` for the failures that produced plausible wrong ones, and `reference_status.md` for how far each reference arm got against its deposited target.

Read against the cluster and the repository on 2026-11-18.

## The two chains

Both are written and both take `--from/--to/--only` over the same stages.
Do not write a new driver for either; extend the stage that is wrong.

| stage | monochromatic | polychromatic |
|---|---|---|
| front door | `scripts/mono/cannon/run_all.sh` | `scripts/poly/cannon/run_all.sh` |
| reference integration | DIALS, per dataset | `scripts/poly/laue_process.sh` (laue-dials) |
| shoeboxes | `refltorch.mksbox-dials` | `refltorch.mksbox-laue` |
| training | `integrator.train --config <yaml>` | the same |
| export | `integrator.predict --write-refl` | `integrator.predict --write-mtz` |
| scale and merge | `dials.scale` then `dials.merge` | `careless poly` |
| per-shell statistics | `scripts/mono/emit_merging_stats.py` | `scripts/poly/emit_merging_stats.py`, then the `stats` step |
| refinement | `phenix.refine`, the dataset's own `.eff` | `phenix.refine`, two passes |
| anomalous peaks | `rs.find_peaks` | `rs.find_peaks` and `anomalous_peak_heights.py` |
| orchestration | `03_pipeline.sh` to `process_single.py` | `scripts/poly/run_pipeline.py` |

The polychromatic chain is one Python driver reading one configuration, `scripts/poly/poly_pipeline_cfg.yaml`, and `--dry-run` prints every command it would submit without submitting any.
The monochromatic chain is a chain of SLURM jobs linked with `--dependency=afterok`, one per stage, because training holds a GPU for hours and DIALS and phenix are CPU work that would hold it idle.

### How many checkpoints get scored

One setting, resolved for both arms by `integrator.utils.checkpoints`: `last`, `every:<n>`, `all`, or an epoch number.

- **Monochromatic defaults to `every:5`.** DIALS and phenix are CPU, so a trajectory is affordable.
- **Polychromatic defaults to `last`.** One careless scale is tens of GPU-minutes.

`every:<n>` is a **stride in epochs, not a divisibility test**.
Checkpoints do not land on multiples of $n$ -- 9b7c saves at 4, 9, ..., 49 -- and the final checkpoint is always kept.
`tests/test_checkpoint_selection.py` pins this.

## What every run shares

### Shoeboxes and the manifest

A training dataset is a directory holding `counts.npy`, `masks.npy`, `metadata.pt` and a `dataset.yaml` manifest, written by `refltorch`.
The manifest carries the geometry, the crystal, and the normalization statistics:

```yaml
geometry:   {d: 1, h: 21, w: 21, data_dim: 2d}
stats:
  anscombe: [6.998962787650769, 2.585584214583349]   # mean, sd
  raw:      [12.517766079375894, 2392.620870282125]
```

The data module transforms every network input as `(anscombe(counts) - mean) / sd` using those numbers, so a manifest that disagrees with its own counts mis-scales and mis-centres everything the encoder sees, silently.

**Three Laue manifests did disagree.** `fix_manifest_stats.py` recomputed them over every mask-valid pixel: 1118 hewl3 and 1118 hewl360 were wrong by 1.85x in sd, 1117 by 1.47x.
PDZ2 and the monochromatic 9b7c agreed with their counts and were not touched.
The wrong values were preserved twice, as a comment block in the rewritten manifest and as a timestamped backup, **because a checkpoint trained before the fix needs the old statistics for inference**.
So a predict pass on a pre-fix checkpoint of 1118 or 1117 must be run against the old statistics; a predict pass on PDZ2 or 9b7c needs no such check.
`pooled/dataset.yaml` for PDZ2 has no comment block, which is how to tell it was never rewritten.

### Training

`integrator.train --config <yaml>` resolves the YAML to dataclasses through `REGISTRY` and `construct_integrator`.
A new configuration key defaults off, so an existing run stays bit-identical -- add, never change or remove.
A new configuration is tested through the factory path, not by reading it.

Training is **unseeded**: two trainings of one configuration differ by about 0.0050 in R-free and 0.86 $\sigma$ in mean peak height, so an $n=1$ gap below roughly 0.008 is noise.
`laue.integrate` is deterministic and carries no such term, which means a model-against-reference comparison has one noise source on only one side.

### Export, and the two estimators

One training gives **two** sets of intensities and sigmas, and which one wins is not monotone across arms, so both are scored:

- `model` -- the integrator's posterior, `qi_mean` and `qi_var`.
- `rb` -- the Rao-Blackwellized profile fit computed at export, `i_rb` and `var_rb`, from `integrator.predict --rb-export`.

`rao_blackwellized_intensity` in `model/integrators/integrator_utils.py` is the variance-weighted profile fit $I = \sum_p w_p (c_p - b)$ over the model's own profile and background, with two things the classical form omits.
It fits once per profile sample and adds the spread of the fits to $\sigma$ by the law of total variance, rather than conditioning on one profile; and its conditional term is the Schur complement of the $(I,b)$ information rather than $1/F_{II}$, so it charges for the background being estimated.
Negative intensities survive: only the rate inside the weights is floored, at the background, because a negative $\hat I$ is a legitimate estimate but an impossible rate.

**Which columns are written is a configuration choice, and it is the one that cannot be undone later.**
`predict_keys` in the integrator's arguments names them.
No polychromatic run has ever included `qp_mean`, so the profile is absent from every Laue prediction parquet and the shoebox-against-profile figure cannot be drawn for any of them without a second predict pass.
The monochromatic parquets carry it.

### What is logged during training

| quantity | where | state |
|---|---|---|
| ELBO and its terms per epoch | `plots/loss_history.csv` | every run, both arms |
| Wilson scale and $B$ per epoch | `plots/wilson_params.csv` | every run; `epoch, B, G_mean` |
| the learned spectrum $G(\lambda)$ | `plots/wilson_spectrum.csv` | polychromatic runs |
| every `self.log` metric | `plots/metrics.csv` | every run |
| the profile basis per epoch | `figures/basis_snapshots.npz` | **absent**: `ProfileBasisLogger` is gated behind `integrator.train --figures` and no production run passed it |
| tracked shoeboxes, latents | `figures/` | same gate, same absence |

`loss_history.csv` has **train and val only**.
The test reflections are the exported ones, so a test loss needs an evaluation pass rather than a column, and the figure that is written to draw three splits draws two.

The Wilson numbers are read by `WilsonParamLogger` from the loss's own `get_G()` and `get_B()` each epoch, not from a stored copy, so they cannot drift from what the prior actually used.
Where there is more than one scale group it also writes `G_min` and `G_max`, because the spread across groups says how much the crystals differ and a single mean would hide it.

The profile basis is a parameter of `surrogates.qp.decoder`, so **every checkpoint holds it** and the trajectory is recoverable after the fact even though the logger never ran: `scripts/figures/dump_basis.py` reads the state dicts directly and writes the same `basis_snapshots.npz` and `basis_diagnostics.parquet` the logger would have.
It needs no data, no GPU and no retraining.
`learned_basis_profile` and `geometric_basis_profile` both decode through an `nn.Linear` and so both have a basis that moves; `shapelet_profile` and `physical_profile` evaluate an analytic family and `fixed_basis_profile` is frozen by construction, so for those a snapshot over epochs is a constant.

## Monochromatic

### The chain, stage by stage

`scripts/mono/cannon/run_all.sh` submits four jobs. Stage 3 is the one that produces every reported number, and `03_pipeline.sh` runs it in this order:

1. `integrator.predict -v --run-dir <run> --ckpt <ckpt> --write-refl`, once per selected checkpoint.
2. `scripts/mono/post_config.py --run-dir <run> --process-cfg <cfg>`, which writes `dials_phenix_cfg.yaml` listing the reflection files to process.
3. `scripts/mono/process_single.py --config <cfg> --index <i>`, once per file, which is where DIALS and phenix run.
4. `scripts/mono/emit_merging_stats.py --run-dir <run>`.

Inside `process_single.py`:

```
dials.scale  <refl> <expt>  output.{reflections,experiments,html,log}=...
dials.merge  dials_scaled.{refl,expt}  output.{log,html,mtz}=...
phenix.refine <the dataset's own .eff, with the merged MTZ substituted in> overwrite=true
rs.find_peaks *[0-9].mtz *[0-9].pdb -f ANOM -p <the phase label the .eff asked for> ...
```

`dials.scale` and `dials.merge` run at their **defaults**; nothing in the chain overrides a scaling or merging parameter.
The refinement is not at defaults: it uses the `.eff` that belongs to the dataset, named by `PROCESS_CFG`, and the phase label passed to `rs.find_peaks` is read back out of that `.eff` by `integrator.utils.phenix.requested_phase_label` rather than hardcoded, because phenix 2.0 writes `PANOM` where 1.20 wrote `PHANOM` and naming one would be wrong half the time.

**The refinement chain is attached to the dataset and must stay fixed.**
phenix version, the `.eff`, the R-free flag source and the pass count each move R-free by as much as the effects being measured.
9b7c scored through a different chain gave 0.1602 and silently discarded its anomalous data.

The reference arm for a monochromatic dataset is the same `dials.scale`/`dials.merge`/`phenix.refine` chain run on the reference integrator's own reflection file, so the model and reference differ only in the intensities.

A merged MTZ carries both `F(+)/SigF(+)` and `I(+)/SigI(+)`.
**Both are refined**, every time, and a gap between them is an export defect rather than a result: that check is what located a float32 cancellation presenting as a 6.5x anomalous-peak deficit.

### HEWL 9b7c -- iodide SAD

Configuration `configs/mono/hewl9b7c_mono_expmean_lp.yaml`, run `105gh86k`, 50 epochs, 0.40M parameters, scored at epoch 49.
Shoeboxes are 3 slices of 31x31; the basis decoder is `(2883, 12)`, so twelve components over a 3x31x31 box.
Knobs that matter and were changed together, so the win is not attributed to any one of them: `mean_fano` with `positive_constraint: exp`, `lp_correction: true`, `decoder_weight_decay: 0.0`.

`lp_correction: true` is the DIALS convention: DIALS applies the Lorentz-polarization correction during scaling, so the model must not apply it again.

Scored through 9b7c's **own** `phenix.eff`.
Endpoint: R-work and R-free on the deposited model, plus anomalous peak heights at the iodide and sulfur sites, over 13 sites shared with the reference.

### SBGrid 821, 845, 834, 140 and 6VYN

Ingested by the framework in `scripts/sbgrid/`: `fetch.py` for the raw images, `dials_reference.py` for the reference integration, `combine_sweeps.py` where a dataset has several, `refine.py` for the refinement and anomalous peaks, `reference_stats.py` and `summary_table.py` for the comparison.
`characterize.py` and `shoebox_size.py` set the window from the data rather than by assumption.

Per-dataset specifics -- which sweeps were kept, the resolution limits, the beta used -- are recorded in `results_by_dataset.md` and `reference_status.md` and were **not** re-read from the scripts for this document.
Two things about them are worth carrying here because they are methods, not results.

845's post-integration processing was verified identical between arms: the same `dials.scale` and `dials.merge` parameters, the same `7RIN.updated.pdb`, the same 1.87 Å limit, the same 0.998 completeness, and reflection counts that reconcile.
So its gap is in the intensities and is present at the start of refinement.

140 is **broken**, not weak: 0.584/0.606 against a deposited 0.166/0.221, from a single-sweep bug.
4U4H's deposited structure factors carry no `I(+)/I(-)`, so its anomalous reference has to come from reprocessing rather than from the deposit.

## Polychromatic

### The reference integration

`scripts/poly/laue_process.sh` reproduces the rs-station laue-dials tutorial command sequence for one sweep:

```
dials.import -> laue.find_spots -> laue.index -> laue.sequence_to_stills
  -> laue.optimize_indexing -> laue.refine -> laue.predict -> laue.integrate
```

It writes both halves of the comparison from one run: `predicted.refl` plus the refined stills, which is what `refltorch.mksbox-laue` consumes, and `integrated.mtz`, which is the laue-dials baseline.
Every crystal- and geometry-specific value is a parameter, so one script runs any sweep: `OSC` (scan oscillation start and step), `SG`, `CELL`, `LAM_MIN`, `LAM_MAX`, `DMIN`, `GAIN`, `IMAGE_RANGE`, `INVERT_ROT`, `INDEX_METHOD`.
Defaults are DHFR / 1117: `OSC=0,1`, `SG=19`, cell `31.773,42.656,93.465,90,90,90`, $\lambda$ 0.9--1.2 Å, `DMIN=1.4`, `GAIN=0.30`.

Two operational facts are baked into the script and both change the outcome.
`OMP_NUM_THREADS=1` and the other BLAS pins are **required**: `laue.predict` forks a multiprocessing pool immediately after training a scipy KDE, `fork()` copies memory but not threads, and the BLAS locks the KDE left held in the parent deadlock every worker's first call -- 11.5 hours to 92 seconds.
And each step runs only if its final output is missing, so a walltime timeout resumes rather than redoing the slow predict.

**What `laue.integrate` does and does not produce.** Read off `laue_runs/1118_hewl3/integrated.refl`:

```
BATCH  H  K  L  PARTIAL
background.sum.value
intensity.sum.value  intensity.sum.variance
s1.0  s1.1  s1.2
xyzcal.px.0  xyzcal.px.1  xyzcal.px.2
```

There is **no profile-fitted intensity**, so `intensity.prf.value` is a monochromatic quantity and the polychromatic model-against-reference comparison is against the summation integrator only.
There is a background, but it is a **sum over the region integrated**, not a per-pixel mean, so it is not on one axis with the model's `qbg_mean` without the pixel count, which is not exported.
`laue.integrate` v0.5.1 sums flat inside an ellipse whose area varies per reflection, so a fixed shoebox volume is the wrong divisor.
The correlation between the two is available today; a scatter on equal axes is not.

`laue.integrate` is **not** naive summation: it is the classical variance-weighted profile fit with profile borrowing from strong nearest neighbours and Poisson optimal weighting, so it is near-optimal on exactly the strong reflections that carry an iodide signal.
Its iteration is positivity-clamped at two places in `algorithms/integration.py`, so its fixed point is structurally non-negative and it emits no negative intensity on any dataset checked -- 0.00% across 1117, 1118 and 1116.

### The model chain

`scripts/poly/run_pipeline.py --run-dir <run> --steps predict,careless,stats,phenix,peaks`, reading `poly_pipeline_cfg.yaml`.
Every value below is from that file.

**careless**, config 4, the anomalous plus positional-encoding recipe, `dmin: 1.5`, flags from `scripts/poly/careless_configs.py`, output into `<predictions>/epoch_XXXX/scaling/config4_*`.
`seed: null` takes careless's default; set it for reproducibility.
The careless **merging seed dominates the refinement seed about ninefold** on Laue data, so an arm comparison has to average over merging seeds, not refinement seeds.
Three seeds give a standard deviation of about 0.0012 in R-free and 0.012 in CCsym; that is the floor a difference has to clear.

**stats**, every `careless.stats` tool, each to its own CSV plus a long-format `careless_stats.csv` carrying `method`, `dataset`, `epoch`, `tool`, `statistic`, `bin`, `shell`, `group`, `value`.
The three careless outputs are **not interchangeable** and handing a tool the wrong one raises a `KeyError` on the column it wanted:

| tool | reads | why |
|---|---|---|
| `cchalf`, `ccanom`, `rsplit` | `<base>_xval_0.mtz` | carries `repeat` and `half` |
| `ccpred` | `<base>_predictions_0.mtz` | carries `test`, `Iobs`, `Ipred` |
| `isigi`, `completeness` | `<base>_0.mtz` | the merged result |

`isigi` has no default intensity key and raises `KeyError: None` unless one is named: an anomalous merge carries `I(+)`/`I(-)` and never a plain `I`.
All careless output for a run sits under `predictions/epoch_0000/` whatever epoch the checkpoint was, so the step needs `--epoch 0` on the `ablate2` runs or it looks under `epoch_0019` and finds nothing.

**phenix**, two passes, through the phenix 2.0 install at `/n/lab_storage/hekstra_lab/garden_2/phenix-2.0-5729` -- an install prefix, not an env script, because 2.0's `phenix_env.sh` exports the pre-migration prefix and leaves nothing on `PATH`.
`eff1` and `eff2` are the data-manager form of `laue_analysis/custom_refinement_param_{1,2}_.eff`, which 2.0 rejects; same cell, strategy, macro-cycles, generated R-free set and map labels, so only the version differs.
Refinement starts from `reference_data/shaken_9b7c.pdb`, the **monochromatic** 9b7c model, which is correct for 1118.

**`rfree: null` is a reproducibility defect that is live.**
Unset, the `.eff` files generate a fresh 10% R-free set per MTZ, and `refinement.main.random_seed` moves that draw as well, so two runs of one arm differ by which reflections were held out -- about 0.008 in R-free on weak Laue data -- on top of differing by the data.
The fix is in the tree and not switched on: set `rfree` to the mono chain's `reference_data/rfree.mtz` and point `eff1`/`eff2` at `laue_refine_pass{1,2}_fixedfree.eff`, which read `$FREEFILE`.

**peaks**, both methods: `rs.find_peaks -f ANOM -z 5.0` writing `peaks.csv`, and `anomalous_peak_heights.py` with `elements: "[S,I]"` against `reference_data/9b7c.pdb` writing `peak_heights.csv`, plus `careless.ccanom` on the cross-validation MTZ.
`phi` is left null and the phase label is read off the MTZ for the same phenix-version reason as the mono arm.

**`rs.find_peaks` writes an empty `name` column on every row** in the fork in use, so the atom carrying each peak is resolved against the refined PDB beside the peak list, with gemmi, and taken only where its chain, residue number and residue name agree with what the peak search reported.
The search must exclude hydrogens -- `NeighborSearch(...).populate(include_h=False)` -- or a riding hydrogen is named as the anomalous atom; the real 9b7c list produced `ILE 88 HG12` at 10.96 $\sigma$ before that, and `ILE 88 CG1` at 0.98 Å after.
The `cen*` columns in that fork's output are wrong and `coord*` must be read instead.

### HEWL 1118 -- the experimental benchmark

NaI-soaked HEWL, BioCARS, P4$_3$2$_1$2, cell 79.4 79.4 37.8.
**Frame counts are not interchangeable**: the dataset exists at 360, 1000, 1440 and 3049 stills and the iodine peaks differ by 5--6 $\sigma$ between 360 and 1440, so pairing a 360-frame reference with a 1440-frame model invents a result.
Every comparison is frame-matched and the frame count is stated with the number.

Best configuration family `screen_l8_k7_exp`: latent 8, first kernel 7, `mean_fano` with exp mean, `stabilize_scale`, `init_G 5000`.
Kernel size plateaus at 7 because it tracks the 5--10 px spot and not the 25 px shoebox.

The published Laue CC$_{1/2}$ of 0.976 at 1.7 Å is reproduced by the laue-dials plus careless reference arm, and that is the project's one clean published-number replication.
A current run of the reference chain reproduces IOD 204 at 20.77 against 21.1, IOD 206 at 19.87 against 19.7, and R-free 0.1992 against 0.217, which makes it a validated baseline rather than an assumed one.

The 1000-frame ablation (`docs/ablation_1118_1000.md`) is the current model-selection table: 18 arms crossing `svae` and `hierarchical` with five profile surrogates, R-free and mean peak height with careless-seed error bars on each.

### SBGrid 1117 -- DHFR

No deposited PDB, so refinement is not a valid reference and the comparison is CC$_{1/2}$ through identical careless: config 1, `dmin` 1.4, 30,000 iterations.
MR of 1RX2 produced no solution, so that route is closed rather than pending.

### PDZ2 / 1116 -- electric-field TR-Laue

The one dataset whose endpoint is **CCsym** rather than anomalous peaks, and the one whose chain is a deposited pipeline reproduced verbatim rather than our own.

**The data.** LNX2 PDZ2, the Nature 2016 EF-X study, raw `.mccd` under `sbgrid/1116/`.
Four passes `e35c/d/e/f` times four conditions `off/50ns/100ns/200ns` = 16 sweeps.
Passes c, d, e are 2°-spaced and f is 1°-spaced; f is the strong one at 181 images, c and d about 46, e about 91.
Reference off-state structure is PDB 5E11 in the same crystal form as the data, so no molecular replacement is needed; the field-on structure 5E22 is in P1, the primitive cell of the same lattice.

**The reference chain, and the five things we got wrong the first time.**
The chain is the laue-dials tutorial's, downloaded to `~/Downloads/laue_tutorials/pdz2/` and copied verbatim into `laue_runs/pdz2_ref/scripts/tutorial_orig/`; the adapted copies differ only in partition, walltime, environment and BLAS pinning.
Our first attempt copied the command order but missed, in order of impact:

1. **Delays reuse the off-pass geometry.** `one_pass-from_off.sh` seds the off pass's `stills.expt` with the delay image names. We indexed all 16 sweeps independently, so same-crystal cells disagreed -- e35c β ranged 117.3° to 122.4°.
2. **`expt_concat.py` drops slipped and misindexed images**, by BATCH: pass e keeps $\le 85$, pass f keeps $\le 95$, which is half of f. That is why the strong pass alone had failed.
3. P1 copies are written alongside the C2 ones, change of basis `a+b,a-b,-c`, plus `_shared_index` versions.
4. **One joint careless over eight files** with `--separate-files`, not one per condition.
5. Per-pass oscillation is c `0,2`, d `92,2`, e `181,2`, f `361.5,1`; we had used `0,1` for all. And pass e needs `dials.reindex -x,y,-z`, which the tutorial's own `process.sh` skips through a `[ pass == e ]` typo while `process-slurm.sh` has it right.

With those in place the reference reproduces the Laue-DIALS paper's result: CCsym in the first bin 0.190 / 0.357 / 0.371 / 0.415 for off / 50 / 100 / 200 ns against the published 0.19 / 0.33 / 0.38 / 0.40, and the weighted 200 ns difference map reproduces all ten published peaks in sign.
**So the earlier conclusion that our reprocessing could not reach the ~1% field signal was our processing error and is withdrawn.**

**Pass d needs a workaround.** With laue-dials 0.5.1 and DIALS 3.26, pass d at the tutorial's start angle of 92° is misindexed from frame 1 -- median 15.6 px RMSD, 0 of 44 frames under 2 px -- which is laue-dials issue #49, still open.
Mono indexing of the same images is fine, so the fault is in `laue.sequence_to_stills`; the imported step is stored as 2.0000000000000027, which looks like a float floor.
Importing pass d with start 0 removes it entirely, 35 of 45 frames good, and gives an identical Miller setting (2850 of 2850 spots) so no reindex is needed.
Passes c, e and f are fine at 0.47 / 0.28 / 0.78 px median over the tutorial's kept frames, and e degrades after frame 85 and f after 95, which is what the tutorial's cuts are for.

**The merge.** One `careless poly` call over eight files, from `careless-cdef-ohp-mlpw.sh`:

```
--iterations=12_000 --dmin=1.7 --separate-files --test-fraction=0.1
--positional-encoding-frequencies=5 --positional-encoding-keys="xcal,ycal"
--wavelength-key=wavelength --studentt-likelihood-dof=64
--merge-half-datasets --half-dataset-repeats=3 --mlp-width=32
"Hobs,Kobs,Lobs,xcal,ycal,wavelength,dHKL,is_c,is_d,is_e,is_f,BATCH"
```

No `--anomalous`, so the merged MTZ carries plain `I`/`SigI` and `careless.ccanom` has nothing to correlate.
The eight inputs are, in order: off, 50 ns, 100 ns, 200 ns in C2 with a shared index, then the same four in P1.
So **`merged_e35_0.mtz` is the field-off C2 merge** -- the file the refinement endpoint uses, already matching `5e1y.pdb` -- and the field response lives in indices 4 to 7.

**CCsym.** `custom_ccsym.py merged_e35_xval_[4-7].mtz --spacegroup=5 --op=1 -b 10 --cb_op=a+b,a-b,-c`, from `stats_for_production.sh`.
It correlates $F(h) - F(\text{2-fold}\cdot h)$ between the two half-datasets, in **P1**, over three cross-validation repeats.
The statistic is **Spearman**: that is the script's default and the production script passes no `-m`.
Three properties of its output trip up a reader: it **doubles the bin index** before saving, so ten bins are numbered 0, 2, ..., 18; it saves **no resolution at all**, the shell edges existing only as tick labels on its SVG; and it carries a comment flagging a known bug in how repeats interact with the P1 expansion.
The deposited `stats_for_production.sh` also leaves `careless.cchalf` commented out, so CC$_{1/2}$ was never computed for PDZ2 by anyone until this project ran it.

**Raw CCsym is not a signal measurement.**
`laue.integrate` reads 0.10--0.20 in the **field-off** state where the true answer is exactly zero, so much of any raw value is reproducible non-field asymmetry and only the response means anything.
Splitting halves by image also lets a method borrow across the split, which rewards exactly the neighbour profile borrowing `laue.integrate` does by construction.
The recommended estimator is `CC_Delta`: form the per-reflection field response inside each half and correlate those, reporting $2CC/(1-CC)$.
Normal-theory standard errors understate by about 1.8x from heavy tails, so significance needs a paired reflection bootstrap.

**The difference map.** `wdm.py`, rs-booster `rs.scaleit` plus `rs.diffmap`, 200 ns P1 against off P1, phases from PDB 5E1Y through `phenix.fmodel`, $\alpha = 0.05$, cut to 1.8--5 Å.
Peaks are measured at the twenty sites fixed by the deposited model -- three residues times two conformers times two symmetry mates -- by `wdm_peaks_compare.py`.
Peak heights at positions fixed by a deposited model cannot be inflated by profile borrowing, which is why this and not CCsym is the endpoint to weigh.
The map is in primitive P1 and 5E1Y is in C2, so the fractional transform is chosen by requiring distance preservation, validated against 5E1Y's own primitive cell; the data cell is 0.4% larger.

**The refinement endpoint is ours, because the deposited pipeline has none.**
`reduce.sh` and `stats_for_production.sh` run only careless, `custom_ccsym.py` and `wdm.py`, and `one_pass-from_off.sh:87` says so outright.
We refine the field-off state, which has full C2 symmetry, against `merged_e35_0.mtz`, with `pdz2_refine_prep.py`: one R-free set of 404 reflections (4.5%) generated once over the union of Miller indices and applied to **every** arm, so arms are compared on the same held-out reflections, and the model re-celled to the data (65.261/39.450/38.961/β117.456 against 64.912/39.288/38.800/β117.410, the 0.5% isomorphous mismatch that also forced `check_isomorphous=False` in the deposited phase transfer).
Three macro-cycles on `F,SigF`.
All arms came out with identical 8963-reflection sets.

**The model arm.** Shoeboxes from `refltorch.mksbox-laue`, which needs `--max-images` and works from `predicted.refl` plus `poly_refined.expt` with no `integrated.refl`, pooled by `pool_pdz2.py` to 993,143 reflections as 21x21 single slices with per-sweep row ranges in `sweep_ranges.yaml`.
Trained from `configs/poly/pdz2_1116*.yaml`, derived from `dhfr_1117.yaml`.
Split back out by `split_pdz2.py`, and **the predictions are positional**: the parquet `refl_ids` are each sweep's own ids and restart per sweep, so a join on id across sweeps silently mixes reflections.
Then the identical reference downstream -- same geometry, same frame cuts, same `expt_concat`, same careless flags, same CCsym -- so integration is the only variable.

**The BATCH convention differs between the arms and corrupted ours once.**
laue-dials writes `BATCH = refls["id"] + 1`, the dense DIALS experiment index into the list given to `laue.integrate`; we write `BATCH = image_num + 1`, the true image number from the filename.
They coincide only when nothing was dropped upstream, and where refinement dropped experiments laue's index compresses and ours does not, so the difference drifts by -4 to +3 depending on the sweep.
Since `expt_concat.py` applies the tutorial's frame cuts **by BATCH**, we cut the wrong frames.
Our identifier is arguably better but wrong for this pipeline.
The provably correct post-hoc repair is a join on `(H, K, L, round(xcal), round(ycal))`, which matches 100% in all 16 sweeps with purity 1.0000 and 0 ambiguous; BATCH is not usable as a join key.

**The arm inventory and the endpoint numbers** are in `results_by_dataset.md`.
The methods point is that 18 arms were trained, all on `integrator: hierarchical`, and that the export-only arms (`k7m_ml`, `k7m_mlclamp`, `combo`, the four swap trees) are not retrainings: they rewrite a trained arm's intensities with `ml_from_preds.py` or `build_swap_trees.py` and re-run the identical downstream.

### The nanoBragg DHFR simulation

Synthetic Laue stills from Derek Mendez with per-pixel ground-truth Miller index, wavelength and orientation.
Configurations `configs/poly/dhfr_sim_nanobragg.yaml` and `dhfr_sim_geometric_control.yaml`.
`HDF5_USE_FILE_LOCKING=FALSE` is required to read them, and the CBFs carry no rotation.

Two things about the truth that affect how it is used.
The simulation **does** have shot noise, $\mathrm{Var} = 0.173\mu + 7.10$, which is gain 0.7 through a PSF with readout sd 2.7, so the likelihood is not the cause of a slope deficit.
And "the truth-mask sum" and "the truth-profile fit" are two different estimators with a 3x variance gap -- a flat sum over the pixels whose true Miller index matches, against $\hat I = \sum p(c-b)/\sum p^2$ -- so which one a comparison used has to be named.
The truth-mask sum is **not** a precision ceiling: over 570 flat pixels it is 2.25x noisier than `laue.integrate`.

`scripts/simulate_laue_merging.py` is the separate small synthetic: a Laue stills dataset in the real loader's format with the truth beside it, 50k observations on 250 stills generated in 1.8 s, six epochs in under three minutes on a laptop, scored by `scripts/score_simulated_merge.py`.
It is calibrated against `hewl_1118` on what is known and deliberately mismatched on crystal-frame absorption and tilted-plane background so that it does not flatter the model.

## The model comparisons

Three comparisons the project reports, with how each is measured and what
makes a number in it valid.

### svae against hierarchical, at a fixed profile family

The comparison is only interpretable when the two arms differ by the
integrator family **alone**, so the pairs are chosen on that basis rather
than on which arms happen to exist.

| dataset | svae arm | hierarchical arm | `qp` | basis |
|---|---|---|---|---|
| hewl_9b7c | `mono9b7c/hewl9b7c_svae` | `mono9b7c/hewl9b7c_mono_expmean_lp` | `learned_basis_profile` | (1323, 12) |
| hewl_1118 | `full1000/learned` | `hier1000/softmax_std` | `learned_basis_profile` | (625, 12) |
| pdz2_1116 | `runs/pdz2_svae` (training) | `runs/pdz2_w21_k7m` | `learned_basis_profile` | (441, 16) |

A config pair differs in exactly two places and nothing else: the
`integrator.name` line, and the intensity encoders -- the hierarchical model
takes four (`k_i`, `r_i`, `k_bg`, `r_bg`), svae takes one and derives
$q(I)$ by a conjugate Poisson-Gamma update.
`configs/poly/ablation_1000/softmax_std.yaml` against
`configs/poly/ablation_1000/hier/softmax_std.yaml` is the reference diff, and
`configs/poly/pdz2_1116_svae.yaml` was derived from `pdz2_1116.yaml` by making
the same two changes.

**Both configs of a pair are built through `construct_integrator` before the
run is submitted**, never checked by reading. That is what caught the PDZ2
svae config asking for one encoder where the integrator requires two; a read
would have passed it and the failure would have appeared in the queue.

The basis itself is compared by `scripts/figures/dump_basis.py`, which reads
`surrogates.qp.decoder.weight` out of each checkpoint. The two arms of a pair
have the same basis shape, so their components are directly comparable, and
`basis_components.csv` carries each component's norm and its cosine against
its own final value.

### Parameter counts and training throughput

`scripts/figures/compare_models.py`, over run directories. Neither number
needs a rerun.

**Parameters** are summed from the checkpoint's state dict, per top-level
module as well as in total. Counting from the state dict rather than by
constructing the model counts what was actually trained, and needs no config
resolution, no GPU and no data. The per-module split is the useful half: two
families can match on the total and differ entirely in whether the mass sits
in the encoders or the decoder.

**Training throughput** is the median seconds per epoch, from the checkpoint
file timestamps -- consecutive checkpoints are a fixed stride apart, so the
gap divided by the stride is seconds per epoch. The median rather than the
mean, because a requeued or preempted job leaves one enormous interval that a
mean would absorb and report as a slow model. The number of intervals is
reported beside it, so a run with too few to mean anything is visible.

Two limits, both of which make a number invalid rather than noisy:

- **Seconds per epoch compares only runs on the same hardware and the same
  dataset.** Two arms of one ablation are comparable. An arm on 9b7c against
  an arm on 1118 is not, and neither are two runs that landed on different
  nodes. The run directories hold no job logs, so the GPU is usually not
  recoverable and the comparison rests on the two arms having been submitted
  together.
- **Inference time is recorded nowhere.** It needs a predict pass with a
  clock around it on fixed hardware, and no such measurement exists.

### PDZ2 against `laue.integrate`

Three endpoints, and they do not agree, which is the point rather than a
problem. All three come from tables in `docs/figures/data/`.

| endpoint | file | what it is |
|---|---|---|
| CCsym by delay | `pdz2_ccsym.csv` | the field response; **confounded**, see below |
| field-off refinement | `pdz2_refinement.csv` | R-work and R-free, the non-gameable one |
| per-shell merging | `careless_stats_pdz2.csv` | CC-half, R-split, CCpred, $I/\sigma_I$ |

The reference arm is the row with `method = laue`, built by running the
deposited chain verbatim; every model arm swapped its intensities into the
same 16 `integrated.mtz` slots and ran the identical downstream, so
integration is the only variable.

**Rank the arms on R-free, not on CCsym.** CCsym is a half-dataset
correlation and `laue.integrate` borrows profiles from strong nearest
neighbours, which feeds both halves and inflates it -- the same confound as
the CC-half profile-borrowing artifact. Peak heights at positions fixed by a
deposited model, and R-free on a held-out set shared across arms, cannot be
inflated that way. The off-state column is the control and is read, not
subtracted: `laue.integrate` reads 0.067 there where the true value is
exactly zero.

A difference between arms has to clear the **careless merging seed**, which
moves CCsym by about 0.012 and R-free by about 0.0012, and dominates the
phenix seed roughly ninefold on Laue data. Arms are therefore averaged over
merging seeds, and `replicate` in `pdz2_refinement.csv` is that seed.

### Exporting the Rao-Blackwellized intensity

`integrator.predict --rb-export --write-mtz` on an existing checkpoint. No
retraining, and the two traps are both silent until the very end.

**`--rb-export` only tells the MTZ writer which columns to read.** The
parquet's columns come from `predict_keys`, so a config that does not list
`i_rb` and `var_rb` produces a parquet without them and the export dies on a
`KeyError` after the whole predict pass has already run.
`scripts/poly/rb_predict_config.py` derives a predict config with the four
RB columns added.

**It derives that config from the run's own `config_log.yaml`, never from
the repository config.** The checkpoint has to load into the architecture it
was trained with, and the PDZ2 arms differ from each other and from the
repository file in window and kernel size; the repository `pdz2_1116.yaml`
has a 3x3 first kernel where `pdz2_w21_k7m` trained a 7x7.

**The console script on the cluster is not the current code.**
`integrator.predict` resolves to the `lab_storage` checkout, which predates
`--rb-export` and fails on an unrecognised argument. The current `src/`
is staged into the job directory and put first on the path with `python -m
integrator.cli.predict`, rather than changing a shared checkout that other
jobs may depend on. That is a workaround for the stale-checkout problem
below, not a fix for it.

## What is not reproducible from the repository today

Verified against the repository and both cluster checkouts on 2026-11-18.
These are reproducibility defects, listed so they can be fixed rather than rediscovered.

1. **The PDZ2 chain is not in the repository.** About 110 scripts exist only in `/n/netscratch/.../laue_runs/pdz2_model/`, including `pool_pdz2.py`, `split_pdz2.py`, `split_pdz2_ml.py`, `ml_from_preds.py`, `build_swap_trees.py`, `wdm_peaks_compare.py`, `pdz2_refine_prep.py`, `collect_rfree.py`, `unmerged_ccsym.py` and every `.sbatch`. There is no PDZ2 pipeline configuration and no figure script that reads CCsym. That filesystem has already purged this project's raw images once.
2. **Three cluster checkouts exist and none is current.** `$S/integrator_plumbing/src` is the one `pull_figdata.sh` puts on `PYTHONPATH` and the most complete: it has `checkpoints.py` and `metrics.py`, but its `metrics.py` predates the `dials_log` split and the peak-atom resolution, so anything reading a `dials.scale` table or naming the atom under a peak still fails there. `$S/integrator_cur/src/integrator/utils/` has neither `checkpoints.py` nor `metrics.py`, so `run_pipeline.py` fails at import against it; the `lab_storage` checkout, which `scripts/mono/cannon/env.sh` points `INTEGRATOR_ROOT` at, has less again. The chain is currently run by staging the needed modules into the job directory and putting that first on `PYTHONPATH`. That is a workaround, not a fix: one checkout should be the one that runs.
3. **The polychromatic R-free set is redrawn per MTZ.** `rfree: null`; see the phenix paragraph above. The fix exists in the tree and is not switched on.
4. **careless and training are both unseeded by default.** `seed: null` for careless; training has no seed at all. The error bars quoted on polychromatic arms are careless-seed only and do **not** include training variance, which is real.
5. **No run has the basis or tracked-shoebox dumps**, because `integrator.train --figures` was never passed. The basis is recoverable from the checkpoints; the tracked shoeboxes and latents are not, since they depend on which reflections were selected at the time.
6. **`qp_mean` is in no polychromatic `predict_keys`**, so no Laue run can be shown as shoeboxes against its own predicted profiles without a second predict pass.

## Where the rest is written down

`exporting.md` -- export paths, column modes, ASU tables, the SigF defect, refinement chains.
`downstream_stats.md` -- the shared per-shell contract and which tool computes each quantity.
`results_by_dataset.md` -- what the numbers came out as, per dataset and per arm.
`ablation_1118_1000.md` -- the current polychromatic model-selection table.
`model_selection.md` -- which dataset answers which question, and which are valid references.
`pitfalls.md` -- the failures that produced plausible wrong numbers.
`pdz2_training.md` -- how to launch a PDZ2 training run and how to read its result.
`figures/README.md` and `figures/coverage.md` -- the figure set, and what a run must produce before each figure can be drawn.

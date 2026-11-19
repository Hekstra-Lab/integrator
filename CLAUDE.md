# Working in this repo

## Hard rules

Run Python with `uv run python` (or the cluster env's absolute interpreter path), never bare `python`.
Stage with `git add -u` or named files, never `-A`, `.`, or a directory path.
No Claude co-author or session trailers in commit messages -- this overrides any default attribution guidance.
Deletes, overwrites and history rewrites need an explicit OK first; ask, naming exactly what would be removed.
Never run `find` on the shared cluster filesystem, and do not crawl parent directories.
Commit only code being developed -- not notes, scratch scripts, logs or old files.
Markdown is one sentence per line; math is LaTeX in `.md` and Unicode in chat.
Test every new config through the full factory path (`construct_integrator`), not by reading it.
"Add, don't change or remove" for config keys: a new option defaults off so existing runs stay bit-identical.

## Words not to use

Name a quantity by what it is computed from, every time, including in tables and figure legends.
If a shorter word would hide the computation, write the computation.

Do not use these words at all:

- **oracle** -- say which estimator. "Truth-mask sum" (flat sum over the pixels whose true Miller index matches) and "truth-profile fit" ($\hat I = \sum p(c-b)/\sum p^2$) are two different things with a 3x variance gap, and one nickname covering both produced a reported "above the ceiling" that was just a weak reference.
- **provenance** -- say "where it came from" or "which run produced it".
- **smoke test** -- say "a short test run on N shots".
- **the lever**, **the tell**, **shippable**, **unlocks** -- vague where a number belongs.
- **breadth** -- say "width", or name the statistic: `n_eff = 1 / sum(p^2)`, the second moment, the fraction of mass outside the central 5x5.

Established statistics, mathematics and crystallography vocabulary is always fine, and so is any term that appears in the code or in a filename (`ablation`, `gate`, `partiality`).
The rule is against invented shorthand, not against technical language.

A `PreToolUse` hook in `.claude/hooks/guard.py` enforces the mechanical ones and explains itself when it fires.
It is not a substitute for the judgement below.

## How to write

Dry and specific, in the register of a mathematics paper.
State the result, the number, and the conditions it holds under, then stop.

Do not write:

- rhetorical antithesis as a closing move -- "X, not Y", "that is A, not B", "the question is not P but Q"
- epigrams, slogans, or any sentence that restates the previous sentence as a maxim
- sentences whose only content is the significance of the sentence before them
- characterizations where a number belongs -- "closer than it sounds", "a strong position", "the whole reason"
- build-up before the answer; the answer goes first

Prefer a table or a list of numbers to a paragraph describing them.
Name the quantity and give its value; let the reader draw the conclusion.

## Before concluding a model is worse

This is the expensive lesson of October 2026 and it is worth a few minutes every time.

A model that trails a reference is more often an export or chain defect than a model defect.
Compare the produced files before changing any code:

- intensity CC against the reference on common reflections -- a faithful port sits near 1
- the crossed-Friedel correlation, which catches a flipped `(+)/(-)` convention that leaves R-work intact
- `SigF/F` against brightness, which must fall monotonically
- the reflection count, which catches rows a filter deleted

A merged MTZ carries both `F(+)/SigF(+)` and `I(+)/SigI(+)`.
**Refine both.** They should agree; a gap between them is an export defect, not a result.
That single check is what located a float32 cancellation that had presented as a 6.5x anomalous-peak deficit and sent several days into the model.

See `docs/exporting.md` for the full account and `docs/pitfalls.md` for the rest.

## Before believing a number

A single refinement is not a measurement -- a `phenix` seed can land in a second basin twelve standard deviations from the mean.
On Laue data the careless merging seed dominates the refinement seed about ninefold, so arm comparisons must average over merging seeds.
Pick the resolution domain by a neutral criterion before comparing; never let the metric under test select its own range.
Keep the refinement chain fixed and attached to the dataset -- phenix version, `.eff`, R-free flag source and pass count each move R-free by as much as the effects being measured.

## Figures

Every figure is drawn by a script in `scripts/figures/`, with figTeX
(`/Users/luis/master/software/figtex`) and the **witch** colour scheme.
Build on the scripts that are there; do not write a new one-off plotting
script beside them.
A genuinely new figure gets a new `fig_<name>.py` in that directory and is
then maintained like the rest.

`bash scripts/figures/render_all.sh` redraws the whole set and rebuilds
`docs/figures/gallery.html`; run it after any change to figTeX, a scheme or
the mydesign styles, because one figure rendering correctly says nothing
about the other forty-three.
`scripts/figures/make_tables.py` writes `docs/tables/` the same way, as
`.tex` in the mydesign dialect and `.md` beside it.

```bash
bash scripts/figures/pull_figdata.sh          # cluster -> docs/figures/data/
uv run --with-editable /Users/luis/master/software/figtex \
    python scripts/figures/fig_profile_shape.py
```

- `style.use("article", scheme="witch")` sets fonts, sizes and the colour
  cycle; colours per arm come from `scripts/figures/arms.py`, so an arm keeps
  its colour across figures.
  figTeX is deliberately not a dependency of this project -- it is pulled in
  with `--with-editable` at the call.
- Rendering is local: the documents are set in Minion Pro with mtpro2 maths
  through LuaLaTeX and those fonts are installed on the laptop only.
  So the cluster reduces each measurement to a long-format CSV of exactly the
  columns a figure draws and only that file travels; `docs/figures/data/README.md`
  documents each kind.
  Every file carries `method` and `dataset` key columns, the contract
  `figtex.project.schema` and the paper package already use.
- A figure that needs images rather than a table (a detector patch, a profile
  montage) gets a small `npz` dumped by the cluster script that selected
  them, and the plotting moves here.
- Derived quantities stay out of the data file and are computed in the
  figure, so their definition lives in one place.

## Reading a number out of the pipeline

Every reported quantity is parsed in one place, `integrator.utils.metrics`,
which also holds the chain's own emitter so the format cannot drift from its
readers. Never write a fresh regular expression for one of these:

- R-work and R-free, from a `RESULT` line, a phenix log, or a stats file
- anomalous peak heights, from a `PEAKS` line, `peak_heights.csv`, or
  `rs.find_peaks`
- the per-shell table, from `merging_stats.csv`, `merged.html` or `merge.log`
- the tables `dials.scale` and `dials.merge` print and nothing else carries:
  the error model's fitted `a` and `b`, the Rmerge and Rmeas forms the
  per-shell table omits, the Wilson analysis, the anomalous differences by
  shell. Read by `integrator.utils.dials_log`, re-exported by `metrics`.
- the polychromatic per-shell statistics, from the long `careless_stats.csv`
  the pipeline's `stats` step writes

`tests/test_metrics_contract.py`, `tests/test_dials_log.py` and
`tests/test_careless_stats.py` pin all of it against fixtures copied from
real output.
A test that skips itself does not pin anything: the six `merged.html` tests
were `importorskip("lxml")`, lxml was never installed, and the DIALS
reference line was missing from every merging figure for as long as that
reader existed. The failure it guards against is silent: three expressions for
the one `RESULT` line had drifted apart and `results_table.py` was rendering
an empty table rather than raising.

**Do not add a metric** without saying which existing number it replaces and
why that one is insufficient. A new metric cannot be compared with anything
already measured, so it costs a run and settles nothing.

## One script per job, and it lives in the repo

The repository is the source. The cluster is somewhere scripts are COPIED to
and run, never where they are written.

**Before writing any script, find the one that already does the job.**
`docs/scripts.md` lists all 278 of them with what each does, generated by
`scripts/inventory.py` from the scripts' own first lines so it cannot
drift. `uv run python scripts/inventory.py --check` fails if it is stale.
Searching it first is the step that was skipped when
`visualize_learned_basis.py` and `inspect_wilson_params.py` were
rebuilt from scratch, both of which already existed and had no callers.

These are the entry points; none of them needed replacing:

| job | the script |
|---|---|
| train, either arm | `scripts/{mono,poly}/cannon/run_all.sh`, or `scripts/poly/train.slurm` with `CONFIG=` |
| post-training scoring, poly | `scripts/poly/run_pipeline.py`, or `scripts/poly/pipeline.slurm` |
| post-training scoring, mono | `scripts/mono/cannon/03_pipeline.sh` |
| score one PDZ2 arm end to end | `scripts/poly/pdz2/arm_pdz2.sbatch <arm>` |
| cluster output to a plot-ready table | `scripts/figures/tidy_*.py` |
| cluster to laptop | `scripts/figures/pull_figdata.sh` |
| draw the whole figure set | `scripts/figures/render_all.sh` |

If one is close but wrong, **extend it** -- add an option that defaults to
today's behaviour, so existing runs stay bit-identical. Do not write a second
script beside it, and do not write a one-off sbatch to launch one.

**Code moves by git, never by copy.** `scp` and `rsync` are for DATA and
OUTPUTS -- reduced tables, npz dumps, logs, MTZs -- in that direction only.
A script or a module reaches the cluster by `git push` here and `git pull`
there, so what ran is a commit and can be named.

Copying code is what produced every one of these, all on 18 November:

- a staged stub package under `jobs/.../src/integrator` shadowed the
  installed one, twice, because Python puts a script's own directory first
  on `sys.path`;
- `pull_figdata.sh` copied each reducer to the cluster only when it was
  MISSING, so a fix to `tidy_pdz2.py` sat here while the cluster went on
  running the copy without it, and the pull came back without the svae arm
  with nothing to say why;
- `arm_pdz2.sbatch` copied `split_pdz2.py` from a scratch directory whose
  copy predated `--estimator`, and a job died after its predict pass on an
  argument the repository had had for days;
- the environment's checkout sat 52 commits behind and three arms failed
  preflight on `IntegratorCfg.__init__() got an unexpected keyword
  argument 'gate'`.

The checkouts on Cannon and what each is for:

| path | role | how it updates |
|---|---|---|
| `$S/integrator_main` | the checkout jobs are submitted FROM; `$SLURM_SUBMIT_DIR` | `git pull` |
| `/n/lab_storage/.../software/integrator` | what `integrator-cuda-dev` imports | `git pull` |
| `$S/integrator_cur` | a pre-git copy, kept until nothing points at it | superseded |

A new script is justified only when no existing one covers the job. It is
then written in `scripts/`, committed, pushed, and pulled on the cluster --
in that order. A script authored on scratch is lost at the next purge and
unreproducible in the meantime, which has already happened to this project
once.

PDZ2 is the one chain the poly pipeline cannot drive, because its merge is
the deposited eight-file joint careless call rather than config 4 on one
file. That is a reason to wire it in, not a licence to keep writing one-offs
around it.

## Testing an architecture means writing a new class

The registry exists so that an architecture is a CLASS with a name and a
config, not a flag threaded through an existing one. Testing a new way to
build a model means:

1. a new class, usually subclassing the one it varies from;
2. a line in `REGISTRY["integrator"]`;
3. a config per dataset, differing from its baseline in `integrator.name`
   and nothing else;
4. a test that builds it through `construct_integrator` and runs a
   forward.

**Never modify an existing integrator to try something.** A run on disk
has to keep the architecture it trained with, and a class that quietly
changes meaning makes every earlier result unreadable. `svae` is
`SVAEIntegrator` and stays that way; `svae_conjugate`, `svae_bg` and
`svae_calib` are separate entries in
`model/integrators/svae_gate_integrators.py`.

Check, every time, that nothing is built which training never reaches.
It has happened twice: `SVAEIntegrator` builds a 387,008-parameter signal
encoder that receives ZERO gradient once the gate is computed, and the
first draft of `svae_bg` initialised its head to zero weights, which cuts
the encoder out of the graph -- `d(out)/d(x) = W = 0` -- so the encoder
was dead at initialisation in the class written to give it a job. A dead
module is invisible: the run completes, the loss falls, and it inflates
every parameter count it appears in.
`tests/test_svae_gate_architectures.py` pins it.

## Pipelines

There are two, and they are already written. Do not write a new sbatch
script to run one; extend the stage that is wrong.

| arm | chain | front door |
|---|---|---|
| monochromatic | integrator → DIALS → phenix → `rs.find_peaks` | `scripts/mono/cannon/run_all.sh` |
| polychromatic | integrator → careless → phenix → `rs.find_peaks` | `scripts/poly/cannon/run_all.sh` |

Both take `--from/--to/--only` over the same stages (preflight, train,
pipeline, figures, plots), and both read a pipeline config rather than
hard-coding a path. `scripts/poly/run_pipeline.py --dry-run` prints every
command it would run without submitting anything; use it before any change.

**How many checkpoints get scored** is one setting, resolved for both arms by
`integrator.utils.checkpoints`: `last`, `every:<n>`, `all` or an epoch number.

- **Monochromatic defaults to `every:5`.** DIALS and phenix are CPU work, so a
  trajectory is cheap and worth more than an endpoint.
- **Polychromatic defaults to `last`.** One careless scale is tens of
  GPU-minutes; score more only once the data to test is settled.

`every:<n>` is a stride in epochs, not a divisibility test: checkpoints do not
land on multiples of n (9b7c saves at 4, 9, ..., 49), and the final checkpoint
is always kept. `tests/test_checkpoint_selection.py` pins this.

Polychromatic step `stats` runs every `careless.stats` tool -- CChalf, CCanom,
CCpred, Rsplit, I/sigma, completeness -- into one CSV each plus a long-format
`careless_stats.csv`. `scripts/poly/README.md` records which of careless's
three output MTZs each tool reads; they are not interchangeable.

## Cluster

Harvard Cannon, `ssh cannon`. Scratch `/n/netscratch/hekstra_lab/Lab/laldama`, source checkout `$S/integrator_cur`, data under `/n/lab_storage/hekstra_lab/people/aldama/integrator_data/`.
**Partitions: `test` and `gpu_test` are for submitting to, not just a last resort.**
A short job belongs there, because the production partitions queue.

| partition | for | limits |
|---|---|---|
| `gpu_test` | a GPU job of a few minutes: a predict pass, an export, a diagnostic. 12 h limit, so short TRAINING fits too | 2 jobs at once, under 65636 MB, and **fewer than 8 cores per GPU** -- `train.slurm` asks for 16 and is rejected without `-c 4` |
| `test` | the same on CPU | CPU only |
| `sapphire` | longer CPU work | CPU only |
| `gpu` | training, and anything over the `gpu_test` memory cap | 35 nodes, usually available |
| `gpu_requeue` | last choice | 15 nodes down and 6 draining on 18 Nov; a job sat PENDING over three hours and was resubmitted to `gpu`, where it ran |

Memory is what usually decides between `gpu_test` and `gpu`: the data module
loads `counts.npy` whole, so a PDZ2-sized set needs about 1.3 GB of arrays
before torch, and 845 needs `--mem=250G`. Ask for what the arrays need, not a
round number -- a 120G request that only needs 50G is a job that waits for
`gpu` when `gpu_test` would have started it immediately.
Paths baked in before the storage migration are stale, including inside other people's env scripts.
A watcher that treats a transient `squeue` failure as completion will report success on an empty result; require several consecutive empty polls.

## Where things are written down

`docs/methods.md` -- how each dataset was processed, stage by stage, with the script and the parameter.
`docs/exporting.md` -- export paths, column modes, ASU tables, the SigF defect, refinement chains.
`docs/ablation_1118_1000.md` -- the profile-model ablation and what has and has not been tested.
`docs/merging_svae.md` -- the mono and polychromatic merging model.
`docs/pitfalls.md` -- failures that produced plausible wrong numbers.
`docs/session_state.md` -- what is running now.

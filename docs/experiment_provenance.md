# Experiment provenance

Which code produced which numbers.
Results from before and after a merge are not comparable unless this says they are.

## 2026-10-08: merging the scaling/merging work into `nanobragg-prior`

### Why

The pixel-to-merging SVAE runs on monochromatic data but lands in the noise regime on Laue.
The three documented fixes for that -- a peaked profile warm-start, a background prior fitted to the data, and a structured `ChebyshevSpectrum` for G(lambda) -- all live on `scaling-refinement` / `add-mkrefl-cli-commands`, 110 commits that were never on this branch.
So does `group_by_asu_id` (HKL-held-out validation) and `cli/utils/merged_mtz_writer.py`.

Branch ancestry was checked first: `nanobragg-prior` descends from `profile-geometry-conditioning` (merge-base is that branch's tip) and already contains `svae-integrator` in full, so nothing was stranded on either.

### The pin

| | |
|---|---|
| last pre-merge commit | **`de23bcc`** "Export the likelihood, not the posterior..." |
| its parent | `bbb8428` "Add a nanoBragg profile surrogate with learned global physics" |
| branch | `nanobragg-prior` |
| merged in | `add-mkrefl-cli-commands` (110 commits, base `b1aea10`) |

**Everything below was produced at `de23bcc` or earlier and must be read against it.**

### Results pinned to the pre-merge code

**The 20-arm ablation on HEWL 1118 at 360 frames** (jobs 51367853, 51370688).
Three preprocessing arms, seven profile arms, six Gamma arms, four KL-weight arms, one knob each from a common baseline.
Scored by gate-conditioned precision: CC-half between two half-merges, read only where `Var(z)` is near 1.
Run against the source synced to `netscratch/.../integrator_cur` at `de23bcc`.
**Anything relaunched after the merge is not comparable to these.**

**The export comparison on HEWL 1118 at 1440 frames** (`lik_cmp/`).
posterior vs cavity vs rb, one prediction, three MTZs, identical careless.
R-free 0.2462 / 0.2349 / 0.2663; CC-half 0.9663 / 0.8310 / 0.8817; IOD 204 17.23 / 14.17 / 14.07.
The gate says those CC-half and peak numbers are reweighting, not information: `Var(z1-z2)` is 23.3 / 21.1 / 6.1 against an honest 2.0.

**The careless Gamma likelihood** (`lik_cmp/gamma/`, superseded).
Inverted construction; see the README in that directory.
The environment patch at `envs/crls/.../likelihoods/{mono,laue}.py` is NOT in git -- it lives in the micromamba env with `.orig` backups beside it, and is off unless `CARELESS_GAMMA_LIKELIHOOD` is set.

### Known not to carry over

12 of the 34 files modified before the merge are also touched by the 110 commits, including `base_integrator.py`, `factory_utils.py`, `poly_data_module.py`, `wilson_loss.py`, `registry.py` and `integrator_utils.py`.
Conflicts there were resolved by hand, so a behaviour that held at `de23bcc` should be re-checked rather than assumed.

### How to reproduce a pre-merge number

```bash
git checkout de23bcc            # detached; do not commit here
uv run pytest tests/            # 408 passed, 1 pre-existing failure
```

The pre-existing failure is `test_wilson_stabilize.py::test_legacy_default_unchanged_and_no_new_state`, which predates this session.

## Dataset boundary: the `centric` flag on 1118 (October 2026)

`laue_runs/1118_hewl360/shoeboxes/metadata.npy` gained a `centric` column.
Everything trained before this point treated every reflection as acentric; everything after gets `Gamma(1/2)` for centrics as Wilson statistics require.
The pre-flag file is kept as `metadata.npy.pre_centric`.

Space group `P 43 21 2`, so of 1,283,813 observations over 194,861 distinct reflections, **111,847 (8.7%) are centric** -- and the fraction is resolution-dependent:

| d (A) | centric |
|---|---|
| 99.0-3.0 | **14.8%** |
| 3.0-2.2 | 8.8% |
| 2.2-1.9 | 6.7% |
| 1.9-0.0 | 5.8% |

The low-resolution shell matters most: that is where the strong reflections carrying the anomalous signal are, and an acentric `Gamma(1)` prior on a centric reflection is too tight near zero, so those reflections were being shrunk away from zero.

**What this means for comparisons.** The 20 ablation arms in `ablation_1118_360.md` and the measured laue-dials baseline all predate the flag, so they remain comparable among themselves.
Any new arm on this dataset is a different prior and must not be put in that table without saying so.
The poly SVAE (`hewl1118_svae_merge.yaml`) is the first run on the far side of the boundary.

This was listed as "real but secondary" in [[posterior_exported_as_likelihood]] -- 7-14% of reflections -- and the measured 8.7% with a 14.8% low-resolution shell confirms that range.

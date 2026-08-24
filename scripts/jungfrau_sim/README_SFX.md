# SFX per-image-Wilson experiment

End-to-end harness for the novel piece: a **Wilson prior applied during integration**,
with **per-image G and B learned** from raw JUNGFRAU shoeboxes. Standard SFX software puts
G/B in a downstream *scaling* model and integrates by summation; here the prior lives in
the integrator and the per-image parameters are learned jointly with the intensities.

## Pipeline

```
sfx_generate.py   per-image G/B Wilson -> true I -> profile -> JUNGFRAU detector
                  -> shoeboxes (integer + real-valued) + all ground truth
sfx_experiment.py the REAL 5-encoder hierarchical stack (ProfileEncoder + 4x
                  IntensityEncoder) -> Gamma q(I)/q(bg) + learned-basis ProfileSurrogate;
                  per-image-G Embedding + B in the Wilson prior; ELBO with the real
                  CountLikelihood; checkpoints + per-epoch recovery metrics
sfx_analyze.py    recovery-vs-training curves + final recovered-vs-true scatter
```

**The architecture is the production one.** Same `ProfileEncoder` + 4x `IntensityEncoder`
(built through the factory's own 2d presets), same `build_gamma("shape_rate")` surrogates
for q(I)/q(bg), same learned-basis `ProfileSurrogate` for q(profile), same
`rate = zI * zp + zbg` composition, same `CountLikelihood`. The ONE deliberate difference
is the prior: the Wilson G is per-image (an `Embedding` indexed by image_id) instead of a
single global scalar -- the thing under study. `--profile known` swaps in the true profile
as an oracle control to isolate scale/intensity recovery from profile learning.

## Quick start (local)

```bash
uv run python scripts/jungfrau_sim/sfx_generate.py --n-images 150 --n-refl 100 --out data/sfx_sim
uv run python scripts/jungfrau_sim/sfx_experiment.py --likelihood poisson --epochs 50 --device cpu
uv run python scripts/jungfrau_sim/sfx_analyze.py
```

## Full matrix (cluster)

```bash
N_IMAGES=2000 EPOCHS=400 DEVICE=cuda bash scripts/jungfrau_sim/run_sfx_experiments.sh
```

Each arm is one independent process writing to `data/sfx_runs/<tag>/`, so they can equally
be submitted as separate cluster jobs. Runs are **resumable** (`--resume`, on by default in
the script) and `history.json` is flushed every eval, so a preempted job loses nothing.

## The knobs (compose freely)

| flag | values | question it answers |
|---|---|---|
| `--likelihood` | `poisson` / `normal_coupled` / `normal_free` | integer+Poisson vs real+Normal |
| `--profile` | `learned` (default) / `known` | learned-basis surrogate vs oracle profile |
| `--scale` | `per_image` / `global` | per-image Wilson G vs one shared G |
| `--per-image-B` | flag | per-image B (needs data from `--sigma-b > 0`) |

Data axis: the Poisson arm reads `counts_poisson` (rounded integers), the Normal arms read
`counts_real` (the calibrated real-valued detector output). Same shoeboxes, same truth.

## What each recovery metric means

- `corr_logI` — per-observation intensity vs truth (log). Ceiling here is the
  matched-filter ~0.68; the prior can push past it by shrinking weak reflections.
- `corr_logG` — per-image scale vs truth. 0 for `--scale global` (nothing per-image).
- `B_mean` / `B_err` — recovered Wilson B (true 20; matched-filter ceiling ~17). B is the
  slow one: its only gradient is the weak Wilson-KL resolution slope.
- `corr_bg` — background vs truth.
- `profile_cos` — cosine of the learned `q(profile)` mean vs the true profile (learned only).

## Tuning notes carried over from the local runs

- **Two-timescale LR is load-bearing.** The Wilson hyperparameters get their own groups: G
  at 10x and B at 40x the encoder LR, because their only gradient is the KL term, far
  weaker than the reconstruction gradient. Without this B barely moves off its init.
- **G embedding init** = log of the per-image mean matched-filter intensity (not raw
  counts, which are background-inflated). Prevents the scale collapsing to the floor.
- **Longer helps B, not G.** G/background lock in within a few epochs; B climbs across the
  whole run, which is why the cluster budget (`EPOCHS=400`) targets B convergence.
- **The old baseline numbers do not carry over directly.** The earlier local result (150
  images x 100 refl, 50 epochs: corr(logG) ~0.96, B ~17, corr(bg) ~0.996, corr(logI) ~0.72,
  all three likelihood arms within noise) was measured with a *small purpose-built CNN* and
  a profile-matched-filter anchor on q(I). This harness now uses the **production
  encoders + learned-basis profile**, which has ~545k parameters (vs ~50k) and no anchor --
  the Gamma heads must find the photon scale themselves, pulled there by the per-image G
  prior. Expect slower early convergence and re-measure the baseline; the qualitative
  finding to re-test is whether the likelihood arms still land within noise of each other.
- **The intensity ceiling is architecture-dependent too.** The ~0.68 matched-filter number
  is what an *oracle-profile* linear estimator achieves. With `--profile learned` the model
  has to earn that through `q(profile)`; `profile_cos` tells you how close it is.

## Related standalone studies (no training, parameter-recovery only)

- `wilson_per_image_prior.py` -- the closed-form MLE for G/B, the identifiability, the case
  1 vs case 2 (per-image B) crossover, and the partiality misspecification.
- `wilson_scale_architectures.py` -- SGD on a free per-image-G Embedding reaches the MLE;
  amortized-MLP vs free vs hierarchical parameter-count/accuracy tradeoff.

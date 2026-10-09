# Best configuration per dataset

What to start from, and which knobs are load-bearing.
Numbers and their references live in `results_by_dataset.md`; this file is about the settings.

A config here is "best" only against the reference arm named beside it, at the frame count named beside it.
Where a sweep gave a front rather than a winner, the front is listed.

## The settings that recur

Four choices show up in every winning monochromatic config, and they were each arrived at by a sweep rather than by taste.

| setting | value | why |
|---|---|---|
| `reparameterization` | `mean_fano` | defaults silently drop anomalous peaks 36 -> 31 |
| `positive_constraint` | `exp` | same sweep; pairs with `mean_fano` |
| `decoder_weight_decay` | `0.0` | 821's `config_full` is "+lp +exp +no basis decay" |
| profile basis penalties | `0.0` | `smoothness_weight` and `orthogonality_weight` both off in the winning arms |

**Carrying a penalty or a weight decay across from an older template is the easiest way to lose a result**, because neither changes anything visible until the anomalous peaks are measured.

## Monochromatic

### HEWL 9b7c -- the strongest result

`configs/mono/hewl9b7c_mono_expmean_lp.yaml`, run `105gh86k` epoch 49.
R-work/R-free 0.161/0.180, IOD 204 at **36.5 sigma** against the DIALS reference's 32.7; wins 10 of 13 sites.

Load-bearing: `mean_fano` + `positive_constraint: exp`, `lp_correction: true`, `decoder_weight_decay: 0.0`.
Four things changed at once to get here, so no single one is credited.

### SBGrid 821 -- 7LVC, Mn/S-SAD

`config_full` (+lp +exp +no basis decay): R-free **0.1626**, top MN 42.9.
The `+lp` arm alone gives the best anomalous (MN 44.0, 19 sites) at R-free 0.1642.

**Best R-free and best anomalous are different arms.** Pick by which one the claim needs.

### SBGrid 845 -- 7RIN, native S-SAD

A Pareto front over the KL weight, not a single config:

| beta | top sigma | cc_anom | R-free | |
|---|---|---|---|---|
| 0.5 | 16.1 | 0.357 | **0.1835** | best R-free |
| 0.25 | **18.0** | 0.310 | 0.1852 | the S-SAD default |
| 0.1 | 17.5 | **0.516** | 0.1936 | pushes consistency hardest |

beta 0.25 and 0.5 both Pareto-dominate DIALS.
**beta = 0 is an artifact**, not a result: `cc_anom` 0.949 comes from collapsed variances making DIALS reject 26% of observations.

## Polychromatic (Laue)

### HEWL 1118 -- per-observation integration

Family `screen_l8_k7_exp`: latent 8, kernel 7, `mean_fano` + exp mean, `stabilize_scale`, `init_G 5000`.
At 1440 frames this **ties** `laue.integrate` on iodine (21.1 / 19.7 / 8.4 against 20.5 / 19.4 / 8.6).

Kernel size plateaus at 7 (iodine mean sigma 8.16 / 9.01 / **9.12** / 8.70 / 8.92 / 8.74 for k = 3/5/7/9/11/13): it tracks the 5-10 px spot, not the 25 px shoebox.

### HEWL 1118 -- pixel-to-merging SVAE (new, untested)

`configs/poly/hewl1118_svae_merge.yaml`, 360 frames.
The monochromatic version of this model works; the Laue version has previously landed in the noise regime, which is the documented background/profile identifiability collapse.
Three settings address it and all three are necessary:

1. `warmstart_basis_path` -- a PEAKED profile. Without it the decoder starts near uniform, and on ~96.6%-background Laue shoeboxes a uniform profile makes intensity x profile a flat field indistinguishable from the background.
2. a background prior FITTED to the data (`prepare_bg_prior` writes `bg_prior.pt`, auto-injected) instead of the 1.0 count/pixel default, against Laue's ~12.
3. `polychromatic_wilson` with a low-order Chebyshev `G(lambda)`, NOT a learned MLP scale -- a black-box MLP cannot recover the incident spectrum against the merge gauge.

Also required, and easy to miss:

- `pi_weight` / `intensity_kl_weight` must be **0**: the integrator supplies the intensity ELBO itself and would otherwise double-count the KL. It refuses to construct otherwise.
- `group_by_asu_id: true` holds out whole reflections. With 92.8 observations per HKL an observation-level split leaks the answer across it.
- `n_hkl` must match the dataset: 13,829 for 1118 at 360 frames (`scripts/prepare_asu_ids.py` prints it).

Harmonics need no handling here: lambda spans 0.97-1.25 A, a ratio of 1.289, and two orders of one ray both in the bandpass would need >= 2.0. Measured: no observation shares a ray.

**Status: builds and is validated through the factory; not yet trained.**

## Known-bad settings

| setting | what happens |
|---|---|
| `beta = 0` (no intensity KL) | variance collapse; `cc_anom` 0.949 is an artifact of outlier rejection |
| `transform: log1p` | R-free 0.4016 and IOD 204 at 1.06 sigma in the 20-epoch ablation -- no anomalous signal, despite the best sigma calibration of any arm |
| a learned MLP scale for Laue | stays near-constant (CV 0.08); cannot learn G(lambda) against the merge gauge |
| `mean_fano` defaults without `exp` | anomalous peaks 36 -> 31 |

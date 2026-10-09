# Ablation: HEWL 1118, 360 frames, October 2026

Twenty arms, one knob each, on the 360-frame set (1,283,813 observations, 13,829 unique ASU reflections, 25x25 2D shoeboxes).
Twenty epochs each -- against 40-50 for production -- so absolute values sit below the documented references and only the differences between arms mean anything.

Every arm went through the identical chain: predict -> careless config 4 (dmin 1.5, seed 2024) -> merging stats -> two-pass phenix -> `peak_heights.csv`.

## The baseline, and the noise floor it accidentally gave us

The common baseline is `configs/poly/hewl1118_poly_ref.yaml`: `transform: anscombe`, `reparameterization: shape_rate`, `positive_constraint: exp`, `intensity_kl_weight: 1.0`, `learned_basis_profile`.

**Four arms varied a knob to the value the baseline already had**, so they are four independent runs of one configuration.
There is no seeding in the training CLI, so they are genuine replicates:

| arm | R-work | R-free | IOD 204 | IOD 206 |
|---|---|---|---|---|
| q1_transform_anscombe | 0.1708 | 0.2196 | 12.46 | 10.36 |
| q2_profile_learned | 0.1789 | 0.2312 | 13.03 | 10.64 |
| q3_gamma_shape_rate_exp | 0.1811 | 0.2310 | 12.76 | 11.12 |
| q4_kl_1p0 | 0.1832 | 0.2362 | 13.18 | 11.66 |

**Noise floor:** R-work sd 0.0054 · R-free sd **0.0070** · IOD 204 sd **0.31** · IOD 206 sd 0.57 · IOD 205 sd 0.27 · MET 105 sd 0.20 · CYS 64 sd 0.26 · CYS 30 sd 0.28.

Nothing below 2 sigma of those should be read as an effect.
This was not planned; it is the single most useful number the ablation produced, because it retires several comparisons made earlier in the project.

## Results

Sorted by R-free. Full per-site values are in the session transcript; the headline sites are here.

| arm | R-work | R-free | IOD 204 | IOD 206 | IOD 205 |
|---|---|---|---|---|---|
| **q4_kl_0p1** | **0.1578** | **0.2024** | **15.84** | **14.89** | **8.36** |
| q3_gamma_nakagami_exp | 0.1685 | 0.2154 | 13.85 | 11.28 | 7.00 |
| q4_kl_0p5 | 0.1655 | 0.2163 | 13.89 | 12.59 | 7.18 |
| q3_gamma_mean_fano_exp | 0.1721 | 0.2183 | 13.01 | 11.80 | 6.04 |
| q2_profile_shapelet | 0.1724 | 0.2188 | 12.75 | 10.22 | 6.28 |
| q1_transform_anscombe | 0.1708 | 0.2196 | 12.46 | 10.36 | 6.00 |
| q1_transform_standardization | 0.1743 | 0.2214 | 12.41 | 10.82 | 6.22 |
| q2_profile_geomboth | 0.1701 | 0.2221 | 13.32 | 12.39 | 6.32 |
| q3_gamma_nakagami_softplus | 0.1797 | 0.2261 | 12.37 | 11.19 | 5.46 |
| q2b_coordchan | 0.1793 | 0.2280 | 13.92 | 11.68 | 7.64 |
| q3_gamma_mean_fano_softplus | 0.1813 | 0.2286 | 13.04 | 11.33 | 6.12 |
| q2b_coordchan_geomboth | 0.1763 | 0.2293 | 11.78 | 10.63 | 5.63 |
| q2_profile_physical | 0.1786 | 0.2296 | 12.46 | 10.57 | 6.07 |
| q3_gamma_shape_rate_exp | 0.1811 | 0.2310 | 12.76 | 11.12 | 6.23 |
| q2_profile_learned | 0.1789 | 0.2312 | 13.03 | 10.64 | 5.59 |
| q2_profile_fixed | 0.1830 | 0.2348 | 12.48 | 11.15 | 6.16 |
| q4_kl_1p0 | 0.1832 | 0.2362 | 13.18 | 11.66 | 6.02 |
| q4_kl_0p25 | 0.1855 | 0.2388 | 7.62 | 9.12 | 4.73 |
| q3_gamma_shape_rate_softplus | 0.3048 | 0.3997 | 0.82 | -0.38 | 1.96 |
| q1_transform_log1p | 0.2973 | 0.4016 | 1.06 | -0.08 | 2.09 |

## What actually cleared the noise floor

| effect | R-free | IOD 204 | |
|---|---|---|---|
| `intensity_kl_weight: 0.1` | +3.9 sigma | **+9.4 sigma** | the one large effect |
| `intensity_kl_weight: 0.5` | +1.9 | +3.3 | same knob |
| `nakagami` + `exp` | +2.0 | +3.1 | real but modest |
| `profile_coord_channels` | +0.2 | +3.4 | peaks only |

**Everything else is inside run-to-run variation**, including the entire profile-family comparison and anscombe vs standardization.
An earlier reading of this table called geom_both a clear winner; at 1.1 and 1.5 sigma it is not distinguishable.

`kl_0.1` lifts **every site**, not just the iodides -- IOD 205 +8.9 sigma, MET 105 +6.5 sigma, CYS 30 +4.5 sigma.
A uniform lift across strong and weak scatterers is what signal recovery looks like; a reweighting artifact would favour some sites.

## Two catastrophic settings

`transform: log1p` and `shape_rate` + `softplus` both give R-free ~0.40 and peaks at or below 1 sigma, with NEGATIVE peaks at most sites.
Both had the BEST sigma calibration in the whole ablation (Var(z) 28.1 and 15.7 against a 1.0 ideal).
A model can be perfectly honest about its uncertainty and measure nothing.

`q4_kl_0p25` is out of line with its neighbours (IOD 204 7.62 between 15.84 at kl 0.1 and 13.89 at kl 0.5), depressed roughly uniformly across sites while R-work is only slightly worse.
Treat it as a suspected bad run until a replicate says otherwise.

## Against the classical pipeline

`laue.integrate` on the same 360 frames (documented): IOD 204 **15.7**, IOD 206 **15.7**.
Our best arm: 15.84 / 14.89 -- a tie on one iodide and about -1.4 sigma on the other.

That matches the documented pattern: the model's edge is a high-multiplicity phenomenon, and at 360 frames the margin narrows rather than widens (at 1440 frames we tie, 21.1 vs 20.5).
R-free for the laue arm was never measured at 360 frames; that run is in flight.

## Metrics that did NOT work

A gate-conditioned score was built to triage arms cheaply: CC-half between two half-merges, read only where `Var(z)` is near 1.
**It is anti-predictive.** The two best-calibrated arms are the two total failures, and the worst-calibrated arm is the winner.
CC-half from careless behaves the same way -- the winner has among the lowest.
Of everything computed, only R-free and peak height tracked each other.

## Configs

Generated by `.scratch_ablate/gen_ablation.py` into `configs/ablate/` (committed), and the follow-on combination arms by `.scratch_ablate/gen_combo.py` into `configs/combo/`.
Both filter surrogate args against each class's signature, because the baseline's `latent_dim` / `prior_scale` / `smoothness_weight` are written for the learned basis and the fixed and shapelet surrogates reject them.

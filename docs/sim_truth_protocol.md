# Grading integration methods on simulated Laue data with known truth

This protocol scores an integration method against the known truth of a datsimx simulation.
How the data are simulated is in `docs/sim_truth.md`; this document fixes what is compared, on which reflections, and how.

Status markers: **[implemented]** names the script that computes the quantity; **[to build]** marks a quantity whose definition is fixed here but has no script yet.

## 1. Data

Each dataset is one row of `scripts/sim_truth/grids/paper_full.tsv`, under `/n/netscratch/hekstra_lab/Lab/laldama/sims/paper_full/<row>/`.

| file | content | units |
|---|---|---|
| `truth/shots_1_rank0.h5:sim_image/<shot>` | noiseless Bragg image $B_p$ | photons |
| `truth/shots_1_rank0.h5:{h,k,l,wave}_data/<shot>` | per-pixel intensity-weighted average fractional $(h, k, l)$ and wavelength | — , Å |
| `truth/background_1.h5:background` | noiseless background $b_p$, one image for all shots | photons |
| `noisy/run_1_master.h5` | the data every method receives | ADU |
| `truth/ground_truth.mtz` | structure factors of the simulated crystal (7LVC) | — |
| `truth/truth_reflections.csv` | per-reflection truth, §2 | photons |
| `noisy/geom_run1.expt`, `truth/shots_1_rank0.h5:{Umat,Bmat}/<shot>` | detector, beam, true crystal per shot | — |

The noisy image is $c_p = g\,\mathcal{P}\!\left[(B_p + b_p)(1+\epsilon_p)\right] \ast \mathrm{PSF} + a + r_p$, with gain $g = 0.7$ ADU/photon, calibration error $\epsilon_p \sim \mathcal{N}(0, 0.03^2)$, nanoBragg's PSF (FWHM 0.0847 mm = 1.13 px, kernel radius 5 px), ADC offset $a = 10$ ADU and readout $r_p \sim \mathcal{N}(0, 3^2)$ ADU, in nanoBragg's order (Poisson, calibration, PSF, gain, offset, readout).
Its expectation in photons is $\mathbb{E}[(c_p - a)/g] = (\mathrm{PSF} \ast (B + b))_p$.

## 2. Truth quantities

A pixel $p$ belongs to reflection $r = (h, k, l)$ of shot $s$ when $B_p > 0$ and $\mathrm{round}(h_p, k_p, l_p) = (h, k, l)$.
Write $P_r$ for that pixel set.

| quantity | definition | status |
|---|---|---|
| truth intensity | $I_r = \sum_{p \in P_r} B_p$ | [implemented] `truth_table.py`, column `i_true` |
| truth profile, before the detector | $q_{r,p} = B_p / I_r$ on $P_r$ | [implemented] from the maps |
| truth profile, as recorded | $\tilde q_{r,p} = (\mathrm{PSF} \ast B^{(r)})_p \,/\, \sum_p (\mathrm{PSF} \ast B^{(r)})_p$, with $B^{(r)}_p = B_p \mathbb{1}[p \in P_r]$ | [to build] nanoBragg's PSF applied to each reflection's image |
| truth background | $b_p$ in photons; in ADU, $g\,b_p + a$ | [implemented] `background_1.h5` |
| map residual | $\rho_p = \max(\lvert h_p - \mathrm{round}\,h_p\rvert, \lvert k_p - \mathrm{round}\,k_p\rvert, \lvert l_p - \mathrm{round}\,l_p\rvert)$ | [implemented] |
| mixed fraction | $m_r = \sum_{p \in P_r} B_p \mathbb{1}[\rho_p > 0.25] / I_r$ | [implemented] column `mixed_0p25` |
| truth wavelength | $\lambda_r = \sum_{p \in P_r} B_p \lambda_p / I_r$ | [implemented] column `lam_mean` |

$I_r$ is the number of Bragg photons of the observation before noise; it includes the spectrum weight, Lorentz and polarization factors and the solid angle, which is the quantity an integration method estimates per observation.
The maps hold intensity-weighted averages over all contributions to a pixel, so where two reflections contribute to one pixel the label names one of them; $\rho_p$ is large on such pixels and on a reflection's own far tails, so $m_r$ is an upper bound on the share of $I_r$ that rests on shared pixels.

## 3. Join

A method's observation is matched to a truth row on (shot, $h$, $k$, $l$), with shot = `BATCH` in a laue-dials unmerged MTZ.
With the true crystal injected (§6), both sides use the true setting and no asymmetric-unit mapping is applied.
Two controls are reported with every join: the join rate of the sign-inverted key $(-h, -k, -l)$, which a correct join exceeds, and $\lvert \lambda_\text{method} - \lambda_r \rvert$ on joined rows.
[implemented] `score_against_truth.py`.

## 4. Domain

Fixed before any method is scored and computed from the truth alone:

- $\lambda_r \in [1.035, 1.165]$ Å, inside the simulated spectrum, where reflections are fully excited;
- $m_r \le 0.05$.

Every metric is reported on this domain and, separately, on all joined reflections.
No metric may select its own range.

## 5. Metrics

Notation: $\hat I_r$ and $\hat\sigma_r$ are a method's intensity and standard error in photons (ADU divided by $g$); $d_r = \hat I_r - I_r$.
Binned metrics use bins of $I_r$: edges 0, 1, 3, 10, 30, 100, 300, 1000, 3000, ∞ photons.
Binning on $I_r$, which carries no noise, does not attenuate a slope the way binning on $\hat I_r$ does.

### 5.1 Intensity bias

Per bin: $\mathrm{mean}(d_r)$, $\mathrm{median}(d_r)$, and $\mathrm{mean}(d_r / I_r)$.
A method without bias has $\mathrm{mean}(d_r) = 0$ in every bin.
[implemented] `score_against_truth.py`, columns `dev_mean`, `dev_median`, `rel_dev_mean`.

### 5.2 Intensity scatter

Per bin: $\mathrm{sd}(d_r)$.
Read against the same quantity for the truth-profile fit (§7.2) on the same reflections.
[implemented] column `dev_sd`.

### 5.3 Error-estimate calibration

Per bin: mean and standard deviation of $z_r = d_r / \hat\sigma_r$.
Calibrated error estimates give $\mathrm{mean}(z) = 0$ and $\mathrm{sd}(z) = 1$; $\mathrm{sd}(z) > 1$ means $\hat\sigma$ is too small.
[implemented] columns `z_mean`, `z_sd`.

### 5.4 Profile distance

Per reflection: $L_1 = \sum_p \lvert \hat q_{r,p} - \tilde q_{r,p} \rvert$ over the union of the method's support and $P_r$, with $\hat q_r$ the method's profile normalised to sum 1; $0 \le L_1 \le 2$.
The reference is the recorded truth profile $\tilde q_r$, not the noisy spot: a profile model estimates the expected spot shape, and the noisy spot differs from it by the noise.
Reported as median and 95th percentile per bin.
[to build] needs $\tilde q_r$ and each method's profile on the pixel grid.

### 5.5 Profile width

Per reflection: $n_\text{eff} = 1 / \sum_p q_p^2$ for $\hat q_r$ and for $\tilde q_r$; reported as the median of $n_\text{eff}(\hat q_r) / n_\text{eff}(\tilde q_r)$ per bin.
[to build]; `truth_table.py` column `n_eff` holds $n_\text{eff}(q_r)$, before the PSF.

### 5.6 Mask capture

Per reflection: $\sum_{p \in M_r} \tilde q_{r,p}$, the share of the recorded truth profile inside the method's integration region $M_r$.
A method that sums inside $M_r$ cannot recover more than this share of $I_r$ without a model of the remainder.
[to build].

### 5.7 Background error

Per reflection, over the method's shoebox $S_r$: $\mathrm{mean}_{p \in S_r}(\hat b_p - b_p)$ and $\left(\mathrm{mean}_{p \in S_r}(\hat b_p - b_p)^2\right)^{1/2}$, in photons per pixel, with $\hat b_p$ the method's background converted as $(\hat b^\text{ADU}_p - a)/g$.
The background error summed over the integration region, $\sum_{p \in M_r}(\hat b_p - b_p)$, is reported beside 5.1, since it enters $d_r$ directly.
[to build]; for laue-dials, `BG` and `SIGBG` in the MTZ are per-reflection totals, so only the summed form applies.

### 5.8 Neighbour contamination

Per reflection: $\sum_{p \in M_r} \sum_{r' \ne r} B^{(r')}_p \,/\, I_r$, the photons of other reflections inside the method's integration region, relative to $I_r$, from the truth labels.
Reported per bin beside 5.1, so that a bias can be split into the part neighbours account for and the rest.
[to build].

### 5.9 Merged structure factors

Computed for every method and for the references §7.2 and §7.3.
After merging with careless under the pipeline's fixed settings: Pearson correlation and $R = \sum \lvert F - F_\text{true} \rvert / \sum F_\text{true}$ between merged $F$ and `ground_truth.mtz`, after one least-squares scale, by resolution shell, on reflections present in both.
[to build]; careless runs through `scripts/poly/run_pipeline.py`.

### 5.10 Refinement

Computed for every method and for the references §7.2 and §7.3.
R-work and R-free from phenix against the merged data, starting from 7LVC, with R-free flags generated once per dataset and pinned for every method (`docs/reference_data.md`), averaged over careless merging seeds.
Parsed by `integrator.utils.metrics`.
[to build].

## 6. Methods and geometry

Every method receives the noisy images and the true crystal of each shot (`scripts/sim_truth/paper_sims/sim_laue_process.sbatch`, step 0), so indexing and refinement do not differ between methods.
laue-dials runs `laue.find_spots` (gain 1.0, max separation 10) → `sim_add_s1.py` → `laue.optimize_indexing` (λ 1.00–1.20 Å, $d_\text{min}$ 1.15 Å) → `laue.refine` → `laue.predict` → `laue.integrate`, as on the laue-dials paper's simulations; `sim_laue_all.sbatch` repeats prediction with `cutoff_log_probability=-20`.
A method trained on the data (our integrator) uses the reflections `laue.predict` lists, so all methods are scored on the same observations.

## 7. References

### 7.1 laue-dials

`integrated.mtz` and `integrated_all.mtz` from §6, scored by every metric it supports.

### 7.2 Truth-profile fit

$\hat I_r = \sum_p \tilde q_{r,p}\,(y_p - \tilde b_p) / \sum_p \tilde q_{r,p}^2$ over $P_r$, with $y_p = (c_p - a)/g$ and $\tilde b_p = (\mathrm{PSF} \ast b)_p$.
It uses the true recorded profile and the true background, so it shows how far a method is from the estimate available with that knowledge; it is not a bound on precision.
Its standard error is $\hat\sigma_r^2 = \sum_p \tilde q_{r,p}^2 v_p / (\sum_p \tilde q_{r,p}^2)^2$ with $v_p$ the pixel variance of §1.
[to build].

### 7.3 Truth intensities

$\hat I_r = I_r$ for every observation, with $\hat\sigma_r$ taken from §7.2 so that merging weights are those a measurement would carry.
It enters only §5.9 and §5.10: merged and refined through the same chain as the methods, it gives the result of that chain without measurement noise, which is limited by the merging model and by the difference between the generating structure factors and the refinement model.

### 7.4 Reading the references

On the same observations, through the same careless and phenix settings, averaged over merging seeds with pinned R-free flags:

- the difference between §7.3 and §7.2 is the cost of the measurement noise to a method that integrates each observation separately;
- the difference between §7.2 and a method is the most that better profiles and backgrounds can recover for such a method.

Neither is a bound for a method that pools information across observations of one reflection or across neighbouring reflections, and a prior on the intensity can move merged statistics in either direction without changing the per-observation error; per-observation metrics (§5.1–5.3) are reported beside every merged result for that reason.

## 8. Reporting

Every number is reported with the dataset row, the domain (§4 or all joined), the bin, and the count $n$.
The output of each scorer is a long-format CSV with `method` and `dataset` columns, which `scripts/figures/` reads.
A comparison between methods on one dataset uses the same joined reflections for both.

## 9. Known differences from real data

- The spot profile is datsimx's `gauss`; the laue-dials paper used `square`.
- The detector is an Eiger at 169.8 mm emulating the paper's Rayonix in angle per pixel; there are no panel gaps.
- One noiseless background image serves all shots; shots differ in background only through the noise.
- No divergence (measured effect 0.48 % median on $I_r$ at 0.1° mosaic, `docs/sim_truth.md`).
- At mosaic 0.5° the sampling (322 energies, 100 domains) renders spots as separate per-domain lines, as in the paper's own data, and $m_r$ is largest (1.65 % of Bragg photons on pixels with $\rho_p > 0.25$).
- Whether nanoBragg's 5-px PSF kernel conserves the total photons of a spot is unverified; until it is, $I_r$ and the expected recorded total may differ by the kernel's truncation.

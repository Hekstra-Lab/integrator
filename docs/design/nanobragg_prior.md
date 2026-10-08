# The nanoBragg profile prior

How the model on branch `nanobragg-prior` works: what it assumes, what it is trained on, what it learns, and what it does not do yet.
Code references are to commit `56d7584`.

## In one paragraph

Every shoebox is modelled as a Poisson image whose expected counts are an intensity times a profile plus a background.
The intensity and background are inferred per reflection by neural encoders, as in the rest of the integrator.
The profile is not learned freely.
It is rendered by a PyTorch port of James Holton's nanoBragg, on the shoebox's own pixels, from the orientation of the image the shoebox came from.
What is learned about the profile is a handful of physical numbers shared by the whole dataset (crystal size, mosaic spread, spectrum), plus one small per-reflection correction (a centroid shift) that a prior keeps close to zero.
So weak reflections take their shape from physics fitted on the bright ones, and bright reflections can move their centroid where the data demand it.

## Generative model

For shoebox $j$, taken from image $s(j)$, with Miller index $h_j$ and pixels $p = 1, \dots, P$ ($P = 441$, a $21 \times 21$ window):

$$
\begin{aligned}
\Theta &= (N, \eta, \log S_{1..K}) && \text{global physics, point-estimated} \\
z_j &\sim \mathcal{N}(0, \tau^2 I_2) && \text{centroid shift in pixels} \\
I_j &\sim \mathrm{Exp}\!\left(\text{mean } \Sigma(\lambda_j, d_j)\right) && \text{Wilson prior, acentric} \\
b_j &\sim \mathrm{Gamma}(\alpha_{\mathrm{bin}(j)}, \beta_{\mathrm{bin}(j)}) && \text{per-resolution-bin background prior} \\
\pi_j &= \mathrm{normalize}\!\left(\mathrm{nanoBragg}(p - z_j;\; A_{s(j)}, h_j, \Theta)\right) && \text{deterministic given } z_j \\
c_{jp} &\sim \mathrm{Poisson}\!\left(I_j\, \pi_{jp} + b_j\right) && \text{for every valid pixel } p
\end{aligned}
$$

$A_{s} = U_s B$ is the true setting matrix of image $s$ (columns $a^*, b^*, c^*$ in the lab frame), read from the simulation and never estimated.
$\Sigma(\lambda, d) = G(\lambda)\, e^{-2 B s^2}$ is the Wilson mean (the loss works with its inverse, the rate), with $s^2 = 1/(4 d^2)$, a learned spectrum scale $G(\lambda)$ (Chebyshev, degree 5) and a learned $B$; this is the existing `polychromatic_wilson` loss, unchanged.

```
          Θ = (N, η, log S)                 G(λ), B
               │                               │
   A_s ──►  π_j(z_j) ──┐                       ▼
  [images]     ▲       ├──►  c_jp  ◄──  I_j ~ Exp(Σ)
               │       │      ▲
              z_j      │      │
                       └──── b_j ~ Gamma(bin)
          [shoeboxes j]     [pixels p]
```

### What nanoBragg computes for one pixel

The kernel (`src/integrator/physics/nanobragg.py`) is a line-for-line port of `nanoBragg.c`.
For a pixel with unit diffracted direction $\hat s_1$ and incident direction $\hat s_0$, the scattering vector at wavelength $\lambda$ is $S = (\hat s_1 - \hat s_0)/\lambda$, and nanoBragg sums

$$
I_{\mathrm{pix}} \propto \mathrm{polar} \cdot \Omega \sum_{\text{domains } m} \int S(\lambda)\, F_{\mathrm{cell}}^2\, F_{\mathrm{latt}}^2\!\left(R_m^{-1} S\right) d\lambda ,
$$

where $R_m$ are the mosaic-domain rotations, $\Omega$ the pixel solid angle and polar the Kahn polarization factor.
Two facts make this a profile:

- Only the target reflection's lattice point is rendered, so $F_{\mathrm{cell}}$ is one constant per shoebox and cancels when the image is normalized over the box.
  The normalized result is $\pi_j$, and it does not depend on the structure factor.
  A neighbouring reflection inside the box is treated as background to this profile.
- With nanoBragg's `gauss` crystal shape, $F_{\mathrm{latt}}^2 = N^2 \exp(-\alpha\, |S - R_m q_h|^2)$ with $N = N_a N_b N_c$ and $\alpha = 2N^2/0.63$.
  Along a pixel, $S = k u$ with $k = 1/\lambda$ and $u = \hat s_1 - \hat s_0$ fixed, so the wavelength integral is a one-dimensional Gaussian integral done in closed form, with the spectrum taken at the crossing wavelength.
  This replaces nanoBragg's discrete wavelength grid by its limit (`render_shoeboxes_continuous`).

The centroid shift $z_j$ enters by sampling the rendered pattern at $p - z_j$, so the kernel is evaluated at fractional pixel positions; it is differentiable in $z_j$.

## Inference

All variational posteriors are amortized: one network maps a shoebox to its posterior parameters.

| latent | posterior | produced by |
|---|---|---|
| $I_j$ | $\mathrm{Gamma}$ (shape-rate) | intensity encoders `k`, `r` |
| $b_j$ | $\mathrm{Gamma}$ (shape-rate) | the same two encoders, separate heads |
| $z_j$ | $\mathcal{N}(\mu_j, \mathrm{diag}\,\sigma_j^2)$ | profile encoder, then a linear head `qp.head` (zero-initialized, so $q(z_j)$ starts at the prior) |
| $\Theta$ | none (point estimate) | `nn.Parameter`s in the surrogate |

The integrator is `hierarchical_3enc` (encoders `profile`, `k`, `r`, output width 32), fed Anscombe-standardized counts.
The loss is the negative ELBO,

$$
-\mathcal{L} = -\mathbb{E}_q\!\left[\sum_p \log \mathrm{Poisson}(c_{jp} \mid I_j \pi_{jp} + b_j)\right] + \mathrm{KL}(q(I_j)\,\|\,p(I_j)) + \mathrm{KL}(q(b_j)\,\|\,p(b_j)) + \mathrm{KL}(q(z_j)\,\|\,\mathcal{N}(0, \tau^2)),
$$

estimated with 50 Monte Carlo draws of $(I_j, b_j)$ per step.
The profile is rendered once per step at a single reparameterized draw of $z_j$ and shared by those 50 draws, which keeps the estimate unbiased while rendering once.
At evaluation the profile is rendered at $z_j = \mu_j$.

## Training data

Derek Mendez's simulated ecDHFR Laue data, `hkl_data` set: 180 stills, rotating 1° per shot, generated with nanoBragg through simtbx.

| | |
|---|---|
| source | `/n/netscratch/hekstra_lab/Lab/data/move_from_tier_3/data/hkl_data` |
| shoeboxes | `.../dhfr_sim/nanobragg_prior/full180`, 670,139 reflections |
| cut by | `scripts/physics/make_sim_shoeboxes.py` (`cut` in the `ld` env, `finalize` in `integrator-cuda-dev`) |
| geometry | detector and beam from the CBF headers; one panel, 3840 × 3840 px of 0.08854 mm at 200.5 mm |
| orientation | the TRUE $U$, $B$ per shot from the `.h5`, not from indexing |
| reflections | every lattice point with $d \geq 1.2$ Å whose diffracting wavelength is in 1.02–1.18 Å and whose ray hits the panel |
| windows | 21 × 21, centred on `floor(xyzcal.px)` and slid inward at panel edges, as `make_shoeboxes --laue` does |
| masks | pixels with negative counts, and pixels owned by a nearer neighbour (39% of boxes overlap a neighbour) |
| harmonics | 0.2% of spots overlap a harmonic; flagged as `n_harmonics > 1`, not removed |
| split | 80/20 train/validation; 10% flagged `is_test` |

Using the true orientation is deliberate.
It removes indexing error, so any spot the model cannot reproduce points at the physics or the code.
The predictor agrees with the simulation's per-pixel truth maps on 98.8–99.4% of spot centres.

Not known yet, and therefore learned or assumed: the simulation's spectrum shape, crystal size and shape, divergence, and background model.

## Hyperparameters

From `configs/poly/dhfr_sim_nanobragg.yaml`.

| setting | value | meaning |
|---|---|---|
| `shape` | `gauss` | nanoBragg crystal shape; required for the closed-form wavelength integral |
| `wavelength_integration` | continuous (default for gauss) | closed form; `grid` sums `n_lambda` wavelengths instead |
| `init_n_cells` | 10 | starting crystal size; best match on bright spots was 10–12 |
| `init_mosaic_deg` | 0.1 (read from the `.h5`) | starting mosaic spread |
| `n_domains` | 50 | mosaic domains, drawn once (seed 0) and held fixed |
| `spectrum_knots` | 8 | knots of the piecewise-linear log-spectrum across 1.02–1.18 Å, initialized flat |
| `residual` | `centroid` | per-reflection 2D shift; `none` gives pure physics |
| `centroid_prior_px` | 0.5 | prior sd $\tau$ of the shift, in pixels |
| `learn_globals` | true | fit $\Theta$ through the ELBO |
| `encoder_out` | 32 | encoder output width |
| `mc_samples` | 50 | Monte Carlo draws of $(I, b)$ |
| optimizer | Adam, lr $10^{-3}$, no weight decay | one learning rate for everything, including $\Theta$ |
| batch | 256 shoeboxes | |
| loss | `polychromatic_wilson`, 30 resolution bins | Wilson prior with learned $G(\lambda)$ (degree 5) and $B$ (init 30); per-bin Gamma background prior fitted from the data before training |
| trainer | 10 epochs, gradient clipping 1.0, fp32 | |

## What is learned

| quantity | where | shared by |
|---|---|---|
| crystal size $N_a, N_b, N_c$ | `qp.log_n_cells` | all reflections |
| mosaic spread $\eta$ | `qp.log_mosaic` | all reflections |
| spectrum $\log S$ at 8 knots | `qp.log_spectrum` | all reflections |
| centroid-shift posterior | `qp.head` (linear) on the profile encoder | amortized |
| intensity and background posteriors | encoders `k`, `r` and their heads | amortized |
| Wilson spectrum $G(\lambda)$ and $B$ | `loss.spectrum`, `loss.raw_B` | all reflections |

Two caveats about $\Theta$:

- With the `gauss` shape, nanoBragg's lattice factor depends on $N_a, N_b, N_c$ only through their product $N$.
  The three are therefore one quantity; their gradients are identical, they stay equal, and only $N^{1/3}$ is identified.
  An anisotropic crystal needs the `square` or `round` shape, which uses the wavelength grid.
- The two spectra are different objects.
  $S(\lambda)$ in the profile is the spectral density inside one spot (it shapes and shifts the spot); $G(\lambda)$ in the loss is the intensity scale across spots.
  Only the shape of $S$ matters to the profile, because each profile is normalized.

## What the model does not do yet

- No orientation refinement: $A_s$ is the truth.
  On real data it would be the indexed setting matrix, held fixed.
- No partiality or structure factors: the unnormalized render (the partiality times geometric factor $K_j$) is computed but thrown away, and intensities are per observation, not merged.
- No divergence, detector point-spread, thickness or phi rotation in the kernel.
- No neighbour modelling: a neighbour's photons in the box are background.
- Laue only, stills only, one panel.
- The closed-form wavelength integral assumes the spectrum is smooth across a lattice point's crossing; it is exact to $10^{-4}$ in that regime and off by several percent when the spectrum varies on that scale (near band edges).

## Validation so far

- Kernel vs compiled `nanoBragg.c`: worst pixel $5.9 \times 10^{-8}$ relative over 8 cases (all shapes, oversampling, polarization, 25 mosaic domains, 9 sources), the float32 floor of the C output.
- Batched kernel vs single-shoebox kernel: $10^{-10}$ on boxes centred on real spots.
- Closed-form wavelength integral vs a 30,001-point grid: $10^{-4}$ when the spectrum is smooth over the crossing.
- `gradcheck` passes for crystal size, mosaic spread, wavelength and pixel shift.
- On the 60 brightest isolated spots of 3 shots, the rendered profile matches the observed background-subtracted spot with median cosine similarity 0.876 at 12 cells, and its centroid sits 0.5 px from the observed one.
- One CPU epoch on 3 shots through `integrator.train`: finite ELBO, $N$ 10 → 10.5, $\eta$ 0.100° → 0.095° (truth 0.1°).

## Where the pieces live

| file | role |
|---|---|
| `src/integrator/physics/nanobragg.py` | kernel: `render_pixels`, `render_shoeboxes`, `render_shoeboxes_continuous`, mosaic domains |
| `src/integrator/physics/laue_predict.py` | reflections from a known orientation |
| `src/integrator/physics/shoebox_geometry.py` | batch metadata → pixel indices, detector, per-image crystals |
| `src/integrator/model/distributions/nanobragg_profile.py` | `NanoBraggProfileSurrogate` (`nanobragg_profile`) |
| `scripts/physics/make_sim_shoeboxes.py` | cut the simulation at the true orientation |
| `scripts/physics/validate_nanobragg_port.py` | pixel-for-pixel check against the C program |
| `scripts/physics/check_sim_registration.py` | registration and shape check on real shoeboxes |
| `configs/poly/dhfr_sim_nanobragg.yaml` | the training config |
| `tests/test_nanobragg_kernel.py`, `tests/test_nanobragg_profile.py` | tests |

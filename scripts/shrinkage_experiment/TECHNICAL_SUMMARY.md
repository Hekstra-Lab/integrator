# Technical Summary: Shrinkage Bias Experiment

## 1. Data Setup

### 1.1 Source Data

The experiment loads two files from the pre-simulated dataset at
`dirichlet_profile/`:

- `profiles.pt`: tensor of shape $(200\,000, 441)$, dtype `float32`.
  Each row is a normalized profile $\mathbf{p}_i$ with $\sum_{j=1}^{441} p_{ij} = 1$.
  Values range from $1.175 \times 10^{-38}$ to $0.187$.
  The 441 pixels correspond to a flattened $21 \times 21$ shoebox.

- `reference.pt`: a dictionary containing (among other keys)
  `"intensity"` of shape $(200\,000,)$ and `"background"` of shape $(200\,000,)$.
  Intensities were drawn from $\mathrm{Exp}(0.001)$ (mean 1000, range $[0.005, 12\,966]$).
  Backgrounds were drawn from $\mathrm{Exp}(1.0)$ (mean 1, range $[7 \times 10^{-7}, 12.3]$).

### 1.2 Stratified Subsampling

From the 200,000 reflections, $N = 1000$ are selected by deterministic stratification on $I_{\mathrm{true}}$:

1. Sort all 200,000 indices by ascending $I_{\mathrm{true}}$.
2. Compute stride $= \lfloor 200\,000 / 1000 \rfloor = 200$.
3. Take indices $\{0, 200, 400, \ldots, 199\,800\}$ from the sorted order.
4. Truncate to the first 1000.

This yields approximately uniform coverage across the $I_{\mathrm{true}}$ range.
The resulting subsets are:
- `profiles`: $(1000, 441)$
- `I_true`: $(1000,)$, range $[0.005, 6958]$
- `B_true`: $(1000,)$, range $[0.002, 7.15]$

### 1.3 Count Simulation

Both cases use the **same** `(I_true, profiles, B_true)` triplet.
The random seed is set to 42 via `torch.manual_seed(42)` before each simulation.

**Case 1** (zero background). For each reflection $i$ and pixel $j$:
$$
\lambda_{ij}^{(1)} = I_{\mathrm{true},i} \cdot p_{ij}, \qquad
x_{ij}^{(1)} \sim \mathrm{Pois}(\lambda_{ij}^{(1)}).
$$
Implemented as:
```python
rates = I_true.unsqueeze(1) * profiles   # (1000, 441)
counts = torch.poisson(rates)            # (1000, 441)
```

**Case 2** (nonzero background). For each reflection $i$ and pixel $j$:
$$
\lambda_{ij}^{(2)} = I_{\mathrm{true},i} \cdot p_{ij} + B_{\mathrm{true},i}, \qquad
x_{ij}^{(2)} \sim \mathrm{Pois}(\lambda_{ij}^{(2)}).
$$
Note that $B_{\mathrm{true},i}$ is a scalar per reflection (constant across pixels).
Implemented as:
```python
rates = I_true.unsqueeze(1) * profiles + B_true.unsqueeze(1)
counts = torch.poisson(rates)
```

**Important**: Case 1 and Case 2 use **different** random seeds in practice. Case 1 calls `torch.manual_seed(seed)` inside `run_case1`, and Case 2 calls `torch.manual_seed(seed)` inside `run_case2_direct`. Since Case 1 runs first and consumes RNG state, the Poisson draws for Case 2 are from a different RNG state despite using the same seed value, because the seed is re-set at the entry of each function. The counts are therefore independent draws.

**However**, the encoder experiment (`case2_encoder`) receives the **same** `counts` tensor as Case 2 direct (passed explicitly as an argument), ensuring a fair comparison between the direct and amortized approaches.

### 1.4 Prior Settings

Two prior configurations are tested sequentially:

| Label | $\alpha_0$ | $\beta_0$ | $\mu_0 = \alpha_0/\beta_0$ | $w = \beta_0 / (\beta_0 + 1)$ |
|-------|-----------|----------|---------------------------|-------------------------------|
| Strong | 2.0 | 0.1 | 20.0 | 0.0909 |
| Weak | 2.0 | 0.02 | 100.0 | 0.0196 |

---

## 2. Case 1: Analytical Conjugate Posterior

**File**: `case1.py`, function `run_case1`.

### 2.1 Model

Under $B_i = 0$ and known profile, the generative model is:
$$
X_{ij} \mid I_i \sim \mathrm{Pois}(I_i \, p_{ij}), \qquad I_i \sim \mathrm{Gamma}(\alpha_0, \beta_0).
$$
The Poisson–Gamma conjugacy gives the exact posterior:
$$
I_i \mid \mathbf{X}_i \sim \mathrm{Gamma}(\alpha_0 + S_i, \; \beta_0 + 1),
$$
where $S_i = \sum_{j=1}^{441} x_{ij}$ is the total count.

### 2.2 Computed Quantities

The code computes, per reflection $i$:

$$
\alpha_i^* = \alpha_0 + S_i, \qquad \beta_i^* = \beta_0 + 1, \qquad \mu_i^* = \frac{\alpha_i^*}{\beta_i^*} = \frac{\alpha_0 + S_i}{\beta_0 + 1}.
$$

**Bias** (per-reflection, single-sample):
$$
\mathrm{bias}_i = \mu_i^* - I_{\mathrm{true},i} = \frac{\alpha_0 + S_i}{\beta_0 + 1} - I_{\mathrm{true},i}.
$$

**Predicted bias** (the deterministic/frequentist expected bias from the theorem):
$$
\mathrm{predicted\_bias}_i = w \cdot (\mu_0 - I_{\mathrm{true},i}), \qquad w = \frac{\beta_0}{\beta_0 + 1}, \quad \mu_0 = \frac{\alpha_0}{\beta_0}.
$$
This is the expectation $\mathbb{E}_{S_i}[\mu_i^*] - I_{\mathrm{true},i}$ over data realizations, evaluated at the known $I_{\mathrm{true},i}$.
It differs from `bias_i` by shot noise: $\mathrm{bias}_i = \mathrm{predicted\_bias}_i + \frac{S_i - I_{\mathrm{true},i}}{\beta_0 + 1}$.

### 2.3 Log-Evidence (ELBO)

Since the optimal variational distribution equals the exact posterior,
$\mathrm{KL}(q^* \| p(I_i \mid \mathbf{X}_i)) = 0$ and
$\mathrm{ELBO}_i = \log p(\mathbf{X}_i)$ exactly.

The log-evidence is computed in closed form as:
$$
\log p(\mathbf{X}_i) = \underbrace{\sum_{j=1}^{441} \bigl[ x_{ij} \log p_{ij} - \log \Gamma(x_{ij}+1) \bigr]}_{\text{profile \& factorial terms}}
+ \underbrace{\alpha_0 \log \beta_0 - \log \Gamma(\alpha_0)}_{\text{prior normalizer}}
+ \underbrace{\log \Gamma(\alpha_0 + S_i) - (\alpha_0 + S_i) \log(\beta_0 + 1)}_{\text{posterior normalizer}}.
$$

The profile term uses `torch.log(profiles.clamp(min=1e-38))` to handle near-zero profile entries.
The gamma function is computed via `torch.lgamma` and `math.lgamma`.

---

## 3. Case 2 Direct Optimization

**File**: `case2_direct.py`, function `run_case2_direct`.

### 3.1 Learnable Parameters

There are exactly $2N = 2000$ scalar learnable parameters: `raw_alpha` and `raw_beta`, each of shape $(N,) = (1000,)$. These are unconstrained real-valued tensors.

**Transformation to distribution parameters**:
$$
\alpha_i = \mathrm{softplus}(\texttt{raw\_alpha}_i) + 10^{-6}, \qquad
\beta_i = \mathrm{softplus}(\texttt{raw\_beta}_i) + 10^{-6},
$$
where $\mathrm{softplus}(x) = \log(1 + e^x)$ and $\varepsilon = 10^{-6}$.

**Initialization**: the raw parameters are initialized so that $(\alpha_i, \beta_i)$ starts near the conjugate posterior (pretending $B = 0$):
$$
\alpha_{\mathrm{init},i} = \max(\alpha_0 + S_i, \; 1.0), \qquad
\beta_{\mathrm{init},i} = \beta_0 + 1.
$$
The inverse softplus is applied: $\texttt{raw} = \mathrm{softplus}^{-1}(\alpha_{\mathrm{init}} - \varepsilon)$, implemented as:
```python
def inv_softplus(x):
    return torch.where(x > 20, x, torch.log(torch.expm1(x.clamp(min=eps))))
```

### 3.2 Loss Function

The loss is $\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \widehat{\mathrm{ELBO}}_i$, where the per-reflection ELBO estimate is:

$$
\widehat{\mathrm{ELBO}}_i = \underbrace{\frac{1}{S} \sum_{s=1}^{S} \sum_{j=1}^{441} \log p(x_{ij} \mid I_i^{(s)}, p_{ij}, B_i)}_{\text{MC estimate of } \mathbb{E}_{q}[\log p(\mathbf{X}_i \mid I_i)]} - \underbrace{\mathrm{KL}\bigl(\mathrm{Gamma}(\alpha_i, \beta_i) \;\|\; \mathrm{Gamma}(\alpha_0, \beta_0)\bigr)}_{\text{analytic KL}},
$$

with $S = 100$ MC samples (the `mc_samples` parameter).

**Expected log-likelihood (MC estimate)**:

The samples $I_i^{(s)} \sim \mathrm{Gamma}(\alpha_i, \beta_i)$ are drawn via the **reparameterization trick** (`q.rsample`), producing a tensor of shape $(100, 1000)$.

The per-sample, per-pixel Poisson log-likelihood is:
$$
\log p(x_{ij} \mid I_i^{(s)}) = x_{ij} \log\bigl(I_i^{(s)} p_{ij} + B_i + \varepsilon\bigr) - \bigl(I_i^{(s)} p_{ij} + B_i\bigr) - \log \Gamma(x_{ij} + 1),
$$
where $\varepsilon = 10^{-6}$ is added **inside the log only** (not to the rate in the linear term).

This is summed over pixels ($j = 1, \ldots, 441$) to get shape $(100, 1000)$, then averaged over MC samples (dim 0) to get shape $(1000,)$.

**KL divergence**: computed analytically via `torch.distributions.kl.kl_divergence(Gamma(α,β), Gamma(α₀,β₀))`, which returns a tensor of shape $(1000,)$. The closed-form Gamma–Gamma KL is:
$$
\mathrm{KL} = (\alpha - \alpha_0)\psi(\alpha) - \log\Gamma(\alpha) + \log\Gamma(\alpha_0) + \alpha_0(\log\beta - \log\beta_0) + \alpha\frac{\beta_0 - \beta}{\beta},
$$
where $\psi$ is the digamma function.

**Aggregation**: the loss is the negative mean ELBO across all $N$ reflections:
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \widehat{\mathrm{ELBO}}_i.
$$

### 3.3 Optimization

- **Optimizer**: Adam with learning rate $0.01$.
- **Schedule**: cosine annealing over $T_{\max} = 3000$ steps to $\eta_{\min} = 0.01 \times 0.01 = 10^{-4}$.
- **Gradient clipping**: `clip_grad_norm_` with `max_norm=10.0` applied jointly to `[raw_alpha, raw_beta]`.
- **Steps**: 3000 (no mini-batching; all 1000 reflections processed in each step).
- **No early stopping**.

### 3.4 Final Parameter Extraction

After optimization, the final parameters are:
$$
\alpha_i = \mathrm{softplus}(\texttt{raw\_alpha}_i) + 10^{-6}, \qquad
\beta_i = \mathrm{softplus}(\texttt{raw\_beta}_i) + 10^{-6}, \qquad
\mu_i = \frac{\alpha_i}{\beta_i}.
$$
The bias is $\mathrm{bias}_i = \mu_i - I_{\mathrm{true},i}$.

The final ELBO is re-estimated with $10 \times$ more MC samples ($S = 1000$), using the frozen parameters:
```python
I_s = q_final.rsample([mc_samples * 10])  # (1000, 1000)
```
This higher-sample ELBO is what gets compared to the quadrature log-evidence to compute the KL gap.

---

## 4. Case 2 Encoder

**File**: `case2_encoder.py`, function `run_case2_encoder`.

### 4.1 Encoder Architecture

The encoder is a CNN that maps a flattened $441$-dimensional standardized shoebox to Gamma parameters.

**Preprocessing**: the raw counts (shape $(N, 441)$) are Anscombe-transformed and standardized:
$$
\texttt{ans}_{ij} = 2\sqrt{x_{ij} + 3/8}, \qquad
\texttt{std}_{ij} = \frac{\texttt{ans}_{ij} - \bar{a}}{\sigma_a},
$$
where $\bar{a}$ and $\sigma_a$ are the global mean and standard deviation of the Anscombe-transformed counts across all $N \times 441$ entries.

The standardized vector is reshaped to $(B, 1, 21, 21)$ before being passed to the CNN.

**SimpleEncoder** (CNN backbone):

| Layer | Type | Params | Output Shape |
|-------|------|--------|-------------|
| conv1 | Conv2d(1→16, 3×3, pad=1) | $16 \times 1 \times 3 \times 3 + 16 = 160$ | $(B, 16, 21, 21)$ |
| norm1 | GroupNorm(4 groups, 16 ch) | 32 | $(B, 16, 21, 21)$ |
| ReLU | — | — | $(B, 16, 21, 21)$ |
| pool | MaxPool2d(2, 2, ceil=True) | — | $(B, 16, 11, 11)$ |
| conv2 | Conv2d(16→32, 3×3, pad=0) | $32 \times 16 \times 9 + 32 = 4640$ | $(B, 32, 9, 9)$ |
| norm2 | GroupNorm(4 groups, 32 ch) | 64 | $(B, 32, 9, 9)$ |
| ReLU | — | — | $(B, 32, 9, 9)$ |
| conv3 | Conv2d(32→64, 3×3, pad=1) | $64 \times 32 \times 9 + 64 = 18496$ | $(B, 64, 9, 9)$ |
| norm3 | GroupNorm(8 groups, 64 ch) | 128 | $(B, 64, 9, 9)$ |
| ReLU | — | — | $(B, 64, 9, 9)$ |
| adaptive_pool | AdaptiveAvgPool2d(1) | — | $(B, 64, 1, 1)$ |
| squeeze | — | — | $(B, 64)$ |
| fc | Linear(64→64) | $64 \times 64 + 64 = 4160$ | $(B, 64)$ |
| ReLU | — | — | $(B, 64)$ |

Total backbone parameters: $\approx 27{,}680$.

**GammaHead** (distribution parameters):

| Layer | Type | Output |
|-------|------|--------|
| linear_k | Linear(64→1) | $(B, 1)$ |
| linear_r | Linear(64→1) | $(B, 1)$ |

$$
\alpha_i = \mathrm{softplus}(\texttt{linear\_k}(h_i)) + 10^{-6}, \qquad
\beta_i = \mathrm{softplus}(\texttt{linear\_r}(h_i)) + 10^{-6},
$$
where $h_i \in \mathbb{R}^{64}$ is the backbone output.

The head outputs are squeezed from $(B, 1)$ to $(B,)$ before constructing $\mathrm{Gamma}(\alpha_i, \beta_i)$.

Total head parameters: $64 + 1 + 64 + 1 = 130$.
Total model parameters: $\approx 27{,}810$.

### 4.2 Loss Function

The loss function is **identical** to Case 2 direct (Section 3.2):
$$
\mathcal{L}_{\mathrm{batch}} = -\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left[ \frac{1}{S}\sum_{s=1}^{S} \sum_{j=1}^{441} \log p(x_{ij} \mid I_i^{(s)}, p_{ij}, B_i) - \mathrm{KL}(\mathrm{Gamma}(\alpha_i, \beta_i) \| \mathrm{Gamma}(\alpha_0, \beta_0)) \right],
$$
with $S = 100$ MC reparameterized samples and analytic KL.

The profile $\mathbf{p}_i$ and background $B_i$ are **not** inputs to the encoder. They enter **only** in the likelihood computation when constructing the rate $\lambda_{ij}^{(s)} = I_i^{(s)} p_{ij} + B_i$. The encoder sees only the standardized shoebox.

### 4.3 Training

- **Optimizer**: Adam with learning rate $10^{-3}$ (no scheduler).
- **Epochs**: 500, with random permutation of the $N = 1000$ reflections each epoch.
- **Batch size**: 128, giving $\lceil 1000/128 \rceil = 8$ batches per epoch.
- **Gradient clipping**: `clip_grad_norm_` with `max_norm=1.0` on all model parameters.
- **No weight decay, no dropout.**

### 4.4 Final Parameter Extraction

After training, the model is set to `eval()` mode. All $N$ reflections are processed in batches of 128 with gradients disabled. The final parameters are:
$$
\alpha_i = q_i.\texttt{concentration}, \qquad \beta_i = q_i.\texttt{rate}, \qquad \mu_i = \frac{\alpha_i}{\beta_i}, \qquad \mathrm{bias}_i = \mu_i - I_{\mathrm{true},i}.
$$

**Note**: unlike Case 2 direct, no re-estimated ELBO with extra MC samples is computed for the encoder. The quadrature comparison (KL gap) is only computed for Case 2 direct.

---

## 5. Quadrature

**File**: `quadrature.py`, functions `log_evidence_quadrature` and `batch_log_evidence`.

### 5.1 Integrand

For a single reflection $i$ with counts $\mathbf{x}_i$, profile $\mathbf{p}_i$, and background $B_i$:
$$
\log p(\mathbf{X}_i) = \log \int_0^\infty p(\mathbf{X}_i \mid I) \, p(I) \, dI,
$$
where the log-integrand (evaluated on a grid of $I$ values) is:
$$
f(I) = \sum_{j=1}^{441} \Bigl[ x_{ij} \log(I \, p_{ij} + B_i) - (I \, p_{ij} + B_i) - \log\Gamma(x_{ij} + 1) \Bigr] + (\alpha_0 - 1)\log I - \beta_0 I + \alpha_0 \log \beta_0 - \log\Gamma(\alpha_0).
$$

### 5.2 Grid

The integration grid is **fixed** (not adaptive to the posterior shape):

- Lower bound: $I_{\min} = 10^{-6}$.
- Upper bound: $I_{\max} = \max(5 S_i, \; 5 \mu_0, \; 10\,000)$, where $S_i = \sum_j x_{ij}$.
- Number of points: $Q = 30\,000$.
- Spacing: uniform, $\Delta I = (I_{\max} - 10^{-6}) / (Q - 1)$.

All computations are performed in `float64` (the grid, counts, and profiles are cast to `torch.float64`).

### 5.3 Numerical Stability

The log-integrand $f(I_k)$ is computed for each grid point $k = 1, \ldots, Q$, accumulating through a loop over the 441 pixels. The integral is then approximated by the **left-endpoint rectangle rule** (not trapezoidal — no $1/2$ weighting at endpoints):
$$
\log p(\mathbf{X}_i) \approx f_{\max} + \log\left( \sum_{k=1}^{Q} \exp\bigl(f(I_k) - f_{\max}\bigr) \cdot \Delta I \right),
$$
where $f_{\max} = \max_k f(I_k)$. This is a standard log-sum-exp stabilization: the exponentials are shifted by the maximum to avoid overflow.

### 5.4 KL Gap

The KL gap is defined per-reflection as:
$$
\text{KL gap}_i = \log p(\mathbf{X}_i) - \mathrm{ELBO}_i.
$$

For **Case 1**, the ELBO comes from the closed-form log-evidence (Section 2.3), so $\text{KL gap}_i$ measures the quadrature error (should be $\approx 0$).

For **Case 2 direct**, the ELBO comes from the re-estimated ELBO with 1000 MC samples (Section 3.4).

No KL gap is computed for the encoder (Case 2 encoder does not produce an ELBO estimate in the saved results).

### 5.5 Validation

For Case 1 ($B_i = 0$), the quadrature is passed `B = torch.zeros_like(B_true)`, meaning the quadrature integrand uses the exact same zero-background likelihood as the analytic formula. The mean KL gap was:
- Strong prior: $6.4 \times 10^{-5}$ nats
- Weak prior: $8.9 \times 10^{-5}$ nats

This confirms the quadrature is accurate to $< 10^{-4}$ nats against the known closed-form answer.

---

## 6. Plotting Definitions

**File**: `plotting.py`, functions `_bin_data` and `make_plots`.

### 6.1 Bias

The bias plotted for each method is the **per-reflection, single-sample** quantity:
$$
\mathrm{bias}_i = \mu_i - I_{\mathrm{true},i},
$$
where $\mu_i = \alpha_i / \beta_i$ is the posterior (or surrogate) mean.

This is **not** the frequentist expected bias $\mathbb{E}_{S_i}[\mu_i] - I_{\mathrm{true},i}$. It includes shot noise: the realized $S_i$ may differ from $I_{\mathrm{true},i}$.

### 6.2 Predicted Bias

The predicted bias line is evaluated **per $I_{\mathrm{true}}$ value** (not binned):
$$
\mathrm{predicted\_bias}(I) = w \cdot (\mu_0 - I), \qquad w = \frac{\beta_0}{\beta_0 + 1}.
$$
It is plotted as a continuous line over sorted $I_{\mathrm{true}}$ values.

### 6.3 Effective Shrinkage Weight

The effective shrinkage weight is computed per-reflection as:
$$
\tilde{w}_i = \frac{\mu_i - S_i}{\mu_0 - S_i},
$$
where $S_i$ is the total count from **Case 2** (with background), and $\mu_i$ is the posterior mean from either Case 2 direct or Case 2 encoder.

**Important**: this uses $S_i$ (total observed counts including background photons), not $I_{\mathrm{true},i}$.

A validity mask excludes reflections where $|\mu_0 - S_i| \leq \max(50, \; 0.05 \, \mu_0)$:
- For the strong prior ($\mu_0 = 20$): threshold is $\max(50, 1) = 50$.
- For the weak prior ($\mu_0 = 100$): threshold is $\max(50, 5) = 50$.

Reflections failing this test are set to NaN and excluded from binning.

For the encoder, $S_i$ comes from the **Case 2 direct** counts (since the encoder uses the same counts; `c2e["S"] = c2d["S"]` is set explicitly in `run.py`).

### 6.4 Binning

Binning is performed by the `_bin_data` function with default `n_bins=12` and `log_spaced=True`.

**Bin edges**: $12 + 1 = 13$ edges, log-spaced from $\log_{10}(\max(I_{\min}, 1.0))$ to $\log_{10}(I_{\max})$ via `np.logspace`.
With the actual data: from $\log_{10}(1.0) = 0$ to $\log_{10}(6957.7) \approx 3.84$.

**Bin centers**: geometric mean of bin edges, $c_k = \sqrt{e_k \cdot e_{k+1}}$.

**Assignment**: reflection $i$ is in bin $k$ if $e_k \leq I_{\mathrm{true},i} < e_{k+1}$.
The last bin uses $\leq$ on the right boundary.

**Minimum count**: bins with fewer than 3 reflections are excluded.

**Plotted statistic**: the **mean** of the values within each bin.

**Error bars**: the **standard error of the mean** (SEM):
$$
\mathrm{SEM}_k = \frac{\mathrm{std}(\{v_i : i \in \text{bin } k\})}{\sqrt{n_k}},
$$
where $n_k$ is the number of reflections in bin $k$. The standard deviation uses the default `np.ndarray.std()` (population std, denominator $n$, not $n-1$).

---

## 7. Verification

### 7.1 Quadrature vs Analytic (Case 1)

The quadrature function `batch_log_evidence` is called on the Case 1 counts with $B_i = 0$ for all reflections. The result is compared to the closed-form $\log p(\mathbf{X}_i)$ from `case1.py`:

$$
\text{error}_i = \left| \log p(\mathbf{X}_i)_{\text{quad}} - \log p(\mathbf{X}_i)_{\text{analytic}} \right|.
$$

The mean error across 1000 reflections is reported as the "Case 1 KL gap." Observed values:
- Strong prior: $6.4 \times 10^{-5}$ nats
- Weak prior: $8.9 \times 10^{-5}$ nats

There is **no explicit tolerance check** in the code — the value is printed and the user inspects it. No assertion or test halts execution if the gap is unexpectedly large.

### 7.2 Case 2 KL Gap Interpretation

For Case 2, the KL gap $= \log p(\mathbf{X}_i)_{\text{quad}} - \widehat{\mathrm{ELBO}}_i$ includes two sources of error:
1. The true $\mathrm{KL}(q^*_{\text{Gamma}} \| p(I_i \mid \mathbf{X}_i))$ — the irreducible gap from using a Gamma surrogate.
2. MC noise in the ELBO estimate (using 1000 samples; see Section 3.4).
3. Any sub-optimality in the Adam optimization (the surrogate may not have fully converged to $q^*$).

The reported mean KL gaps (2.09 and 1.77 nats for strong and weak priors respectively) are therefore upper bounds on the true approximation error of the Gamma family.

# Gamma Reparameterization Analysis

## Overview

We use a Gamma posterior $q(I) = \text{Gamma}(k, r)$ for Bragg reflection intensities in our variational integrator. The inference network predicts two raw (unconstrained) parameters per reflection, which are mapped to the Gamma's concentration $k$ and rate $r$ through one of four reparameterizations (A--D). All four produce the same distributional family but differ in how raw network outputs map to $(k, r)$, and this mapping has significant consequences for gradient flow, posterior calibration, and training stability.

## The four reparameterizations

| Repam | Raw params | Mapping | $k$ depends on | $r$ depends on |
|-------|-----------|---------|----------------|----------------|
| A | $(\text{raw}_k, \text{raw}_r)$ | $k = \text{sp}(\text{raw}_k) + k_{\min}$, $r = \text{sp}(\text{raw}_r) + \epsilon$ | $\text{raw}_k$ only | $\text{raw}_r$ only |
| B | $(\text{raw}_\mu, \text{raw}_F)$ | $\mu = \text{sp}(\text{raw}_\mu) + \epsilon$, $F = \text{sp}(\text{raw}_F) + \epsilon$, $r = 1/F$, $k = \mu r$ | both | $\text{raw}_F$ only |
| C | $(\text{raw}_\mu, \text{raw}_\phi)$ | $\mu = \text{sp}(\text{raw}_\mu) + \epsilon$, $\phi = \text{sp}(\text{raw}_\phi) + \epsilon$, $k = 1/\phi$, $r = 1/(\phi\mu)$ | $\text{raw}_\phi$ only | both |
| D | $(\text{raw}_k, \text{raw}_F)$ | $k = \text{sp}(\text{raw}_k) + k_{\min}$, $F = \text{sp}(\text{raw}_F) + \epsilon$, $r = 1/F$ | $\text{raw}_k$ only | $\text{raw}_F$ only |

where $\text{sp}(x) = \log(1 + e^x)$ is the softplus function, $k_{\min} = 0.1$, and $\epsilon = 10^{-6}$.

## IRG reliability analysis

The foundational constraint on all four reparameterizations was established by analysis of the Implicit Reparameterization Gradient (IRG, Figurnov et al. 2018) used by both PyTorch and TensorFlow Probability to differentiate through Gamma samples.

**The IRG for the concentration parameter** $\alpha$ is:

$$\frac{\partial x}{\partial \alpha} = -\frac{\partial F(x;\alpha)/\partial \alpha}{f(x;\alpha)}$$

where $F$ and $f$ are the Gamma CDF and PDF evaluated at the sample $x$. For the rate parameter $\beta$, the gradient is the exact scale relation $\partial x / \partial \beta = -x/\beta$.

**Key findings:**

1. **Concentration gradient breakdown at small $\alpha$**: For $\alpha \lesssim 0.01$ in float32, the IRG breaks down in *both* PyTorch and TFP. The Gamma sample $x$ itself underflows toward zero (the Marsaglia-Tsang sampler draws $x \approx \alpha$ for small $\alpha$, and float32 cannot represent values below $\sim 1.2 \times 10^{-38}$). Once $x$ underflows, both the numerator $\partial F/\partial\alpha$ and denominator $f(x;\alpha)$ collapse, producing silently zero or garbage gradients.

2. **Rate gradient is more robust**: The rate gradient $-x/\beta$ fails only when $x$ itself underflows, which happens at a more extreme regime. TFP's `log_rate` branch computes $-x$ directly (avoiding a two-step chain rule through $(-x/\beta) \cdot \beta$), giving additional stability near the underflow boundary.

3. **The `log_rate` branch does not fix concentration gradients**: Switching to TFP's `log_rate` parameterization changes only the rate-side gradient path. The concentration gradient formula is identical in both branches. If the network can produce small $\alpha$, the concentration gradient is lost regardless.

**Practical consequence**: The inference network must be designed so that $k > 0.1$ in float32. This is enforced in our implementation via $k_{\min} = 0.1$ (softplus + offset for A/D; clamp for B/C). With this floor, all four reparameterizations produce reliable gradients with 0% NaN and 0% zero-gradient occurrences across all tested operating points.

## Why different reparameterizations produce different results

Given that all four map to the same Gamma family and all have reliable gradients (with $k_{\min} = 0.1$), the differences in training behavior arise from the **Jacobian structure** of the mapping $(\text{raw}_1, \text{raw}_2) \to (k, r)$. This Jacobian determines how ELBO gradients propagate back to the network's parameters.

### Jacobian structure

**Repam A** (diagonal):
$$J_A = \begin{pmatrix} \sigma'(\text{raw}_k) & 0 \\ 0 & \sigma'(\text{raw}_r) \end{pmatrix}$$

**Repam D** (diagonal):
$$J_D = \begin{pmatrix} \sigma'(\text{raw}_k) & 0 \\ 0 & -\sigma'(\text{raw}_F)/F^2 \end{pmatrix}$$

**Repam B** (coupled --- $k$ depends on both raw params):
$$J_B = \begin{pmatrix} \sigma'(\text{raw}_\mu) \cdot r & -\mu \cdot \sigma'(\text{raw}_F)/F^2 \\ 0 & -\sigma'(\text{raw}_F)/F^2 \end{pmatrix}$$

**Repam C** (coupled --- $r$ depends on both raw params):
$$J_C = \begin{pmatrix} 0 & -\sigma'(\text{raw}_\phi)/\phi^2 \\ -\sigma'(\text{raw}_\mu)/(\phi\mu^2) & -\sigma'(\text{raw}_\phi)/(\phi^2\mu) \end{pmatrix}$$

where $\sigma'(x) = \text{sigmoid}(x)$ is the softplus derivative.

A/D have diagonal Jacobians: each raw parameter controls exactly one distributional parameter. B/C have off-diagonal entries: a single raw parameter affects both $k$ and $r$, creating coupling.

### Consequence 1: Jacobian conditioning

The condition number $\kappa = \sigma_{\max}/\sigma_{\min}$ of the Jacobian determines how uniformly different parameter directions are learned.

| Repam | $\kappa$ at $k=1$ | $\kappa$ at $k=100$ | $\kappa$ at $k=500$ | Scaling |
|-------|-------------------|---------------------|---------------------|---------|
| A | 1.2 | 2.0 | 2.0 | $O(1)$ |
| B | 2.4 | 3,468 | 86,646 | $O(k)$ |
| C | 3.2 | 3.7 | 3.7 | $O(1)$ |
| D | 1.8 | 1.0 | 1.0 | $O(1)$ |

B's condition number grows linearly with $k$. At $k = 500$, one direction in parameter space learns 86,000$\times$ faster than the other. This creates severe optimization difficulties for strong reflections.

### Consequence 2: Gradient competition

When the Jacobian is coupled, the ELBO gradient through a single raw parameter receives contributions from both $\partial L/\partial k$ and $\partial L/\partial r$. If these contributions have opposite signs, they partially cancel --- the optimizer receives a weak net signal despite strong individual gradients.

Measured Pearson correlation between $\nabla_{\text{raw}_1} L$ and $\nabla_{\text{raw}_2} L$ (at $y = 1000$, $k = 1$, $N = 1000$ samples):

| Repam | Correlation | Interpretation |
|-------|-------------|----------------|
| A | $-0.68$ | Moderate (from shared sample, not parameterization) |
| B | $-1.00$ | Perfect competition (from coupling) |
| C | $-0.70$ | Moderate |
| D | $+0.71$ | Cooperative (both params pushed same direction) |

B's perfect anti-correlation means the two gradient components nearly cancel for every sample. D's positive correlation means both parameters receive reinforcing gradient signals --- the optimizer makes consistent progress on every step.

### Consequence 3: Convergence on a realistic ELBO

Optimizing $q(I_i) = \text{Gamma}(k_i, r_i)$ for three reflections with observed counts $y = [5, 50, 500]$, prior $\text{Gamma}(1, 0.001)$, Adam with lr $= 0.01$, 64 MC samples per gradient estimate.

All four reparameterizations eventually converge to the correct posterior means, but at very different speeds:

| Repam | Steps to correct means | Final loss (at 15K) | Learned means at 5K steps |
|-------|----------------------|---------------------|---------------------------|
| A | ~650 | $-2739$ | 5.8, 51.6, 496.5 |
| D | ~7,000 | $-2732$ | 6.0, 51.6, 401.2 |
| B | ~50,000+ (extrap.) | $-2348$ (at 15K) | 6.0, 22.6, 25.3 |
| C | ~50,000+ (extrap.) | $-1745$ (at 5K) | 6.0, 24.0, 27.1 |

**A** converges fastest: all three means are correct within 650 steps. **D** converges to the correct means but requires $\sim$10$\times$ more steps, reaching the $y = 500$ target by step 7,000. **B and C** are not stuck but converge extremely slowly --- at 15K steps, B's mean for $y = 500$ has only reached 102. The coupled Jacobian slows mean learning for large intensities because adjusting the mean-controlling parameter simultaneously perturbs $k$, triggering a competing gradient that partially cancels progress.

A notable pathology of B/C during slow convergence: while the mean is still far from the target, $k$ grows to very large values (B: $k \approx 550$ at mean $= 102$ for the $y = 500$ reflection). The optimizer concentrates the posterior tightly around the *wrong* mean before eventually shifting it to the correct value.

### Consequence 4: Posterior calibration (noise-to-signal ratio)

For a $\text{Gamma}(k, r)$ posterior, the noise-to-signal ratio is $\text{std}/\text{mean} = 1/\sqrt{k}$. Comparing the learned posteriors against theoretical bounds (Poisson limit, Cramer-Rao lower bound, Laplace approximation) reveals a calibration difference:

**Without $k_{\max}$**: Repams A and D produce posteriors that are *overconcentrated* at high intensity --- the scatter falls below the theoretical curves, meaning $k$ is too large. Repams B and C track the theoretical curves correctly.

**With $k_{\max} = 500$**: All repams show an artificial floor at $\text{std}/\text{mean} \geq 1/\sqrt{500} \approx 0.045$, making posteriors *underconcentrated* for strong reflections ($I \gtrsim 10^3$) where the correct $k$ exceeds 500.

**Why B/C are naturally calibrated**: The braking mechanism is *gradient competition through coupling*, not a vanishing Jacobian. In repam B, `raw_fano` controls both $k$ and $r$. When the optimizer tries to increase $k$ (by decreasing the Fano factor), it receives a competing gradient from $\partial L/\partial r$ that resists the change. The NLL constrains the mean $\mu$ tightly to the data, leaving the Fano factor as the only degree of freedom for $k$. But adjusting the Fano factor triggers the competing $r$-gradient, which acts as a data-driven restoring force. This competition prevents $k$ from overshooting.

In repam D, `raw_k` receives *only* $\partial L/\partial k$. There is no competing signal. The NLL prefers narrow posteriors (large $k$ reduces variance), and the KL penalty grows only as $O(\log k)$, which is too weak to counterbalance. So $k$ overshoots.

**The fundamental tension**: The coupling that produces correct calibration in B/C is the same coupling that causes their optimization pathologies (condition number blowup, gradient competition, convergence failure). D's independence gives clean optimization but removes the natural brake on $k$.

### Consequence 5: $k_{\max}$ is not a solution

Setting $k_{\max}$ (via sigmoid bounding) prevents $k$ from exceeding a fixed ceiling, addressing both the numerical NaN at $k \approx 950$ and the overconcentration. But the correct $k$ for strong reflections ($I \approx 10^4$) is $k \approx 1/0.03^2 \approx 1100$, which exceeds any practical $k_{\max}$. The result is an artificial plateau in the noise-to-signal plot where the posterior cannot reach the correct concentration.

## Summary

| Property | A | B | C | D |
|----------|---|---|---|---|
| Jacobian | Diagonal | Coupled ($k$ row) | Coupled ($r$ row) | Diagonal |
| Condition number | $O(1)$ | $O(k)$ | $O(1)$ | $O(1)$ |
| Gradient correlation | $-0.68$ | $-1.00$ | $-0.70$ | $+0.71$ |
| Convergence speed | Fastest (~650 steps) | Very slow (~50K+ steps) | Very slow (~50K+ steps) | Moderate (~7K steps) |
| Calibration (no $k_{\max}$) | Overconcentrated | Correct | Correct | Overconcentrated |
| Clamp dead zones | 0% | 46% | 22% | 0% |

No single reparameterization achieves all desirable properties simultaneously. A/D have clean optimization but overconcentrate. B/C are naturally calibrated but have pathological gradient geometry. Resolving this tension requires either (a) a numerical fix that allows large $k$ without NaN (e.g., CLT-based rsample for $k \gg 1$) combined with a statistically appropriate prior that provides adequate pushback, or (b) a new parameterization that introduces data-adaptive braking without full Jacobian coupling.

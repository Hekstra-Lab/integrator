# Why Reparameterizations Change Neural Network Gradients: A Proof

## The Actual ELBO

Our variational integrator models each Bragg reflection with three
independent surrogates predicted by a shared encoder network:

- **Intensity:** $q_I(I) = \text{Gamma}(k, r)$ — the surrogate whose
  reparameterization we are studying
- **Profile:** $q_P(P)$ — Dirichlet or LogisticNormal (independent of $q_I$)
- **Background:** $q_B(B)$ — Gamma or FoldedNormal (independent of $q_I$)

The Poisson rate for pixel $j$ of a reflection is:

$$\lambda_j = I \cdot P_j + B$$

and the full ELBO for one reflection is:

$$\mathcal{L} = \underbrace{-\mathbb{E}_{q_I q_P q_B}\left[\sum_j \log \text{Poisson}(y_j \mid I \cdot P_j + B)\right]}_{\text{NLL}}
+ \underbrace{\text{KL}(q_I \| p_I)}_{\text{intensity KL}}
+ \underbrace{\text{KL}(q_P \| p_P)}_{\text{profile KL}}
+ \underbrace{\text{KL}(q_B \| p_B)}_{\text{background KL}}$$

---

## Why Profile and Background Factor Out

Consider the gradient w.r.t. the intensity surrogate's raw parameters
$(\text{raw}_1, \text{raw}_2)$, which are the outputs of the linear heads
applied to the encoder representation $h$.

The profile and background KL terms do not depend on
$(\text{raw}_1, \text{raw}_2)$ at all:

$$\frac{\partial}{\partial \text{raw}_1} \text{KL}(q_P \| p_P) = 0, \qquad
\frac{\partial}{\partial \text{raw}_1} \text{KL}(q_B \| p_B) = 0$$

So the gradient of the full ELBO w.r.t. the intensity raw parameters
reduces to:

$$\frac{\partial \mathcal{L}}{\partial (\text{raw}_1, \text{raw}_2)}
= \frac{\partial \text{NLL}}{\partial (\text{raw}_1, \text{raw}_2)}
+ \frac{\partial \text{KL}(q_I \| p_I)}{\partial (\text{raw}_1, \text{raw}_2)}$$

The profile and background terms vanish.  This is not an approximation —
it is an exact consequence of the surrogates being predicted by
**separate linear heads** from the shared encoder.

---

## The Chain Rule Through the Reparameterization

The encoder network with weights $\theta$ produces a hidden
representation $h = f_\theta(x)$.  Linear heads map $h$ to two raw
outputs:

$$\text{raw}_1 = w_1^\top h + b_1, \qquad \text{raw}_2 = w_2^\top h + b_2$$

A **reparameterization** $g$ maps these to the Gamma parameters:

$$(k, r) = g(\text{raw}_1, \text{raw}_2)$$

All four reparameterizations (A–D) produce the same distributional family
$q_I = \text{Gamma}(k, r)$, but $g$ differs.

By the chain rule, the gradient of the loss w.r.t. any network weight
$\theta_j$ (including both encoder weights and the linear head weights) is:

$$\boxed{
\nabla_\theta \mathcal{L}
= \underbrace{\begin{pmatrix} \dfrac{\partial \mathcal{L}_I}{\partial k} & \dfrac{\partial \mathcal{L}_I}{\partial r} \end{pmatrix}}_{\text{loss gradient w.r.t. } (k,r)}
\;\cdot\;
\underbrace{J_g}_{\text{repam Jacobian}}
\;\cdot\;
\underbrace{\dfrac{\partial (\text{raw}_1, \text{raw}_2)}{\partial \theta}}_{\text{network Jacobian}}
}$$

where $\mathcal{L}_I = \text{NLL} + \text{KL}(q_I \| p_I)$ contains only
the terms that depend on $(k, r)$.

This is the complete gradient.  No profile or background terms appear.

---

## Expanding the NLL Gradient

The NLL is computed via the reparameterization trick:

$$\text{NLL} = -\mathbb{E}_{q_I q_P q_B}\left[\sum_j \log \text{Poisson}(y_j \mid I \cdot P_j + B)\right]$$

Using the pathwise estimator, we draw $I \sim q_I$, $P \sim q_P$,
$B \sim q_B$, compute $\lambda_j = I \cdot P_j + B$, and differentiate
through the sample.  Since $I = h(k, r, \epsilon)$ where $\epsilon$ is
the reparameterization noise (from `_standard_gamma`), the gradient of
the NLL w.r.t. $(k, r)$ is:

$$\frac{\partial \text{NLL}}{\partial k} = -\sum_j \frac{\partial \log p(y_j | \lambda_j)}{\partial \lambda_j} \cdot P_j \cdot \frac{\partial I}{\partial k}$$

$$\frac{\partial \text{NLL}}{\partial r} = -\sum_j \frac{\partial \log p(y_j | \lambda_j)}{\partial \lambda_j} \cdot P_j \cdot \frac{\partial I}{\partial r}$$

Here $P_j$ and $B$ are sampled independently and **treated as constants**
w.r.t. the intensity parameters.  The profile and background samples
scale the gradient magnitude but do not change the direction or the
relationship between the $k$ and $r$ components.

The KL term is analytic for Gamma–Gamma:

$$\text{KL}(q_I \| p_I) = (\alpha_q - \alpha_p) \psi(\alpha_q) - \log\Gamma(\alpha_q) + \log\Gamma(\alpha_p) + \alpha_p(\log \beta_q - \log \beta_p) + \alpha_q \frac{\beta_p - \beta_q}{\beta_q}$$

This depends only on $(k, r)$ and the prior parameters.

Both $\partial \mathcal{L}_I / \partial k$ and $\partial \mathcal{L}_I / \partial r$
are determined entirely by $(k, r)$ and the current samples.  They are
**the same for all reparameterizations** at a given $(k, r)$.

---

## What Changes Between Reparameterizations

The three factors in the boxed equation are:

| Factor | Depends on repam? | Why |
|--------|-------------------|-----|
| $(\partial \mathcal{L}_I / \partial k,\; \partial \mathcal{L}_I / \partial r)$ | **No** | Same $(k, r)$, same NLL, same KL |
| $J_g$ | **Yes** | This IS the reparameterization |
| $\partial (\text{raw}_1, \text{raw}_2) / \partial \theta$ | **No** | Same network, same linear heads |

The reparameterization Jacobian $J_g$ is a $2 \times 2$ matrix that sits
between the loss gradient and the network gradient.  It acts as a
**linear transformation** on the gradient signal before it reaches the
network weights.

---

## The Four Jacobians

Let $\sigma'(x) = \text{sigmoid}(x)$ denote the softplus derivative.

### Repam A: $k = \text{sp}(\text{raw}_k) + k_{\min}$, $r = \text{sp}(\text{raw}_r) + \varepsilon$

$$J_A = \begin{pmatrix} \sigma'(\text{raw}_k) & 0 \\ 0 & \sigma'(\text{raw}_r) \end{pmatrix}$$

**Diagonal.**  $\partial\mathcal{L}_I/\partial k$ flows only to $\text{raw}_k$;
$\partial\mathcal{L}_I/\partial r$ flows only to $\text{raw}_r$.

### Repam D: $k = \text{sp}(\text{raw}_k) + k_{\min}$, $r = 1/F$, $F = \text{sp}(\text{raw}_F) + \varepsilon$

$$J_D = \begin{pmatrix} \sigma'(\text{raw}_k) & 0 \\ 0 & -\sigma'(\text{raw}_F)/F^2 \end{pmatrix}$$

**Diagonal.**  Same independence.  The $1/F^2$ scaling changes the
effective learning rate for the rate parameter.

### Repam B: $\mu = \text{sp}(\text{raw}_\mu) + \varepsilon$, $F = \text{sp}(\text{raw}_F) + \varepsilon$, $k = \mu/F$, $r = 1/F$

$$J_B = \begin{pmatrix} \sigma'(\text{raw}_\mu)/F & -\mu\,\sigma'(\text{raw}_F)/F^2 \\ 0 & -\sigma'(\text{raw}_F)/F^2 \end{pmatrix}$$

**Coupled.**  The gradient w.r.t. $\text{raw}_F$ receives contributions
from **both** $\partial\mathcal{L}_I/\partial k$ and $\partial\mathcal{L}_I/\partial r$:

$$\frac{\partial \mathcal{L}_I}{\partial \text{raw}_F}
= \frac{\partial \mathcal{L}_I}{\partial k} \cdot \left(-\frac{\mu\,\sigma'}{F^2}\right)
+ \frac{\partial \mathcal{L}_I}{\partial r} \cdot \left(-\frac{\sigma'}{F^2}\right)$$

When these two terms have opposite signs — which happens whenever the NLL
wants to increase the mean while the KL resists increasing $k$ — they
partially cancel.  This is **gradient competition**.

### Repam C: $\mu = \text{sp}(\text{raw}_\mu) + \varepsilon$, $\phi = \text{sp}(\text{raw}_\phi) + \varepsilon$, $k = 1/\phi$, $r = 1/(\phi\mu)$

$$J_C = \begin{pmatrix} 0 & -\sigma'(\text{raw}_\phi)/\phi^2 \\ -\sigma'(\text{raw}_\mu)/(\phi\mu^2) & -\sigma'(\text{raw}_\phi)/(\phi^2\mu) \end{pmatrix}$$

**Coupled.**  Both $\partial\mathcal{L}_I/\partial k$ and $\partial\mathcal{L}_I/\partial r$
contribute to the gradient of $\text{raw}_\phi$.

---

## Why This Holds for the Full Amortized System

The concern is:

> "you can't make conclusions about amortized distributions by training
> distribution parameters directly"

The chain rule factorization resolves this.  Consider what happens in the
full amortized system with shared encoder weights $\theta$:

**1. The reparameterization Jacobian $J_g$ cannot be "undone" by the network.**

$J_g$ is a multiplicative factor between the loss gradient and the
network gradient.  If $J_g$ has condition number $\kappa$ (Repam B:
$\kappa \sim O(k)$), then one direction in raw-parameter space receives
a gradient $\kappa$ times weaker than the other.  The encoder weights
that project onto the weak direction learn $\kappa$ times slower.

The network cannot compensate for this because $J_g$ acts **after** the
network's output.  It would be like trying to fix a badly-aligned
telescope by adjusting how you hold the camera — the optics are what
they are.

**2. Gradient competition is structural, not an artifact.**

In Repam B, the gradient w.r.t. $\text{raw}_F$ is a sum of two competing
terms (from $\partial\mathcal{L}_I/\partial k$ and
$\partial\mathcal{L}_I/\partial r$).  This cancellation happens **after**
the network produces $\text{raw}_F$ and **before** the gradient reaches
the encoder.  The network architecture — depth, width, activation
functions, amortization — is irrelevant.  The cancellation is a property
of the Jacobian $J_g$, which is determined solely by the reparameterization.

**3. The direct-parameter experiments isolate $J_g$.**

When we optimize raw parameters directly (no network), we set the
rightmost factor $\partial(\text{raw}_1, \text{raw}_2)/\partial\theta = I$
(the identity matrix).  The gradient becomes simply:

$$\nabla_{\text{raw}} \mathcal{L}_I = (\partial\mathcal{L}_I/\partial k, \; \partial\mathcal{L}_I/\partial r) \cdot J_g$$

This isolates $J_g$'s effect.  The properties we measure — condition
number, gradient competition, convergence speed — are properties of
$J_g$ alone.  In the amortized case, the gradient is the same expression
multiplied by the network Jacobian on the right, which is a common
factor shared by all reparameterizations.

**4. The network Jacobian is a common factor.**

Different reparameterizations use the same encoder $f_\theta$ and the
same linear heads (only the activation functions after the heads differ).
The network Jacobian $\partial(\text{raw}_1, \text{raw}_2)/\partial\theta$
is identical across reparameterizations.  It scales the gradient
uniformly — it does not selectively amplify the weak direction to
compensate for $J_g$'s anisotropy.

---

## The Gradient Is the Same Whether or Not You "See" $J_g$

One might object: "I just call `loss.backward()` and PyTorch computes the
gradient automatically — where does $J_g$ appear?"

Answer: $J_g$ is **inside** the autograd computation.  When PyTorch
differentiates through `softplus`, `sigmoid`, division, and multiplication
in the reparameterization, it is computing exactly $J_g \cdot (\partial\text{raw}/\partial\theta)$.
The chain rule is not something we add — it is what `backward()` does.

The factorization $\nabla_\theta \mathcal{L} = (\partial\mathcal{L}/\partial(k,r)) \cdot J_g \cdot (\partial\text{raw}/\partial\theta)$
is not a modeling choice or an approximation.  It is a mathematical
identity that describes what PyTorch autograd computes.

---

## Analogy

Consider two parameterizations of a 2D point:

- Cartesian: $(x, y)$
- Polar: $(r, \phi)$ where $x = r\cos\phi$, $y = r\sin\phi$

Minimize $f(x, y) = (x - 3)^2 + (y - 4)^2$:

- In Cartesian: $\nabla f = (2(x-3), 2(y-4))$ — isotropic.
- In polar: $\nabla_{(r,\phi)} f = J_{\text{polar}}^\top \nabla_{(x,y)} f$
  where $J_{\text{polar}}$ has condition number $\sim r$.  At large $r$,
  the $\phi$ gradient is $r$ times larger than the $r$ gradient.

Now put a neural network in front:
$(\text{raw}_1, \text{raw}_2) = W h + b$.

- Network → Cartesian → loss: clean gradients to $\theta$
- Network → Polar → loss: anisotropic gradients to $\theta$

The network cannot fix the coordinate system's anisotropy, because the
Jacobian of the coordinate transformation sits between the network
output and the loss.  The same applies to our Gamma reparameterizations.

---

## Summary

For our ELBO:

$$\mathcal{L} = -\mathbb{E}[\log \text{Poisson}(y \mid I \cdot P + B)] + \text{KL}(q_I \| p_I) + \underbrace{\text{KL}(q_P \| p_P) + \text{KL}(q_B \| p_B)}_{\text{vanish w.r.t. intensity params}}$$

The gradient w.r.t. any network parameter $\theta$ is:

$$\nabla_\theta \mathcal{L} = \left(\frac{\partial \mathcal{L}_I}{\partial k},\; \frac{\partial \mathcal{L}_I}{\partial r}\right) \cdot J_g \cdot \frac{\partial \text{raw}}{\partial \theta}$$

- The profile and background KL terms contribute zero gradient to the intensity parameters.
- $J_g$ is structurally different for each reparameterization (diagonal vs coupled).
- $J_g$ is a multiplicative factor — the neural network cannot cancel it.
- Different $J_g$ → different $\nabla_\theta \mathcal{L}$ → different optimization trajectories.
- This is the chain rule — an algebraic identity, not a convex-optimization result.
- It holds for any network architecture, convex or non-convex, amortized or not.

# IntegratorModelB Dataflow Graphs


```mermaid
flowchart TD
    subgraph Input
        SB["shoebox (B, 1, D, H, W)"]
        Y["counts (B, P)"]
        M["mask (B, P)"]
    end

    subgraph Encoders
        SB --> ENC_P["profile encoder (CNN)"]
        SB --> ENC_IK["intensity-k encoder (CNN)"]
        SB --> ENC_IR["intensity-r encoder (CNN)"]
        SB --> ENC_BK["background-k encoder (CNN)"]
        SB --> ENC_BR["background-r encoder (CNN)"]
        ENC_P --> x_p["x_p (B, 64)"]
        ENC_IK --> x_ik["x_ik (B, 64)"]
        ENC_IR --> x_ir["x_ir (B, 64)"]
        ENC_BK --> x_bk["x_bk (B, 64)"]
        ENC_BR --> x_br["x_br (B, 64)"]
    end

    subgraph Surrogates
        x_p --> SUR_P["qp surrogate"]
        x_ik --> SUR_I["qi surrogate (Gamma repam)"]
        x_ir --> SUR_I
        x_bk --> SUR_BG["qbg surrogate"]
        x_br --> SUR_BG
        SUR_P --> qp["qp distribution"]
        SUR_I --> qi["qi = Gamma(k, r)"]
        SUR_BG --> qbg["qbg distribution"]
    end

    subgraph Sampling
        qi -- "rsample(S)" --> zI["zI (B, S, 1)"]
        qp -- "rsample(S)" --> zP["zP (B, S, P)"]
        qbg -- "rsample(S)" --> zBG["zBG (B, S, 1)"]
    end

    subgraph Rate
        zI --> MUL["×"]
        zP --> MUL
        MUL --> IP["I·P (B, S, P)"]
        IP --> ADD["+"]
        zBG --> ADD
        ADD --> rate["λ = I·P + B (B, S, P)"]
    end

    subgraph Loss
        rate --> NLL["−E[log Poisson(y | λ)]"]
        Y --> NLL
        M --> NLL
        qi --> KL_I["KL(qi ‖ p_I)"]
        qp --> KL_P["KL(qp ‖ p_P)"]
        qbg --> KL_BG["KL(qbg ‖ p_B)"]
        NLL --> SUM["L = NLL + KL_I + KL_P + KL_BG"]
        KL_I --> SUM
        KL_P --> SUM
        KL_BG --> SUM
    end

    style SUR_I fill:#ff9,stroke:#f90,stroke-width:3px
    style qi fill:#ff9,stroke:#f90,stroke-width:3px
```


## Full Model (shared across all Gamma repams)

The outer dataflow is identical for all reparameterizations.
Only the `qi surrogate` box (highlighted) changes internally.

```mermaid
flowchart TD
    subgraph Input
        SB["shoebox (B, 1, D, H, W)"]
        Y["counts (B, P)"]
        M["mask (B, P)"]
    end

    subgraph Encoders
        SB --> ENC_P["profile encoder (CNN)"]
        SB --> ENC_K["k encoder (CNN)"]
        SB --> ENC_R["r encoder (CNN)"]
        ENC_P --> x_p["x_profile (B, 64)"]
        ENC_K --> x_k["x_k (B, 64)"]
        ENC_R --> x_r["x_r (B, 64)"]
    end

    subgraph Surrogates
        x_p --> SUR_P["qp surrogate (Dirichlet/LogNorm)"]
        x_k --> SUR_I["qi surrogate (Gamma repam)"]
        x_r --> SUR_I
        x_k --> SUR_BG["qbg surrogate"]
        x_r --> SUR_BG
        SUR_P --> qp["qp distribution"]
        SUR_I --> qi["qi = Gamma(k, r)"]
        SUR_BG --> qbg["qbg distribution"]
    end

    subgraph Sampling["Monte Carlo Sampling"]
        qi -- "rsample(S)" --> zI["zI (B, S, 1)"]
        qp -- "rsample(S)" --> zP["zP (B, S, P)"]
        qbg -- "rsample(S)" --> zBG["zBG (B, S, 1)"]
    end

    subgraph Rate["Poisson Rate"]
        zI --> MUL["×"]
        zP --> MUL
        MUL --> IP["I·P (B, S, P)"]
        IP --> ADD["+"]
        zBG --> ADD
        ADD --> rate["λ = I·P + B (B, S, P)"]
    end

    subgraph Loss["ELBO Loss"]
        rate --> NLL["−E[log Poisson(y | λ)]"]
        Y --> NLL
        M --> NLL
        qi --> KL_I["KL(qi ‖ p_I)"]
        qp --> KL_P["KL(qp ‖ p_P)"]
        qbg --> KL_BG["KL(qbg ‖ p_B)"]
        NLL --> SUM["L = NLL + KL_I + KL_P + KL_BG"]
        KL_I --> SUM
        KL_P --> SUM
        KL_BG --> SUM
    end

    style SUR_I fill:#ff9,stroke:#f90,stroke-width:3px
    style qi fill:#ff9,stroke:#f90,stroke-width:3px
```

---

## qi Surrogate Internals: Repam A

`raw_k` and `raw_r` are independent.
$J_g$ is diagonal — no coupling.

```mermaid
flowchart TD
    x_k["x_k (B, 64)"] --> LIN_K["linear_k: x_k → raw_k"]
    x_r["x_r (B, 64)"] --> LIN_R["linear_r: x_r → raw_r"]

    LIN_K --> raw_k["raw_k (B, 1)"]
    LIN_R --> raw_r["raw_r (B, 1)"]

    raw_k --> SP_K["softplus"]
    SP_K --> SP_K_OUT["softplus(raw_k)"]
    SP_K_OUT --> ADD_K["+"]
    K_MIN["k_min"] --> ADD_K
    ADD_K --> k["k = softplus(raw_k) + k_min"]

    raw_r --> SP_R["softplus"]
    SP_R --> SP_R_OUT["softplus(raw_r)"]
    SP_R_OUT --> ADD_R["+"]
    EPS_R["ε"] --> ADD_R
    ADD_R --> r["r = softplus(raw_r) + ε"]

    k --> GAMMA["Gamma(k, r)"]
    r --> GAMMA

    style k fill:#adf,stroke:#48f,stroke-width:2px
    style r fill:#fda,stroke:#f84,stroke-width:2px
```

**Jacobian:**

$$J_A = \begin{pmatrix} \sigma'(\text{raw}_k) & 0 \\ 0 & \sigma'(\text{raw}_r) \end{pmatrix}$$

`∂L/∂k` → only `raw_k` → only `x_k`.  `∂L/∂r` → only `raw_r` → only `x_r`.

---

## qi Surrogate Internals: Repam B

`raw_μ` feeds into `k` via `k = μ/F`.
`raw_F` feeds into **both** `k` and `r`.
$J_g$ is upper-triangular — gradient competition in `raw_F`.

```mermaid
flowchart TD
    x_k["x_k (B, 64)"] --> LIN_MU["linear_mu: x_k → raw_μ"]
    x_r["x_r (B, 64)"] --> LIN_F["linear_fano: x_r → raw_F"]

    LIN_MU --> raw_mu["raw_μ (B, 1)"]
    LIN_F --> raw_F["raw_F (B, 1)"]

    raw_mu --> SP_MU["softplus"]
    SP_MU --> SP_MU_OUT["softplus(raw_μ)"]
    SP_MU_OUT --> ADD_MU["+"]
    EPS1["ε"] --> ADD_MU
    ADD_MU --> mu["μ = softplus(raw_μ) + ε"]

    raw_F --> SP_F["softplus"]
    SP_F --> SP_F_OUT["softplus(raw_F)"]
    SP_F_OUT --> ADD_F["+"]
    EPS2["ε"] --> ADD_F
    ADD_F --> fano["F = softplus(raw_F) + ε"]

    fano --> INV["1 / F"]
    INV --> r["r = 1/F"]

    mu --> MUL["×"]
    r --> MUL
    MUL --> MU_R["μ · r = μ/F"]
    MU_R --> CLAMP["clamp(min=k_min)"]
    CLAMP --> k["k = clamp(μ/F, k_min)"]

    k --> GAMMA["Gamma(k, r)"]
    r --> GAMMA

    %% Highlight the coupling: raw_F flows to both k and r
    linkStyle 7,8,9 stroke:#f00,stroke-width:2px
    linkStyle 10,11 stroke:#f00,stroke-width:2px
    linkStyle 12,13 stroke:#f00,stroke-width:2px

    style k fill:#adf,stroke:#48f,stroke-width:2px
    style r fill:#fda,stroke:#f84,stroke-width:2px
    style fano fill:#fcc,stroke:#f00,stroke-width:2px
```

**Jacobian:**

$$J_B = \begin{pmatrix} \sigma'/F & -\mu\sigma'/F^2 \\ 0 & -\sigma'/F^2 \end{pmatrix}$$

`∂L/∂k` → flows to **both** `raw_μ` AND `raw_F`.
`∂L/∂r` → flows to `raw_F`.
**`raw_F` receives competing signals from ∂L/∂k and ∂L/∂r.**

---

## qi Surrogate Internals: Repam C

`raw_φ` feeds into **both** `k` and `r`.
`raw_μ` feeds only into `r`.
$J_g$ has off-diagonal coupling.

```mermaid
flowchart TD
    x_k["x_k (B, 64)"] --> LIN_MU["linear_mu: x_k → raw_μ"]
    x_r["x_r (B, 64)"] --> LIN_PHI["linear_phi: x_r → raw_φ"]

    LIN_MU --> raw_mu["raw_μ (B, 1)"]
    LIN_PHI --> raw_phi["raw_φ (B, 1)"]

    raw_mu --> SP_MU["softplus"]
    SP_MU --> mu["μ = softplus(raw_μ) + ε"]

    raw_phi --> SP_PHI["softplus"]
    SP_PHI --> phi["φ = softplus(raw_φ) + ε"]

    phi --> INV_PHI["1 / φ"]
    INV_PHI --> CLAMP["clamp(min=k_min)"]
    CLAMP --> k["k = clamp(1/φ, k_min)"]

    phi --> MUL_PHI_MU["φ × μ"]
    mu --> MUL_PHI_MU
    MUL_PHI_MU --> PHI_MU["φ · μ"]
    PHI_MU --> INV_R["1 / (φ·μ)"]
    INV_R --> r["r = 1/(φ·μ)"]

    k --> GAMMA["Gamma(k, r)"]
    r --> GAMMA

    style k fill:#adf,stroke:#48f,stroke-width:2px
    style r fill:#fda,stroke:#f84,stroke-width:2px
    style phi fill:#fcc,stroke:#f00,stroke-width:2px
```

**Jacobian:**

$$J_C = \begin{pmatrix} 0 & -\sigma'/\phi^2 \\ -\sigma'/(\phi\mu^2) & -\sigma'/(\phi^2\mu) \end{pmatrix}$$

`∂L/∂k` → flows only to `raw_φ`.
`∂L/∂r` → flows to **both** `raw_μ` AND `raw_φ`.
**`raw_φ` receives competing signals from ∂L/∂k and ∂L/∂r.**

---

## qi Surrogate Internals: Repam D

`raw_k` and `raw_F` are independent.
$J_g$ is diagonal — no coupling, cleanest gradient path.

```mermaid
flowchart TD
    x_k["x_k (B, 64)"] --> LIN_K["linear_k: x_k → raw_k"]
    x_r["x_r (B, 64)"] --> LIN_F["linear_fano: x_r → raw_F"]

    LIN_K --> raw_k["raw_k (B, 1)"]
    LIN_F --> raw_F["raw_F (B, 1)"]

    raw_k --> SP_K["softplus"]
    SP_K --> SP_K_OUT["softplus(raw_k)"]
    SP_K_OUT --> ADD_K["+"]
    K_MIN["k_min"] --> ADD_K
    ADD_K --> k["k = softplus(raw_k) + k_min"]

    raw_F --> SP_F["softplus"]
    SP_F --> SP_F_OUT["softplus(raw_F)"]
    SP_F_OUT --> ADD_F["+"]
    EPS["ε"] --> ADD_F
    ADD_F --> fano["F = softplus(raw_F) + ε"]

    fano --> INV["1 / F"]
    INV --> r["r = 1/F"]

    k --> GAMMA["Gamma(k, r)"]
    r --> GAMMA

    style k fill:#adf,stroke:#48f,stroke-width:2px
    style r fill:#fda,stroke:#f84,stroke-width:2px
```

**Jacobian:**

$$J_D = \begin{pmatrix} \sigma'(\text{raw}_k) & 0 \\ 0 & -\sigma'/F^2 \end{pmatrix}$$

`∂L/∂k` → only `raw_k` → only `x_k`.  `∂L/∂r` → only `raw_F` → only `x_r`.

---

## Side-by-Side Gradient Flow Comparison

The key difference is **where the gradient paths merge**:

```mermaid
flowchart LR
    subgraph A ["Repam A (diagonal)"]
        direction TB
        LA["∂L/∂k"] --> rk_A["raw_k"]
        LrA["∂L/∂r"] --> rr_A["raw_r"]
    end

    subgraph B ["Repam B (coupled)"]
        direction TB
        LB["∂L/∂k"] --> rmu_B["raw_μ"]
        LB --> rf_B["raw_F"]
        LrB["∂L/∂r"] --> rf_B
    end

    subgraph C ["Repam C (coupled)"]
        direction TB
        LC["∂L/∂k"] --> rphi_C["raw_φ"]
        LrC["∂L/∂r"] --> rmu_C["raw_μ"]
        LrC --> rphi_C
    end

    subgraph D ["Repam D (diagonal)"]
        direction TB
        LD["∂L/∂k"] --> rk_D["raw_k"]
        LrD["∂L/∂r"] --> rf_D["raw_F"]
    end

    style rf_B fill:#fcc,stroke:#f00,stroke-width:2px
    style rphi_C fill:#fcc,stroke:#f00,stroke-width:2px
```

**Red nodes** receive competing gradient signals from both `∂L/∂k` and `∂L/∂r`.
This is the structural cause of gradient competition in Repams B and C.

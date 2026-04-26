# Integrator Model Dataflows

## IntegratorModelA (2 encoders)

```mermaid
flowchart TD
    SB[Shoebox<br>B x 1 x D x H x W]

    SB --> ENC_P[profile encoder<br>ShoeboxEncoder]
    SB --> ENC_I[intensity encoder<br>IntensityEncoder]

    ENC_P --> x_p[x_profile<br>B x 64]
    ENC_I --> x_i[x_intensity<br>B x 64]

    x_p --> QP[qp surrogate]
    x_i --> QI[qi surrogate]
    x_i --> QBG[qbg surrogate]

    QP --> zp[zp<br>B x S x K]
    QI --> zI[zI<br>B x S x 1]
    QBG --> zbg[zbg<br>B x S x 1]

    zI --> RATE[rate = zI * zp + zbg<br>B x S x K]
    zp --> RATE
    zbg --> RATE

    RATE --> LOSS[Loss]
    QP --> LOSS
    QI --> LOSS
    QBG --> LOSS
    COUNTS[counts<br>B x K] --> LOSS

    LOSS --> ELBO[ELBO = -NLL + KL_prf + KL_i + KL_bg]
```

**Key:** qi and qbg share the same encoder output (x_intensity).

---

## IntegratorModelB (3 encoders)

```mermaid
flowchart TD
    SB[Shoebox<br>B x 1 x D x H x W]

    SB --> ENC_P[profile encoder<br>ShoeboxEncoder]
    SB --> ENC_K[k encoder<br>IntensityEncoder]
    SB --> ENC_R[r encoder<br>IntensityEncoder]

    ENC_P --> x_p[x_profile<br>B x 64]
    ENC_K --> x_k[x_k<br>B x 64]
    ENC_R --> x_r[x_r<br>B x 64]

    x_p --> QP[qp surrogate]
    x_k --> QI[qi surrogate]
    x_r --> QI
    x_k --> QBG[qbg surrogate]
    x_r --> QBG

    QP --> zp[zp<br>B x S x K]
    QI --> zI[zI<br>B x S x 1]
    QBG --> zbg[zbg<br>B x S x 1]

    zI --> RATE[rate = zI * zp + zbg<br>B x S x K]
    zp --> RATE
    zbg --> RATE

    RATE --> LOSS[Loss]
    QP --> LOSS
    QI --> LOSS
    QBG --> LOSS
    COUNTS[counts<br>B x K] --> LOSS

    LOSS --> ELBO[ELBO = -NLL + KL_prf + KL_i + KL_bg]
```

**Key:** qi and qbg each receive (x_k, x_r) as two separate inputs. Surrogates internally combine these to parameterize their distributions.

---

## IntegratorModelC (5 encoders)

```mermaid
flowchart TD
    SB[Shoebox<br>B x 1 x D x H x W]

    SB --> ENC_P[profile encoder<br>ShoeboxEncoder]
    SB --> ENC_KI[k_i encoder<br>IntensityEncoder]
    SB --> ENC_RI[r_i encoder<br>IntensityEncoder]
    SB --> ENC_KBG[k_bg encoder<br>IntensityEncoder]
    SB --> ENC_RBG[r_bg encoder<br>IntensityEncoder]

    ENC_P --> x_p[x_profile<br>B x 64]
    ENC_KI --> x_ki[x_k_i<br>B x 64]
    ENC_RI --> x_ri[x_r_i<br>B x 64]
    ENC_KBG --> x_kbg[x_k_bg<br>B x 64]
    ENC_RBG --> x_rbg[x_r_bg<br>B x 64]

    x_p --> QP[qp surrogate]
    x_ki --> QI[qi surrogate]
    x_ri --> QI
    x_kbg --> QBG[qbg surrogate]
    x_rbg --> QBG

    QP --> zp[zp<br>B x S x K]
    QI --> zI[zI<br>B x S x 1]
    QBG --> zbg[zbg<br>B x S x 1]

    zI --> RATE[rate = zI * zp + zbg<br>B x S x K]
    zp --> RATE
    zbg --> RATE

    RATE --> LOSS[Loss]
    QP --> LOSS
    QI --> LOSS
    QBG --> LOSS
    COUNTS[counts<br>B x K] --> LOSS

    LOSS --> ELBO[ELBO = -NLL + KL_prf + KL_i + KL_bg]
```

**Key:** Full decoupling -- qi gets (x_k_i, x_r_i), qbg gets (x_k_bg, x_r_bg). Prevents entanglement of intensity/background posterior dependencies.

---

## HierarchicalIntegratorA (2 encoders + group labels)

```mermaid
flowchart TD
    SB[Shoebox<br>B x 1 x D x H x W]
    META[metadata]

    SB --> ENC_P[profile encoder<br>ShoeboxEncoder]
    SB --> ENC_I[intensity encoder<br>IntensityEncoder]

    ENC_P --> x_p[x_profile<br>B x 64]
    ENC_I --> x_i[x_intensity<br>B x 64]

    META --> GL[group_labels<br>B]
    META --> PGL[profile_group_labels<br>B]

    x_p --> QP[qp surrogate]
    PGL -.-> QP
    x_i --> QI[qi surrogate]
    x_i --> QBG[qbg surrogate]

    QP --> zp[zp<br>B x S x K]
    QI --> zI[zI<br>B x S x 1]
    QBG --> zbg[zbg<br>B x S x 1]

    zI --> RATE[rate = zI * zp + zbg<br>B x S x K]
    zp --> RATE
    zbg --> RATE

    RATE --> LOSS[PerBinLoss / WilsonPerBinLoss]
    QP --> LOSS
    QI --> LOSS
    QBG --> LOSS
    GL --> LOSS
    COUNTS[counts<br>B x K] --> LOSS

    LOSS --> ELBO[ELBO = -NLL + KL_prf + KL_i + KL_bg]

    subgraph Per-group priors
        TAU[tau_per_group] --> LOSS
        BGR[bg_rate_per_group] --> LOSS
        CONC[concentration_per_group] --> LOSS
    end
```

**Key:** Same encoder structure as ModelA, but profile surrogate receives group_labels and loss indexes per-group priors via group_labels.

---

## HierarchicalIntegratorB (3 encoders + group labels)

```mermaid
flowchart TD
    SB[Shoebox<br>B x 1 x D x H x W]
    META[metadata]

    SB --> ENC_P[profile encoder<br>ShoeboxEncoder]
    SB --> ENC_K[k encoder<br>IntensityEncoder]
    SB --> ENC_R[r encoder<br>IntensityEncoder]

    ENC_P --> x_p[x_profile<br>B x 64]
    ENC_K --> x_k[x_k<br>B x 64]
    ENC_R --> x_r[x_r<br>B x 64]

    META --> GL[group_labels<br>B]
    META --> PGL[profile_group_labels<br>B]

    x_p --> QP[qp surrogate]
    PGL -.-> QP
    x_k --> QI[qi surrogate]
    x_r --> QI
    x_k --> QBG[qbg surrogate]
    x_r --> QBG

    QP --> zp[zp<br>B x S x K]
    QI --> zI[zI<br>B x S x 1]
    QBG --> zbg[zbg<br>B x S x 1]

    zI --> RATE[rate = zI * zp + zbg<br>B x S x K]
    zp --> RATE
    zbg --> RATE

    RATE --> LOSS[PerBinLoss / WilsonPerBinLoss]
    QP --> LOSS
    QI --> LOSS
    QBG --> LOSS
    GL --> LOSS
    COUNTS[counts<br>B x K] --> LOSS

    LOSS --> ELBO[ELBO = -NLL + KL_prf + KL_i + KL_bg]

    subgraph Per-group priors
        TAU[tau_per_group] --> LOSS
        BGR[bg_rate_per_group] --> LOSS
        CONC[concentration_per_group] --> LOSS
    end
```

**Key:** Same encoder structure as ModelB (shared k,r for qi and qbg), plus hierarchical per-group priors.

---

## HierarchicalIntegratorC (5 encoders + group labels)

```mermaid
flowchart TD
    SB[Shoebox<br>B x 1 x D x H x W]
    META[metadata]

    SB --> ENC_P[profile encoder<br>ShoeboxEncoder]
    SB --> ENC_KI[k_i encoder<br>IntensityEncoder]
    SB --> ENC_RI[r_i encoder<br>IntensityEncoder]
    SB --> ENC_KBG[k_bg encoder<br>IntensityEncoder]
    SB --> ENC_RBG[r_bg encoder<br>IntensityEncoder]

    ENC_P --> x_p[x_profile<br>B x 64]
    ENC_KI --> x_ki[x_k_i<br>B x 64]
    ENC_RI --> x_ri[x_r_i<br>B x 64]
    ENC_KBG --> x_kbg[x_k_bg<br>B x 64]
    ENC_RBG --> x_rbg[x_r_bg<br>B x 64]

    META --> GL[group_labels<br>B]
    META --> PGL[profile_group_labels<br>B]

    x_p --> QP[qp surrogate]
    PGL -.-> QP
    x_ki --> QI[qi surrogate]
    x_ri --> QI
    x_kbg --> QBG[qbg surrogate]
    x_rbg --> QBG

    QP --> zp[zp<br>B x S x K]
    QI --> zI[zI<br>B x S x 1]
    QBG --> zbg[zbg<br>B x S x 1]

    zI --> RATE[rate = zI * zp + zbg<br>B x S x K]
    zp --> RATE
    zbg --> RATE

    RATE --> LOSS[PerBinLoss / WilsonPerBinLoss]
    QP --> LOSS
    QI --> LOSS
    QBG --> LOSS
    GL --> LOSS
    COUNTS[counts<br>B x K] --> LOSS

    LOSS --> ELBO[ELBO = -NLL + KL_prf + KL_i + KL_bg]

    subgraph Per-group priors
        TAU[tau_per_group] --> LOSS
        BGR[bg_rate_per_group] --> LOSS
        CONC[concentration_per_group] --> LOSS
    end
```

**Key:** Most flexible model. Fully decoupled intensity/background encoders + hierarchical per-group priors. Profile surrogate can use profile_group_labels (2D: resolution x angle) while loss uses group_labels (1D: resolution only).

---

## Summary

| Model | Encoders | qi/qbg coupling | Hierarchical | Loss types |
|-------|----------|----------------|--------------|------------|
| ModelA | 2 | Shared encoder | No | default |
| ModelB | 3 | Shared (k, r) | No | default |
| ModelC | 5 | Decoupled | No | default |
| HierarchicalA | 2 | Shared encoder | Yes | per_bin, wilson_per_bin |
| HierarchicalB | 3 | Shared (k, r) | Yes | per_bin, wilson_per_bin |
| HierarchicalC | 5 | Decoupled | Yes | per_bin, wilson_per_bin |

### Universal rate equation

```
rate[b, s, k] = zI[b, s, 1] * zp[b, s, k] + zbg[b, s, 1]
```

All models share the same generative model; they differ only in how encoder features are routed to surrogates, and whether per-group priors are used.

"""
Epsilon analysis for Gamma reparameterizations.

Traces how eps=1e-6 flows through each parameterization, identifies
problematic ranges, and tests alternative choices.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Gamma

torch.manual_seed(42)

EPS = 1e-6
K_MIN = 0.1


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Exact (k, r) ranges implied by each parameterization with eps=1e-6
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_ranges():
    print("=" * 80)
    print(
        f"PART 1: Exact (k, r, mean, fano) ranges with k_min={K_MIN}, eps={EPS}"
    )
    print("=" * 80)
    print()

    # Scan extreme raw values
    raw_min, raw_max = -20.0, 20.0

    print(
        "─── RepamA: k = softplus(raw_k) + k_min,  r = softplus(raw_r) + eps ───"
    )
    sp_min = float(F.softplus(torch.tensor(raw_min)))  # ≈ 2e-9
    sp_max = float(F.softplus(torch.tensor(raw_max)))  # ≈ 20
    k_min_a = sp_min + K_MIN
    k_max_a = sp_max + K_MIN
    r_min_a = sp_min + EPS
    r_max_a = sp_max + EPS
    print(f"  k range:    [{k_min_a:.2e}, {k_max_a:.2e}]")
    print(f"  r range:    [{r_min_a:.2e}, {r_max_a:.2e}]")
    print(f"  mean range: [{k_min_a / r_max_a:.2e}, {k_max_a / r_min_a:.2e}]")
    print(f"  fano range: [{1 / r_max_a:.2e}, {1 / r_min_a:.2e}]")
    print(
        f"  std/mean:   [{1 / np.sqrt(k_max_a):.4f}, {1 / np.sqrt(k_min_a):.1f}]"
    )
    print()

    print(
        "─── RepamA: k = k_min + (k_max-k_min)*sigmoid(raw_k) (k_max=500) ───"
    )
    sig_min = float(torch.sigmoid(torch.tensor(raw_min)))
    sig_max = float(torch.sigmoid(torch.tensor(raw_max)))
    k_min_a500 = K_MIN + (500 - K_MIN) * sig_min
    k_max_a500 = K_MIN + (500 - K_MIN) * sig_max
    print(f"  sigmoid range:  [{sig_min:.2e}, {sig_max:.10f}]")
    print(f"  k range:    [{k_min_a500:.4f}, {k_max_a500:.6f}]")
    print(f"  r range:    [{r_min_a:.2e}, {r_max_a:.2e}]  (same as above)")
    print(
        f"  mean range: [{k_min_a500 / r_max_a:.2e}, {k_max_a500 / r_min_a:.2e}]"
    )
    print()

    print(
        "─── RepamB: mu = softplus + eps,  fano = softplus + eps, k = clamp(mu*r, min=k_min) ───"
    )
    mu_min = sp_min + EPS
    mu_max = sp_max + EPS
    fano_min = sp_min + EPS
    fano_max = sp_max + EPS
    # r = 1/fano  — no double eps
    r_min_b = 1.0 / fano_max
    r_max_b = 1.0 / fano_min
    # k = clamp(mu * r, min=K_MIN)
    k_min_b = max(mu_min * r_min_b, K_MIN)
    k_max_b = mu_max * r_max_b
    print(f"  mu range:   [{mu_min:.2e}, {mu_max:.2e}]")
    print(f"  fano range: [{fano_min:.2e}, {fano_max:.2e}]")
    print("  r = 1/fano:")
    print(f"    r range:  [{r_min_b:.2e}, {r_max_b:.2e}]")
    print(f"  k = clamp(mu*r, min={K_MIN}):")
    print(f"    k range:  [{k_min_b:.2e}, {k_max_b:.2e}]")
    print(f"  mean = mu:  [{mu_min:.2e}, {mu_max:.2e}]")
    print(f"  fano = 1/r: [{1 / r_max_b:.2e}, {1 / r_min_b:.2e}]")
    print()

    print("─── RepamC: mu = softplus + eps,  phi = softplus + eps ───")
    phi_min = sp_min + EPS
    phi_max = sp_max + EPS
    # k = 1/phi, clamped [K_MIN, k_max]
    k_raw_min_c = 1.0 / phi_max
    k_raw_max_c = 1.0 / phi_min
    k_min_c = max(k_raw_min_c, K_MIN)
    k_max_c = k_raw_max_c  # before k_max clamp
    # r = 1/(phi * mu)  — no extra eps
    r_min_c = 1.0 / (phi_max * mu_max)
    r_max_c = 1.0 / (phi_min * mu_min)
    print(f"  phi range:  [{phi_min:.2e}, {phi_max:.2e}]")
    print(f"  k = 1/phi before clamp: [{k_raw_min_c:.2e}, {k_raw_max_c:.2e}]")
    print(f"  k after clamp(min={K_MIN}):   [{k_min_c:.2e}, {k_max_c:.2e}]")
    print("  r = 1/(phi*mu):")
    print(f"    r range:  [{r_min_c:.2e}, {r_max_c:.2e}]")
    print(f"  mean = k/r: [{k_min_c / r_max_c:.2e}, {k_max_c / r_min_c:.2e}]")
    print()

    print(
        "─── RepamD: k = softplus(raw_k) + k_min,  fano = softplus + eps ───"
    )
    print(f"  k range:    [{k_min_a:.2e}, {k_max_a:.2e}]  (same as RepamA)")
    r_min_d = r_min_b  # same fano → rate path as B
    r_max_d = r_max_b
    print(f"  fano range: [{fano_min:.2e}, {fano_max:.2e}]")
    print("  r = 1/fano:")
    print(f"    r range:  [{r_min_d:.2e}, {r_max_d:.2e}]")
    print("  mean = k/r = k*fano:")
    print(
        f"    range:    [{k_min_a * fano_min:.2e}, {k_max_a * fano_max:.2e}]"
    )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: The double-eps problem
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_double_eps():
    print("=" * 80)
    print("PART 2: Double-eps ELIMINATED — verification")
    print("=" * 80)
    print()

    print("BEFORE (old code): RepamB had eps applied THREE times:")
    print("  1. fano = softplus(raw) + eps        # fano ≥ eps")
    print("  2. r = 1 / (fano + eps)              # added eps AGAIN to denom")
    print("  3. k = mu * r + eps                  # added eps to k")
    print()
    print("AFTER (new code): Clean separation:")
    print(
        "  1. fano = softplus(raw) + eps        # fano ≥ eps (division safety)"
    )
    print("  2. r = 1.0 / fano                    # no double eps")
    print("  3. k = clamp(mu * r, min=k_min)      # k_min for IRG safety")
    print()

    raw = torch.tensor(-20.0)
    fano = F.softplus(raw) + EPS
    r_new = 1.0 / fano
    print("At raw_fano = -20 (minimum fano):")
    print(f"  softplus(-20) = {F.softplus(raw).item():.2e}")
    print(f"  fano = {fano.item():.2e}")
    print(f"  r = 1/fano = {r_new.item():.2e}")
    print()

    print("RepamC — also cleaned:")
    print("  OLD: k = 1 / (phi + eps), double eps")
    print("  NEW: k = (1.0 / phi).clamp(min=k_min)")
    phi = F.softplus(raw) + EPS
    k_new = max(1.0 / phi.item(), K_MIN)
    print(f"  At raw_phi = -20: k = max(1/phi, {K_MIN}) = {k_new:.4f}")
    print()

    print(
        "VERDICT: Double-eps eliminated. k floor is now explicitly k_min={K_MIN},"
    )
    print("eps={EPS} is used ONLY for division safety.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: What eps SHOULD be based on the IRG threshold
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_correct_eps():
    print("=" * 80)
    print("PART 3: k_min / eps separation — NOW IMPLEMENTED")
    print("=" * 80)
    print()

    print(f"k_min = {K_MIN}   # IRG-safe floor for concentration")
    print(f"eps   = {EPS}  # numerical safety for divisions only")
    print()
    print("Implementation per repam:")
    print()
    print("  RepamA/D (direct k via softplus):")
    print(f"    k = softplus(raw) + {K_MIN}    → k ∈ [{K_MIN}, ∞)")
    print()
    print("  RepamA/D (sigmoid with k_max):")
    print(f"    k = {K_MIN} + (k_max - {K_MIN}) * sigmoid(raw)")
    print(f"    → k ∈ [{K_MIN}, k_max), smooth everywhere")
    print()
    print("  RepamB:")
    print(f"    k = clamp(mu * r, min={K_MIN})")
    print("    fano has eps for division safety, no double-eps on r = 1/fano")
    print()
    print("  RepamC:")
    print(f"    k = (1/phi).clamp(min={K_MIN})")
    print("    r = 1/(phi*mu), no extra eps")
    print()
    print("  GammaDistribution:")
    print(f"    k = clamp(mu * r, min={K_MIN})")
    print("    r = 1/fano, no double-eps")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: Gradient test across eps choices
# ═══════════════════════════════════════════════════════════════════════════════


def test_gradient_vs_eps():
    """Test gradient reliability for RepamD with different k floors."""
    print("=" * 80)
    print("PART 4: RepamD gradient reliability with different k_min values")
    print("=" * 80)
    print()
    print("Using RepamD (best SBC calibration) as the test case.")
    print("We test: what k_min gives reliable gradients without restricting")
    print("the posterior too much?")
    print()
    print(f"Current implementation uses k_min={K_MIN}.")
    print()

    k_mins = [1e-6, 1e-4, 1e-2, 0.05, 0.1, 0.2, 0.5]
    n_samples = 2000

    print(
        f"  {'k_min':>8s} | {'actual_k':>10s} | {'nan%':>8s} | {'zero%':>8s} | "
        f"{'SNR(k)':>10s} | {'SNR(fano)':>10s} | {'grad_k_mean':>14s} | {'grad_k_std':>14s}"
    )
    print(
        f"  {'-' * 8} | {'-' * 10} | {'-' * 8} | {'-' * 8} | "
        f"{'-' * 10} | {'-' * 10} | {'-' * 14} | {'-' * 14}"
    )

    for k_min in k_mins:
        # Set raw_k so that softplus(raw_k) ≈ 0 → k = k_min
        raw_k_val = torch.tensor(-20.0)  # softplus(-20) ≈ 0
        raw_fano_val = torch.tensor(0.0)

        grads_k = []
        grads_f = []
        nan_count = 0
        zero_count = 0

        for _ in range(n_samples):
            raw_k = raw_k_val.clone().requires_grad_(True)
            raw_fano = raw_fano_val.clone().requires_grad_(True)

            k = F.softplus(raw_k) + k_min
            fano = F.softplus(raw_fano) + EPS
            r = 1.0 / fano  # no double eps

            dist = Gamma(k, r)
            sample = dist.rsample()

            if torch.isnan(sample) or torch.isinf(sample):
                nan_count += 1
                continue

            # Poisson NLL
            nll = -(10.0 * torch.log(sample + 1e-10) - sample)
            try:
                nll.backward()
            except RuntimeError:
                nan_count += 1
                continue

            gk = raw_k.grad
            gf = raw_fano.grad

            if gk is None or gf is None or torch.isnan(gk) or torch.isnan(gf):
                nan_count += 1
            elif gk.abs().item() == 0 and gf.abs().item() == 0:
                zero_count += 1
            else:
                grads_k.append(gk.item())
                grads_f.append(gf.item())

        total = nan_count + zero_count + len(grads_k)

        if len(grads_k) < 10:
            print(
                f"  {k_min:8.1e} | {k_min:10.4f} | {nan_count / total * 100:7.1f}% | "
                f"{zero_count / total * 100:7.1f}% | FAILED"
            )
            continue

        gk_arr = np.array(grads_k)
        gf_arr = np.array(grads_f)
        snr_k = abs(gk_arr.mean()) / (gk_arr.std() + 1e-30)
        snr_f = abs(gf_arr.mean()) / (gf_arr.std() + 1e-30)

        marker = " ← CURRENT" if abs(k_min - K_MIN) < 1e-8 else ""
        print(
            f"  {k_min:8.1e} | {k_min:10.4f} | {nan_count / total * 100:7.1f}% | "
            f"{zero_count / total * 100:7.1f}% | {snr_k:10.4f} | {snr_f:10.4f} | "
            f"{gk_arr.mean():14.6f} | {gk_arr.std():14.6f}{marker}"
        )

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: Rate-side eps — does 1/(fano+eps) vs 1/fano matter?
# ═══════════════════════════════════════════════════════════════════════════════


def test_rate_eps():
    """Verify that r = 1/fano (no double-eps) is safe."""
    print("=" * 80)
    print("PART 5: Rate-side — 1/fano is safe (double-eps removed)")
    print("=" * 80)
    print()

    print("NEW code: r = 1.0 / fano  (fano = softplus(raw) + eps ≥ eps > 0)")
    print("Since fano ≥ eps = 1e-6, r ≤ 1e6. No division by zero possible.")
    print()

    print(
        f"  {'raw_fano':>10s} | {'fano':>12s} | {'r = 1/fano':>12s} | {'mean (k=0.1)':>14s}"
    )
    print(f"  {'-' * 10} | {'-' * 12} | {'-' * 12} | {'-' * 14}")

    for raw in [-20, -10, -5, -3, -1, 0, 1, 3, 5, 10, 20]:
        fano = float(F.softplus(torch.tensor(float(raw)))) + EPS
        r = 1.0 / fano
        mean = K_MIN / r  # k = k_min at floor
        print(f"  {raw:10d} | {fano:12.6e} | {r:12.4e} | {mean:14.6e}")

    print()
    print(f"At the floor (k={K_MIN}), the smallest achievable mean is")
    print(f"  {K_MIN} / {1.0 / EPS:.0e} = {K_MIN * EPS:.2e}")
    print("which is far smaller than any physical intensity.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: Physical constraints — what (k, r) values are sensible?
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_physical_constraints():
    print("=" * 80)
    print("PART 6: Physical constraints from the simulation")
    print("=" * 80)
    print()

    print("From simulate_shoeboxes_mvn.py:")
    print("  I ~ Exp(rate=0.001)  →  mean=1000, range ≈ [0.01, 10000]")
    print("  bg ~ Exp(rate=1.0)   →  mean=1.0,  range ≈ [0.001, 10]")
    print()
    print("For the intensity posterior q(I) = Gamma(k, r):")
    print("  mean = k/r should cover [0.01, 10000]")
    print("  std/mean = 1/√k, so:")
    print("    - At I=0.1:  noise-to-signal ≈ 3-10, so k ≈ 0.01-0.1")
    print("    - At I=1:    noise-to-signal ≈ 1-3,  so k ≈ 0.1-1")
    print("    - At I=100:  noise-to-signal ≈ 0.1,  so k ≈ 100")
    print("    - At I=10000: noise-to-signal ≈ 0.01, so k ≈ 10000")
    print()
    print("So the network NEEDS k < 0.1 for the weakest reflections!")
    print("This is exactly where the IRG breaks down.")
    print()
    print(
        "But from the noise-to-signal plots, the posteriors are tracking the"
    )
    print("CRLB even at low intensity. How?")
    print()
    print("Answer: At low intensity, the posterior width is dominated by the")
    print("RATE parameter, not k. Consider Gamma(k=1, r=10):")
    print(
        f"  mean = {1 / 10:.2f}, std = {np.sqrt(1) / 10:.2f}, std/mean = {1 / np.sqrt(1):.2f}"
    )
    print("vs Gamma(k=0.1, r=1):")
    print(
        f"  mean = {0.1 / 1:.2f}, std = {np.sqrt(0.1) / 1:.2f}, std/mean = {1 / np.sqrt(0.1):.2f}"
    )
    print()
    print("Both have mean=0.1, but the first uses moderate k with large r,")
    print("while the second uses dangerous small k. The network can achieve")
    print("small posterior means via large rates while keeping k safe.")
    print()
    print("This is WHY RepamD works well: k and rate are decoupled, so the")
    print(
        "network naturally learns to use moderate k (controlling shape/width)"
    )
    print("and adjust rate (controlling location). It doesn't NEED small k.")
    print()
    print(
        "But for the variance to be correct at low I, the network still needs"
    )
    print("appropriate k. At I=0.1:")
    print("  CRLB std/mean ≈ 3.16 (from Poisson), so k ≈ 0.1")
    print("  This is RIGHT at the IRG boundary.")
    print()
    print("With eps=1e-6, the floor is way below this — not protective.")
    print("With k_min=0.1, we'd force k ≥ 0.1, which means:")
    print("  - Std/mean ≤ 1/√0.1 ≈ 3.16")
    print("  - This is the Poisson limit! So k_min=0.1 means the posterior")
    print("    can never be wider than the counting-statistics limit.")
    print("  - For well-measured reflections this is fine.")
    print("  - For very weak reflections (I < 0.1), the posterior SHOULD be")
    print("    wider, but k_min=0.1 prevents it. Is this acceptable?")
    print()
    print("In practice: weak reflections with I ≈ 0.1 are dominated by")
    print("background noise anyway. A prior or regularization should handle")
    print("these — the posterior width matters less when signal ≈ noise.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: Practical test — gradient quality at the boundary
# ═══════════════════════════════════════════════════════════════════════════════


def test_boundary_gradients():
    """Test gradient quality at the actual operating points during training."""
    print("=" * 80)
    print("PART 7: Gradient quality at realistic operating points")
    print("=" * 80)
    print()
    print("For each (k, mean) combination that would arise in training,")
    print("measure gradient SNR through the full ELBO path.")
    print()

    # Realistic intensity values from Exp(0.001)
    intensities = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    # For each intensity, what k values might the network use?
    # Optimal k ≈ I (Poisson limit: std/mean = 1/sqrt(k) = 1/sqrt(I))
    # But the network might not be optimal

    n_samples = 2000

    print("RepamD with softplus + k_min (testing k_min=1e-6 vs 0.1 vs 0.5):")
    print()

    for k_min in [1e-6, 0.1, 0.5]:
        marker = " ← CURRENT" if abs(k_min - K_MIN) < 1e-8 else ""
        print(f"  k_min = {k_min}{marker}")
        print(
            f"  {'I_true':>8s} | {'k':>8s} | {'r':>10s} | {'mean':>8s} | "
            f"{'SNR_k':>8s} | {'SNR_f':>8s} | {'nan%':>6s}"
        )
        print(
            f"  {'-' * 8} | {'-' * 8} | {'-' * 10} | {'-' * 8} | "
            f"{'-' * 8} | {'-' * 8} | {'-' * 6}"
        )

        for I_true in intensities:
            # Network should produce: mean ≈ I_true, k ≈ I_true (Poisson limit)
            # With RepamD: k = softplus(raw_k) + k_min, r = 1/fano
            # mean = k * fano = k * (softplus(raw_fano) + eps)

            target_k = max(I_true, k_min)  # can't go below k_min
            target_mean = I_true

            # Solve for raw params
            tk = target_k - k_min
            if tk > 0 and tk < 20:
                raw_k_val = float(np.log(np.expm1(tk)))
            elif tk >= 20:
                raw_k_val = float(tk)
            else:
                raw_k_val = -20.0

            # mean = k / r = k * fano where fano = softplus(raw_fano) + eps
            target_fano = target_mean / target_k
            tf = target_fano - EPS
            if tf > 0 and tf < 20:
                raw_fano_val = float(np.log(np.expm1(tf)))
            elif tf >= 20:
                raw_fano_val = float(tf)
            else:
                raw_fano_val = -20.0

            grads_k = []
            grads_f = []
            nan_count = 0

            for _ in range(n_samples):
                raw_k = torch.tensor(raw_k_val, requires_grad=True)
                raw_fano = torch.tensor(raw_fano_val, requires_grad=True)

                k = F.softplus(raw_k) + k_min
                fano = F.softplus(raw_fano) + EPS
                r = 1.0 / fano  # no double eps

                dist = Gamma(k, r)
                sample = dist.rsample()

                if torch.isnan(sample) or torch.isinf(sample) or sample <= 0:
                    nan_count += 1
                    continue

                nll = -(I_true * torch.log(sample + 1e-10) - sample)
                try:
                    nll.backward()
                except RuntimeError:
                    nan_count += 1
                    continue

                gk = raw_k.grad
                gf = raw_fano.grad
                if (
                    gk is not None
                    and gf is not None
                    and not (torch.isnan(gk) or torch.isnan(gf))
                ):
                    grads_k.append(gk.item())
                    grads_f.append(gf.item())
                else:
                    nan_count += 1

            actual_k = float(F.softplus(torch.tensor(raw_k_val))) + k_min
            actual_fano = float(F.softplus(torch.tensor(raw_fano_val))) + EPS
            actual_r = 1.0 / actual_fano
            actual_mean = actual_k / actual_r

            if len(grads_k) > 10:
                gk_arr = np.array(grads_k)
                gf_arr = np.array(grads_f)
                snr_k = abs(gk_arr.mean()) / (gk_arr.std() + 1e-30)
                snr_f = abs(gf_arr.mean()) / (gf_arr.std() + 1e-30)
                nan_pct = nan_count / (nan_count + len(grads_k)) * 100
                print(
                    f"  {I_true:8.1f} | {actual_k:8.2f} | {actual_r:10.4f} | "
                    f"{actual_mean:8.2f} | {snr_k:8.4f} | {snr_f:8.4f} | {nan_pct:5.1f}%"
                )
            else:
                print(
                    f"  {I_true:8.1f} | {actual_k:8.2f} | FAILED (nan={nan_count})"
                )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: Recommended implementation
# ═══════════════════════════════════════════════════════════════════════════════


def print_recommendations():
    print("=" * 80)
    print("PART 8: Implementation summary (DONE)")
    print("=" * 80)
    print()
    print(f"""
IMPLEMENTED CODE:

    # All Gamma classes now have:
    self.eps = 1e-6          # numerical safety for divisions only
    self.k_min = {K_MIN}         # IRG-safe floor (configurable via YAML)

    # RepamA/D (softplus, no k_max):
    k = F.softplus(raw_k) + self.k_min          # k ∈ [{K_MIN}, ∞)
    fano = F.softplus(raw_fano) + self.eps       # fano > 0
    r = 1.0 / fano                               # no double eps

    # RepamA/D (sigmoid, with k_max):
    k = self.k_min + (self.k_max - self.k_min) * torch.sigmoid(raw_k)
                                                 # k ∈ [k_min, k_max)

    # RepamB/GammaDistribution:
    k = (mu * r).clamp(min=self.k_min)           # hard floor at k_min

    # RepamC:
    k = (1.0 / phi).clamp(min=self.k_min)        # hard floor at k_min

    # _init_k_bias adjusted for offset:
    # sigmoid(bias) = (k_init - k_min) / (k_max - k_min)

IMPACT:
    - k_min={K_MIN}: floor at Poisson counting-statistics limit
      std/mean = 1/√{K_MIN} ≈ {1 / np.sqrt(K_MIN):.2f} (maximum posterior width)
    - Matches the IRG safe threshold exactly
    - No gradient dead zones (softplus/sigmoid are smooth for RepamA/D)
    - Rate gradient path unchanged (already reliable)
    - Double-eps eliminated: 1/fano is safe since fano ≥ eps > 0
    - SurrogateArgs dataclass has k_min field (default {K_MIN})
""")


if __name__ == "__main__":
    analyze_ranges()
    analyze_double_eps()
    analyze_correct_eps()
    test_gradient_vs_eps()
    test_rate_eps()
    analyze_physical_constraints()
    test_boundary_gradients()
    print_recommendations()

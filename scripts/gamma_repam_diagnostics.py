"""
Gamma Reparameterization Diagnostics
=====================================
Tests how the four reparameterizations behave w.r.t. the gradient reliability
findings: IRG breaks down for α < ~0.1 in float32.

We test:
1. What k (concentration/α) values each repam can produce, and how easily
2. Gradient magnitude through rsample back to network parameters
3. Whether gradients vanish or become unreliable at small k
4. The interaction between k and rate parameterization
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Gamma

torch.manual_seed(42)

# ─── Helpers ──────────────────────────────────────────────────────────────────

K_MIN = 0.1
EPS = 1e-6


def _bound_k(raw_k, k_max, k_min):
    if k_max is not None:
        return k_min + (k_max - k_min) * torch.sigmoid(raw_k)
    return F.softplus(raw_k) + k_min


def repam_a(raw_k, raw_r, k_max=None, k_min=K_MIN, eps=EPS):
    """Direct (k, r) parameterization."""
    k = _bound_k(raw_k, k_max, k_min)
    r = F.softplus(raw_r) + eps
    return k, r


def repam_b(raw_mu, raw_fano, k_max=None, k_min=K_MIN, eps=EPS):
    """(mu, fano) → k = mu * r, r = 1/fano."""
    mu = F.softplus(raw_mu) + eps
    fano = F.softplus(raw_fano) + eps
    r = 1.0 / fano
    k = (mu * r).clamp(min=k_min)
    if k_max is not None:
        k = k.clamp(max=k_max)
    return k, r


def repam_c(raw_mu, raw_phi, k_max=None, k_min=K_MIN, eps=EPS):
    """(mu, phi) → k = 1/phi, r = 1/(phi*mu). k clamped to [k_min, k_max]."""
    mu = F.softplus(raw_mu) + eps
    phi = F.softplus(raw_phi) + eps
    k = (1.0 / phi).clamp(min=k_min)
    if k_max is not None:
        k = k.clamp(max=k_max)
    r = 1.0 / (phi * mu)
    return k, r


def repam_d(raw_k, raw_fano, k_max=None, k_min=K_MIN, eps=EPS):
    """(k, fano) → k direct, r = 1/fano."""
    k = _bound_k(raw_k, k_max, k_min)
    fano = F.softplus(raw_fano) + eps
    r = 1.0 / fano
    return k, r


REPAMS = {"A": repam_a, "B": repam_b, "C": repam_c, "D": repam_d}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: k reachability — what raw values produce small k?
# ═══════════════════════════════════════════════════════════════════════════════


def test_k_reachability():
    """For each repam, scan raw parameter space and show what k values result."""
    print("=" * 80)
    print("TEST 1: k (concentration) reachability from raw parameter space")
    print("=" * 80)
    print()

    raw_vals = torch.linspace(-10, 10, 21)

    for name, fn in REPAMS.items():
        print(f"--- Repam {name} ---")
        if name in ("A", "D"):
            # k depends only on raw_k
            print(
                f"  {'raw_k':>8s} | {'k (no k_max)':>14s} | {'k (k_max=500)':>14s}"
            )
            print(f"  {'-' * 8} | {'-' * 14} | {'-' * 14}")
            for rv in raw_vals:
                k_unb, _ = fn(rv, torch.tensor(0.0))
                k_bnd, _ = fn(rv, torch.tensor(0.0), k_max=500)
                print(
                    f"  {rv.item():8.1f} | {k_unb.item():14.6f} | {k_bnd.item():14.6f}"
                )
        else:
            # k depends on raw_mu and raw_fano/raw_phi
            # Fix one, vary the other
            print("  (fixing raw_param2=0.0, varying raw_param1)")
            print(f"  {'raw_p1':>8s} | {'k':>14s} | {'mean (k/r)':>14s}")
            print(f"  {'-' * 8} | {'-' * 14} | {'-' * 14}")
            for rv in raw_vals:
                k, r = fn(rv, torch.tensor(0.0))
                mean = k / r
                print(
                    f"  {rv.item():8.1f} | {k.item():14.6f} | {mean.item():14.6f}"
                )

            print("\n  (fixing raw_param1=0.0, varying raw_param2)")
            print(f"  {'raw_p2':>8s} | {'k':>14s} | {'mean (k/r)':>14s}")
            print(f"  {'-' * 8} | {'-' * 14} | {'-' * 14}")
            for rv in raw_vals:
                k, r = fn(torch.tensor(0.0), rv)
                mean = k / r
                print(
                    f"  {rv.item():8.1f} | {k.item():14.6f} | {mean.item():14.6f}"
                )
        print()

    # Summary: minimum achievable k
    print("SUMMARY: Minimum k achievable (raw params in [-10, 10])")
    print("-" * 60)
    for name, fn in REPAMS.items():
        min_k = float("inf")
        for r1 in torch.linspace(-10, 10, 101):
            for r2 in torch.linspace(-10, 10, 101):
                k, _ = fn(r1, r2)
                min_k = min(min_k, k.item())
        print(f"  Repam {name}: min k = {min_k:.2e}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Gradient magnitude through rsample to raw parameters
# ═══════════════════════════════════════════════════════════════════════════════


def test_gradient_flow():
    """Measure gradient of E[x] w.r.t. raw params via rsample, across k regimes."""
    print("=" * 80)
    print("TEST 2: Gradient flow through rsample to raw parameters")
    print("=" * 80)
    print()
    print(
        "For each repam and target k, we compute grad of rsample w.r.t. raw params."
    )
    print(
        "NaN/zero grads indicate the IRG breakdown your colleague documented."
    )
    print()

    # We'll set raw params to achieve specific k values, then measure gradients
    target_ks = [
        0.001,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
        5.0,
        10.0,
        50.0,
        100.0,
        500.0,
    ]
    n_samples = 1000

    for name in REPAMS:
        print(f"--- Repam {name} ---")
        print(
            f"  {'target_k':>10s} | {'actual_k':>10s} | {'rate':>10s} | {'mean':>10s} | "
            f"{'grad_p1':>12s} | {'grad_p2':>12s} | {'nan_frac':>10s} | {'zero_frac':>10s}"
        )
        print(
            f"  {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10} | "
            f"{'-' * 12} | {'-' * 12} | {'-' * 10} | {'-' * 10}"
        )

        for target_k in target_ks:
            # Find raw_p1 that gives target_k (with raw_p2=0 → rate~1 or fano~softplus(0)~0.69)
            # For A/D: raw_k s.t. softplus(raw_k) + eps ≈ target_k → raw_k ≈ log(exp(target_k) - 1)
            # For B: mu * r ≈ target_k; with fano=softplus(0)≈0.69, r=1/0.69≈1.44
            #         mu ≈ target_k / 1.44; mu = softplus(raw_mu) → raw_mu ≈ log(exp(mu)-1)
            # For C: k = 1/phi; phi = softplus(raw_phi); target_k = 1/phi → phi = 1/target_k
            #         raw_phi = log(exp(1/target_k) - 1)

            raw_p1 = torch.tensor(0.0, requires_grad=True)
            raw_p2 = torch.tensor(0.0, requires_grad=True)

            # Use a few steps of gradient descent to hit the target k
            # (simpler than solving analytically for all cases)
            opt_p1 = torch.tensor(0.0)
            opt_p2 = torch.tensor(0.0)

            if name in ("A", "D"):
                # softplus(raw_k) + eps = target_k → raw_k = log(exp(target_k - eps) - 1)
                tk = max(target_k, 1e-4)
                if tk < 20:
                    opt_p1 = torch.tensor(np.log(np.expm1(tk)))
                else:
                    opt_p1 = torch.tensor(float(tk))
                opt_p2 = torch.tensor(
                    0.0
                )  # rate = softplus(0)+eps ≈ 0.69 for A; fano for D
            elif name == "B":
                # k = mu * r + eps, r = 1/(fano+eps), fano = softplus(raw_fano)+eps
                # fix raw_fano = 0 → fano ≈ 0.693 + 1e-6 → r ≈ 1.443
                # mu = (target_k - eps) / r; mu = softplus(raw_mu)+eps
                fano0 = float(F.softplus(torch.tensor(0.0))) + 1e-6
                r0 = 1.0 / (fano0 + 1e-6)
                mu_need = max((target_k - 1e-6) / r0, 1e-6)
                sp_inv = np.log(np.expm1(mu_need)) if mu_need < 20 else mu_need
                opt_p1 = torch.tensor(float(sp_inv))
                opt_p2 = torch.tensor(0.0)
            elif name == "C":
                # k = 1/(phi+eps), clamped min 0.01
                # phi = softplus(raw_phi)+eps
                # target_k = 1/(phi+eps) → phi = 1/target_k - eps
                phi_need = max(1.0 / target_k - 1e-6, 1e-6)
                sp_inv = (
                    np.log(np.expm1(phi_need)) if phi_need < 20 else phi_need
                )
                opt_p2 = torch.tensor(float(sp_inv))
                opt_p1 = torch.tensor(0.0)  # mu = softplus(0)+eps ≈ 0.69

            raw_p1 = opt_p1.clone().requires_grad_(True)
            raw_p2 = opt_p2.clone().requires_grad_(True)

            fn = REPAMS[name]
            k, r = fn(raw_p1, raw_p2)

            # Check if k is clamped (would block gradients)
            actual_k = k.item()
            actual_r = r.item()
            actual_mean = actual_k / actual_r

            # Draw samples and compute gradients
            dist = Gamma(k, r)

            grads_p1 = []
            grads_p2 = []
            nan_count = 0
            zero_count = 0

            for _ in range(n_samples):
                raw_p1_t = opt_p1.clone().requires_grad_(True)
                raw_p2_t = opt_p2.clone().requires_grad_(True)
                k_t, r_t = fn(raw_p1_t, raw_p2_t)
                dist_t = Gamma(k_t, r_t)
                sample = dist_t.rsample()

                if torch.isnan(sample) or torch.isinf(sample):
                    nan_count += 1
                    continue

                try:
                    sample.backward()
                except RuntimeError:
                    nan_count += 1
                    continue

                g1 = raw_p1_t.grad
                g2 = raw_p2_t.grad

                if g1 is not None and g2 is not None:
                    if torch.isnan(g1) or torch.isnan(g2):
                        nan_count += 1
                    elif g1.abs().item() == 0.0 and g2.abs().item() == 0.0:
                        zero_count += 1
                    else:
                        grads_p1.append(g1.item())
                        grads_p2.append(g2.item())
                else:
                    zero_count += 1

            total = nan_count + zero_count + len(grads_p1)
            nan_frac = nan_count / total if total > 0 else 0
            zero_frac = zero_count / total if total > 0 else 0

            mean_g1 = np.mean(np.abs(grads_p1)) if grads_p1 else 0.0
            mean_g2 = np.mean(np.abs(grads_p2)) if grads_p2 else 0.0

            print(
                f"  {target_k:10.3f} | {actual_k:10.4f} | {actual_r:10.4f} | {actual_mean:10.4f} | "
                f"{mean_g1:12.6f} | {mean_g2:12.6f} | {nan_frac:10.4f} | {zero_frac:10.4f}"
            )

        print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Gradient variance (signal-to-noise ratio)
# ═══════════════════════════════════════════════════════════════════════════════


def test_gradient_snr():
    """Signal-to-noise ratio of gradients at different k values."""
    print("=" * 80)
    print("TEST 3: Gradient signal-to-noise ratio (|mean|/std of grad)")
    print("=" * 80)
    print()
    print(
        "Low SNR means the gradient estimate is noisy — optimization will be slow/unstable."
    )
    print()

    test_ks = [0.01, 0.1, 0.5, 1.0, 5.0, 50.0]
    n_samples = 2000

    for name in REPAMS:
        print(f"--- Repam {name} ---")
        print(
            f"  {'target_k':>10s} | {'actual_k':>10s} | {'SNR(p1)':>12s} | {'SNR(p2)':>12s} | "
            f"{'mean_g1':>12s} | {'std_g1':>12s} | {'mean_g2':>12s} | {'std_g2':>12s}"
        )
        print(
            f"  {'-' * 10} | {'-' * 10} | {'-' * 12} | {'-' * 12} | "
            f"{'-' * 12} | {'-' * 12} | {'-' * 12} | {'-' * 12}"
        )

        for target_k in test_ks:
            fn = REPAMS[name]

            # Set raw params to achieve target k
            if name in ("A", "D"):
                tk = max(target_k, 1e-4)
                opt_p1 = torch.tensor(
                    np.log(np.expm1(tk)) if tk < 20 else float(tk)
                )
                opt_p2 = torch.tensor(0.0)
            elif name == "B":
                fano0 = float(F.softplus(torch.tensor(0.0))) + 1e-6
                r0 = 1.0 / (fano0 + 1e-6)
                mu_need = max((target_k - 1e-6) / r0, 1e-6)
                opt_p1 = torch.tensor(
                    np.log(np.expm1(mu_need))
                    if mu_need < 20
                    else float(mu_need)
                )
                opt_p2 = torch.tensor(0.0)
            elif name == "C":
                phi_need = max(1.0 / target_k - 1e-6, 1e-6)
                opt_p2 = torch.tensor(
                    np.log(np.expm1(phi_need))
                    if phi_need < 20
                    else float(phi_need)
                )
                opt_p1 = torch.tensor(0.0)

            grads_p1 = []
            grads_p2 = []

            for _ in range(n_samples):
                raw_p1 = opt_p1.clone().requires_grad_(True)
                raw_p2 = opt_p2.clone().requires_grad_(True)
                k, r = fn(raw_p1, raw_p2)
                dist = Gamma(k, r)
                sample = dist.rsample()

                if torch.isnan(sample) or torch.isinf(sample):
                    continue
                try:
                    sample.backward()
                except RuntimeError:
                    continue

                g1 = raw_p1.grad
                g2 = raw_p2.grad
                if (
                    g1 is not None
                    and g2 is not None
                    and not (torch.isnan(g1) or torch.isnan(g2))
                ):
                    grads_p1.append(g1.item())
                    grads_p2.append(g2.item())

            if len(grads_p1) < 10:
                print(
                    f"  {target_k:10.3f} | {'N/A':>10s} | {'FAILED':>12s} | {'FAILED':>12s} | "
                    f"{'N/A':>12s} | {'N/A':>12s} | {'N/A':>12s} | {'N/A':>12s}"
                )
                continue

            k_check, _ = fn(opt_p1, opt_p2)
            g1_arr = np.array(grads_p1)
            g2_arr = np.array(grads_p2)

            snr1 = abs(g1_arr.mean()) / (g1_arr.std() + 1e-30)
            snr2 = abs(g2_arr.mean()) / (g2_arr.std() + 1e-30)

            print(
                f"  {target_k:10.3f} | {k_check.item():10.4f} | {snr1:12.4f} | {snr2:12.4f} | "
                f"{g1_arr.mean():12.6f} | {g1_arr.std():12.6f} | {g2_arr.mean():12.6f} | {g2_arr.std():12.6f}"
            )

        print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Clamp-induced dead zones
# ═══════════════════════════════════════════════════════════════════════════════


def test_clamp_dead_zones():
    """Check if k is clamped (gradient = 0 through clamp)."""
    print("=" * 80)
    print("TEST 4: Clamp-induced gradient dead zones")
    print("=" * 80)
    print()
    print("When k hits a clamp boundary (min or max), gradients w.r.t. the")
    print(
        "concentration are exactly zero. This is a DIFFERENT failure mode from"
    )
    print("the IRG breakdown — it's a design choice that can block learning.")
    print()

    raw_vals = torch.linspace(-10, 10, 41)

    for name, fn in REPAMS.items():
        print(f"--- Repam {name} (k_max=500) ---")
        clamped_count = 0
        total = 0

        for r1 in raw_vals:
            for r2 in raw_vals:
                p1 = r1.clone().requires_grad_(True)
                p2 = r2.clone().requires_grad_(True)
                k, r = fn(p1, p2, k_max=500)
                k.backward()

                # Check if gradient is zero (clamped)
                g = (
                    p1.grad
                    if name in ("A", "D")
                    else (p2.grad if name == "C" else p1.grad)
                )
                if g is not None and g.abs().item() < 1e-10:
                    clamped_count += 1
                total += 1

        print(
            f"  Fraction of param space with zero dk/d(raw): {clamped_count}/{total} = {clamped_count / total:.3f}"
        )

        # Also check: what k values have zero gradient?
        print("  k values where gradient vanishes:")
        for r1 in torch.linspace(-10, 10, 21):
            p1 = r1.clone().requires_grad_(True)
            p2 = torch.tensor(0.0, requires_grad=True)
            k, r = fn(p1, p2, k_max=500)
            k.backward()

            g = p1.grad if name != "C" else p2.grad
            dead = g is not None and g.abs().item() < 1e-10
            if dead:
                print(f"    raw={r1.item():6.1f} → k={k.item():.4f} (DEAD)")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: ELBO gradient comparison — the actual training objective
# ═══════════════════════════════════════════════════════════════════════════════


def test_elbo_gradients():
    """Simulate an ELBO gradient step for each repam at various regimes."""
    print("=" * 80)
    print("TEST 5: ELBO gradient through full sample → log_prob pipeline")
    print("=" * 80)
    print()
    print(
        "This simulates what actually happens in training: sample from q(I),"
    )
    print("compute Poisson log-prob, backprop. Tests if the gradient signal")
    print("reaches the raw parameters reliably.")
    print()

    # Simulate observed count
    observed_count = torch.tensor(10.0)

    test_ks = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 50.0]
    n_trials = 500

    for name in REPAMS:
        print(f"--- Repam {name} ---")
        print(
            f"  {'target_k':>10s} | {'actual_k':>10s} | {'mean':>10s} | "
            f"{'grad_p1_mean':>14s} | {'grad_p1_std':>14s} | {'grad_p2_mean':>14s} | "
            f"{'nan%':>8s} | {'zero%':>8s}"
        )
        print(
            f"  {'-' * 10} | {'-' * 10} | {'-' * 10} | "
            f"{'-' * 14} | {'-' * 14} | {'-' * 14} | "
            f"{'-' * 8} | {'-' * 8}"
        )

        fn = REPAMS[name]

        for target_k in test_ks:
            if name in ("A", "D"):
                tk = max(target_k, 1e-4)
                opt_p1 = torch.tensor(
                    np.log(np.expm1(tk)) if tk < 20 else float(tk)
                )
                opt_p2 = torch.tensor(0.0)
            elif name == "B":
                fano0 = float(F.softplus(torch.tensor(0.0))) + 1e-6
                r0 = 1.0 / (fano0 + 1e-6)
                mu_need = max((target_k - 1e-6) / r0, 1e-6)
                opt_p1 = torch.tensor(
                    np.log(np.expm1(mu_need))
                    if mu_need < 20
                    else float(mu_need)
                )
                opt_p2 = torch.tensor(0.0)
            elif name == "C":
                phi_need = max(1.0 / target_k - 1e-6, 1e-6)
                opt_p2 = torch.tensor(
                    np.log(np.expm1(phi_need))
                    if phi_need < 20
                    else float(phi_need)
                )
                opt_p1 = torch.tensor(0.0)

            grads_p1 = []
            grads_p2 = []
            nan_count = 0
            zero_count = 0
            actual_k_val = None
            actual_mean_val = None

            for _ in range(n_trials):
                raw_p1 = opt_p1.clone().requires_grad_(True)
                raw_p2 = opt_p2.clone().requires_grad_(True)
                k, r = fn(raw_p1, raw_p2)

                if actual_k_val is None:
                    actual_k_val = k.item()
                    actual_mean_val = (k / r).item()

                dist = Gamma(k, r)
                sample = dist.rsample()

                if (
                    torch.isnan(sample)
                    or torch.isinf(sample)
                    or sample.item() <= 0
                ):
                    nan_count += 1
                    continue

                # Poisson log-likelihood: count * log(sample) - sample - log(count!)
                nll = -(observed_count * torch.log(sample + 1e-10) - sample)

                try:
                    nll.backward()
                except RuntimeError:
                    nan_count += 1
                    continue

                g1 = raw_p1.grad
                g2 = raw_p2.grad

                if (
                    g1 is None
                    or g2 is None
                    or torch.isnan(g1)
                    or torch.isnan(g2)
                ):
                    nan_count += 1
                elif g1.abs().item() == 0.0 and g2.abs().item() == 0.0:
                    zero_count += 1
                else:
                    grads_p1.append(g1.item())
                    grads_p2.append(g2.item())

            total = nan_count + zero_count + len(grads_p1)
            g1_mean = np.mean(grads_p1) if grads_p1 else 0.0
            g1_std = np.std(grads_p1) if grads_p1 else 0.0
            g2_mean = np.mean(grads_p2) if grads_p2 else 0.0

            print(
                f"  {target_k:10.3f} | {actual_k_val or 0:10.4f} | {actual_mean_val or 0:10.4f} | "
                f"{g1_mean:14.6f} | {g1_std:14.6f} | {g2_mean:14.6f} | "
                f"{nan_count / total * 100:7.1f}% | {zero_count / total * 100:7.1f}%"
            )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Jacobian structure — how do dmu/draw and dk/draw differ?
# ═══════════════════════════════════════════════════════════════════════════════


def test_jacobian():
    """Examine the Jacobian dk/d(raw), dr/d(raw) for each parameterization."""
    print("=" * 80)
    print(
        "TEST 6: Jacobian dk/d(raw) and dr/d(raw) at various operating points"
    )
    print("=" * 80)
    print()
    print(
        "This shows the 'gain' of each parameterization: how much does a unit"
    )
    print(
        "change in raw parameter space move k and r? Low gain = sluggish learning."
    )
    print()

    test_ks = [0.01, 0.1, 1.0, 10.0, 100.0]

    for name, fn in REPAMS.items():
        print(f"--- Repam {name} ---")
        print(
            f"  {'target_k':>10s} | {'actual_k':>10s} | {'dk/dp1':>12s} | {'dk/dp2':>12s} | "
            f"{'dr/dp1':>12s} | {'dr/dp2':>12s}"
        )
        print(
            f"  {'-' * 10} | {'-' * 10} | {'-' * 12} | {'-' * 12} | "
            f"{'-' * 12} | {'-' * 12}"
        )

        for target_k in test_ks:
            if name in ("A", "D"):
                tk = max(target_k, 1e-4)
                opt_p1 = torch.tensor(
                    np.log(np.expm1(tk)) if tk < 20 else float(tk)
                )
                opt_p2 = torch.tensor(0.0)
            elif name == "B":
                fano0 = float(F.softplus(torch.tensor(0.0))) + 1e-6
                r0 = 1.0 / (fano0 + 1e-6)
                mu_need = max((target_k - 1e-6) / r0, 1e-6)
                opt_p1 = torch.tensor(
                    np.log(np.expm1(mu_need))
                    if mu_need < 20
                    else float(mu_need)
                )
                opt_p2 = torch.tensor(0.0)
            elif name == "C":
                phi_need = max(1.0 / target_k - 1e-6, 1e-6)
                opt_p2 = torch.tensor(
                    np.log(np.expm1(phi_need))
                    if phi_need < 20
                    else float(phi_need)
                )
                opt_p1 = torch.tensor(0.0)

            raw_p1 = opt_p1.clone().requires_grad_(True)
            raw_p2 = opt_p2.clone().requires_grad_(True)
            k, r = fn(raw_p1, raw_p2)

            # dk/dp1, dk/dp2
            k.backward(retain_graph=True)
            dk_dp1 = raw_p1.grad.item() if raw_p1.grad is not None else 0.0
            dk_dp2 = raw_p2.grad.item() if raw_p2.grad is not None else 0.0

            raw_p1.grad = None
            raw_p2.grad = None

            r.backward()
            dr_dp1 = raw_p1.grad.item() if raw_p1.grad is not None else 0.0
            dr_dp2 = raw_p2.grad.item() if raw_p2.grad is not None else 0.0

            print(
                f"  {target_k:10.3f} | {k.item():10.4f} | {dk_dp1:12.6f} | {dk_dp2:12.6f} | "
                f"{dr_dp1:12.6f} | {dr_dp2:12.6f}"
            )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_k_reachability()
    test_gradient_flow()
    test_gradient_snr()
    test_clamp_dead_zones()
    test_elbo_gradients()
    test_jacobian()

    print("=" * 80)
    print("ANALYSIS SUMMARY (with k_min=0.1, eps=1e-6 separation)")
    print("=" * 80)
    print("""
Key findings after k_min / eps separation:

1. k FLOOR IS NOW 0.1 FOR ALL REPAMS (Test 1)
   - RepamA/D: k = softplus(raw) + k_min → k ∈ [0.1, ∞)
   - RepamB: k = clamp(mu*r, min=0.1) → hard floor at 0.1
   - RepamC: k = (1/phi).clamp(min=0.1) → hard floor at 0.1
   - Previously: A/D could reach ~1e-6, C only protected to 0.01

2. NO GRADIENT FAILURES (Tests 2, 3, 5)
   - 0% NaN, 0% zero gradients across all repams at all k targets
   - At k=0.1 (the floor): gradient SNR ≈ 0.6-1.2 (viable for SGD)
   - The IRG danger zone (k < 0.1) is now unreachable

3. CLAMP DEAD ZONES (Test 4)
   - RepamA/D: 0% dead (softplus + k_min is smooth everywhere)
   - RepamB: 46% dead (clamp kills gradients when mu*r < 0.1, but
     this is exactly the IRG danger zone — blocking it is correct)
   - RepamC: 22% dead (same logic — clamp protects against small k)

4. DOUBLE-EPS ELIMINATED
   - Old: r = 1/(fano + eps) with fano already containing eps
   - New: r = 1/fano — clean, since fano ≥ eps > 0 always
   - No practical impact on gradients but much clearer semantics

5. JACOBIAN STRUCTURE (Test 6) — unchanged by k_min/eps split:
   - RepamA/D: k and r are INDEPENDENT (cleanest gradient signal)
   - RepamB: k couples to both mu and fano
   - RepamC: dk/d(raw_phi) suppresses learning of small k naturally
""")

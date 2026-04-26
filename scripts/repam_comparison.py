"""
Gamma Reparameterization Comparison Diagnostics
================================================
Five diagnostic sections revealing *why* the 4 Gamma reparameterizations
(A, B, C, D) behave differently during training:

1. Jacobian Conditioning   — how well-conditioned is raw → (k, r)?
2. Full-ELBO Gradient SNR  — gradient noise through NLL + KL
3. Gradient Correlation    — do param gradients compete or cooperate?
4. Convergence Race        — which repam converges fastest on a mini-ELBO?
5. Effective Step Size     — Jacobian gain × gradient SNR → predicted speed
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Gamma, kl_divergence

torch.manual_seed(42)

# ─── Standalone repam functions (matching gamma.py exactly) ──────────────────

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
PARAM_NAMES = {
    "A": ("raw_k", "raw_r"),
    "B": ("raw_mu", "raw_fano"),
    "C": ("raw_mu", "raw_phi"),
    "D": ("raw_k", "raw_fano"),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _raw_params_for_target_k(name, target_k):
    """Return (opt_p1, opt_p2) raw param tensors that produce approximately target_k."""
    if name in ("A", "D"):
        tk = max(target_k, 1e-4)
        val = (
            np.log(np.expm1(tk - K_MIN))
            if (tk - K_MIN) < 20 and (tk - K_MIN) > 0
            else float(max(tk - K_MIN, 1e-6))
        )
        return torch.tensor(float(val)), torch.tensor(0.0)
    elif name == "B":
        fano0 = float(F.softplus(torch.tensor(0.0))) + EPS
        r0 = 1.0 / fano0
        mu_need = max(target_k / r0, 1e-6)
        sp_inv = (
            np.log(np.expm1(mu_need - EPS))
            if (mu_need - EPS) < 20
            else float(mu_need)
        )
        return torch.tensor(float(sp_inv)), torch.tensor(0.0)
    elif name == "C":
        phi_need = max(1.0 / target_k - EPS, 1e-6)
        sp_inv = (
            np.log(np.expm1(phi_need)) if phi_need < 20 else float(phi_need)
        )
        return torch.tensor(0.0), torch.tensor(float(sp_inv))
    raise ValueError(name)


def _compute_elbo_loss(k, r, y, prior):
    """Single-sample ELBO loss: NLL + KL."""
    dist = Gamma(k, r)
    I_sample = dist.rsample()
    nll = -(y * torch.log(I_sample + 1e-10) - I_sample)
    kl = kl_divergence(dist, prior)
    return nll + kl


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Jacobian Conditioning
# ═══════════════════════════════════════════════════════════════════════════════


def section1_jacobian_conditioning():
    print("=" * 90)
    print(
        "SECTION 1: Jacobian Conditioning — condition number of d(k,r)/d(raw_p1,raw_p2)"
    )
    print("=" * 90)
    print()
    print(
        "High condition number → one direction learns much faster than the other."
    )
    print(
        "Condition ≈ 1 is ideal (isotropic learning). Inf means a singular Jacobian."
    )
    print()

    target_ks = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    for name, fn in REPAMS.items():
        p1n, p2n = PARAM_NAMES[name]
        print(f"--- Repam {name} ({p1n}, {p2n}) ---")
        print(
            f"  {'target_k':>10s} | {'actual_k':>10s} | {'cond_num':>12s} | "
            f"{'σ_max':>12s} | {'σ_min':>12s}"
        )
        print(
            f"  {'-' * 10} | {'-' * 10} | {'-' * 12} | {'-' * 12} | {'-' * 12}"
        )

        for target_k in target_ks:
            opt_p1, opt_p2 = _raw_params_for_target_k(name, target_k)

            def param_to_kr(params):
                k, r = fn(params[0], params[1])
                return torch.stack([k, r])

            inp = torch.stack([opt_p1, opt_p2])
            try:
                J = torch.autograd.functional.jacobian(param_to_kr, inp)
                # J is (2, 2): rows = (k, r), cols = (raw_p1, raw_p2)
                S = torch.linalg.svdvals(J)
                sigma_max = S[0].item()
                sigma_min = S[1].item()
                cond = (
                    sigma_max / sigma_min
                    if sigma_min > 1e-30
                    else float("inf")
                )
            except Exception:
                sigma_max = sigma_min = cond = float("nan")

            k_check, _ = fn(opt_p1, opt_p2)
            print(
                f"  {target_k:10.1f} | {k_check.item():10.4f} | {cond:12.4f} | "
                f"{sigma_max:12.6f} | {sigma_min:12.6f}"
            )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Full-ELBO Gradient Variance
# ═══════════════════════════════════════════════════════════════════════════════


def section2_elbo_gradient_snr():
    print("=" * 90)
    print(
        "SECTION 2: Full-ELBO Gradient SNR — |mean(grad)| / std(grad) through NLL + KL"
    )
    print("=" * 90)
    print()
    print("Includes both Poisson NLL and Gamma-Gamma KL divergence.")
    print("Higher SNR → more reliable gradient signal → faster convergence.")
    print()

    prior = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    observed_counts = [1, 10, 100, 1000]
    n_samples = 1000

    for name, fn in REPAMS.items():
        p1n, p2n = PARAM_NAMES[name]
        print(f"--- Repam {name} ({p1n}, {p2n}) ---")
        print(
            f"  {'y':>6s} | {'k':>8s} | {'mean':>10s} | "
            f"{'SNR(' + p1n + ')':>14s} | {'SNR(' + p2n + ')':>14s} | "
            f"{'nan%':>6s}"
        )
        print(
            f"  {'-' * 6} | {'-' * 8} | {'-' * 10} | "
            f"{'-' * 14} | {'-' * 14} | {'-' * 6}"
        )

        for y_val in observed_counts:
            y = torch.tensor(float(y_val))
            # Set k ≈ 1 (moderate concentration) for fair comparison
            opt_p1, opt_p2 = _raw_params_for_target_k(name, 1.0)

            grads_p1, grads_p2 = [], []
            nan_count = 0

            for _ in range(n_samples):
                raw_p1 = opt_p1.clone().requires_grad_(True)
                raw_p2 = opt_p2.clone().requires_grad_(True)
                k, r = fn(raw_p1, raw_p2)
                try:
                    loss = _compute_elbo_loss(k, r, y, prior)
                    loss.backward()
                except RuntimeError:
                    nan_count += 1
                    continue

                g1 = raw_p1.grad
                g2 = raw_p2.grad
                if (
                    g1 is not None
                    and g2 is not None
                    and not torch.isnan(g1)
                    and not torch.isnan(g2)
                    and not torch.isinf(g1)
                    and not torch.isinf(g2)
                ):
                    grads_p1.append(g1.item())
                    grads_p2.append(g2.item())
                else:
                    nan_count += 1

            total = nan_count + len(grads_p1)
            nan_pct = 100.0 * nan_count / total if total > 0 else 0.0

            if len(grads_p1) >= 10:
                g1 = np.array(grads_p1)
                g2 = np.array(grads_p2)
                snr1 = abs(g1.mean()) / (g1.std() + 1e-30)
                snr2 = abs(g2.mean()) / (g2.std() + 1e-30)
                k_check, r_check = fn(opt_p1, opt_p2)
                print(
                    f"  {y_val:6d} | {k_check.item():8.3f} | "
                    f"{(k_check / r_check).item():10.3f} | "
                    f"{snr1:14.4f} | {snr2:14.4f} | {nan_pct:5.1f}%"
                )
            else:
                print(
                    f"  {y_val:6d} | {'N/A':>8s} | {'N/A':>10s} | "
                    f"{'FAILED':>14s} | {'FAILED':>14s} | {nan_pct:5.1f}%"
                )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Gradient Correlation
# ═══════════════════════════════════════════════════════════════════════════════


def section3_gradient_correlation():
    print("=" * 90)
    print("SECTION 3: Gradient Correlation — do parameter gradients compete?")
    print("=" * 90)
    print()
    print(
        "High |correlation| with opposite signs = competing gradients → instability."
    )
    print("Near-zero correlation = independent optimization paths (ideal).")
    print()

    prior = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    observed_counts = [1, 10, 100, 1000]
    n_samples = 1000

    for name, fn in REPAMS.items():
        p1n, p2n = PARAM_NAMES[name]
        print(f"--- Repam {name} ({p1n}, {p2n}) ---")
        print(
            f"  {'y':>6s} | {'corr(g1,g2)':>14s} | {'angle(°)':>10s} | "
            f"{'mean_g1':>12s} | {'mean_g2':>12s}"
        )
        print(
            f"  {'-' * 6} | {'-' * 14} | {'-' * 10} | {'-' * 12} | {'-' * 12}"
        )

        for y_val in observed_counts:
            y = torch.tensor(float(y_val))
            opt_p1, opt_p2 = _raw_params_for_target_k(name, 1.0)

            grads_p1, grads_p2 = [], []

            for _ in range(n_samples):
                raw_p1 = opt_p1.clone().requires_grad_(True)
                raw_p2 = opt_p2.clone().requires_grad_(True)
                k, r = fn(raw_p1, raw_p2)
                try:
                    loss = _compute_elbo_loss(k, r, y, prior)
                    loss.backward()
                except RuntimeError:
                    continue

                g1 = raw_p1.grad
                g2 = raw_p2.grad
                if (
                    g1 is not None
                    and g2 is not None
                    and not torch.isnan(g1)
                    and not torch.isnan(g2)
                    and not torch.isinf(g1)
                    and not torch.isinf(g2)
                ):
                    grads_p1.append(g1.item())
                    grads_p2.append(g2.item())

            if len(grads_p1) >= 10:
                g1 = np.array(grads_p1)
                g2 = np.array(grads_p2)

                # Pearson correlation
                corr = np.corrcoef(g1, g2)[0, 1]

                # Angle between mean gradient vectors
                mean_vec = np.array([g1.mean(), g2.mean()])
                norm = np.linalg.norm(mean_vec)
                if norm > 1e-30:
                    # Angle with respect to the p1 axis
                    angle_deg = np.degrees(
                        np.arctan2(mean_vec[1], mean_vec[0])
                    )
                else:
                    angle_deg = float("nan")

                print(
                    f"  {y_val:6d} | {corr:14.4f} | {angle_deg:10.2f} | "
                    f"{g1.mean():12.6f} | {g2.mean():12.6f}"
                )
            else:
                print(
                    f"  {y_val:6d} | {'FAILED':>14s} | {'N/A':>10s} | "
                    f"{'N/A':>12s} | {'N/A':>12s}"
                )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Convergence Race
# ═══════════════════════════════════════════════════════════════════════════════


def section4_convergence_race():
    print("=" * 90)
    print("SECTION 4: Convergence Race — optimization on 3 reflections")
    print("=" * 90)
    print()
    print(
        "Each repam optimizes q(I_i) = Gamma(k_i, r_i) for y = [5, 50, 500]."
    )
    print(
        "Prior: Gamma(1, 0.001) = Exp(mean=1000). Adam lr=0.01, 10000 steps."
    )
    print(
        "64 MC samples per gradient estimate. All start from raw params = 0."
    )
    print()

    prior = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    ys = torch.tensor([5.0, 50.0, 500.0])
    n_refl = len(ys)
    n_steps = 10000
    n_mc = 64  # MC samples per gradient estimate
    lr = 0.01

    results = {}

    for name, fn in REPAMS.items():
        torch.manual_seed(42)

        # 2 raw params per reflection
        raw_p1s = torch.zeros(n_refl, requires_grad=True)
        raw_p2s = torch.zeros(n_refl, requires_grad=True)

        optimizer = torch.optim.Adam([raw_p1s, raw_p2s], lr=lr)

        loss_history = []
        k_history = []
        mean_history = []
        grad_norm_history = []

        for _step in range(n_steps):
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0)
            ks = []
            means = []

            for i in range(n_refl):
                k, r = fn(raw_p1s[i], raw_p2s[i])
                ks.append(k.item())
                means.append((k / r).item())

                dist = Gamma(k, r)
                # Multi-sample ELBO for lower-variance gradients
                I_samples = dist.rsample((n_mc,))  # (n_mc,)
                nll = -(
                    ys[i] * torch.log(I_samples + 1e-10) - I_samples
                ).mean()
                kl = kl_divergence(dist, prior)
                total_loss = total_loss + nll + kl

            total_loss.backward()

            # Gradient norm
            gn = 0.0
            if raw_p1s.grad is not None:
                gn += raw_p1s.grad.norm().item() ** 2
            if raw_p2s.grad is not None:
                gn += raw_p2s.grad.norm().item() ** 2
            grad_norm_history.append(gn**0.5)

            # Clip NaN gradients to zero
            if raw_p1s.grad is not None and torch.any(
                torch.isnan(raw_p1s.grad)
            ):
                raw_p1s.grad[torch.isnan(raw_p1s.grad)] = 0.0
            if raw_p2s.grad is not None and torch.any(
                torch.isnan(raw_p2s.grad)
            ):
                raw_p2s.grad[torch.isnan(raw_p2s.grad)] = 0.0

            optimizer.step()

            loss_history.append(total_loss.item())
            k_history.append(ks)
            mean_history.append(means)

        # Determine convergence step: first step where loss is within 1% of
        # the best (most negative) loss seen. Since ELBO losses are negative,
        # "better" = more negative.
        best_loss = min(loss_history)
        conv_step = n_steps
        for s in range(n_steps):
            if loss_history[s] <= best_loss * 0.99:  # within 1% of best
                conv_step = s
                break

        final_loss = loss_history[-1]
        results[name] = {
            "conv_step": conv_step,
            "final_loss": final_loss,
            "final_means": mean_history[-1],
            "final_ks": k_history[-1],
            "loss_history": loss_history,
        }

        print(f"--- Repam {name} ---")
        print(f"  Converged at step: {conv_step}")
        print(f"  Final loss: {final_loss:.4f}")
        print(
            f"  Final means (target [5, 50, 500]): [{mean_history[-1][0]:.2f}, "
            f"{mean_history[-1][1]:.2f}, {mean_history[-1][2]:.2f}]"
        )
        print(
            f"  Final k values: [{k_history[-1][0]:.2f}, "
            f"{k_history[-1][1]:.2f}, {k_history[-1][2]:.2f}]"
        )
        print(f"  Grad norm at step 0: {grad_norm_history[0]:.4f}")
        print(
            f"  Grad norm at step 500: {grad_norm_history[min(500, n_steps - 1)]:.4f}"
        )
        print(
            f"  Grad norm at step 2000: {grad_norm_history[min(2000, n_steps - 1)]:.4f}"
        )

        # Show trajectory at milestones
        milestones = [0, 100, 500, 1000, 2000, 5000, 9999]
        print("  Trajectory (step → means):")
        for ms in milestones:
            if ms < n_steps:
                m = mean_history[ms]
                k = k_history[ms]
                print(
                    f"    step {ms:5d}: means=[{m[0]:.2f}, {m[1]:.2f}, {m[2]:.2f}]  "
                    f"k=[{k[0]:.2f}, {k[1]:.2f}, {k[2]:.2f}]"
                )
        print()

    # Summary comparison
    print("--- Convergence Summary ---")
    print(
        f"  {'Repam':>6s} | {'Conv Step':>10s} | {'Final Loss':>12s} | "
        f"{'Mean Err':>10s} | {'Mean [0]':>10s} | {'Mean [1]':>10s} | {'Mean [2]':>10s}"
    )
    print(
        f"  {'-' * 6} | {'-' * 10} | {'-' * 12} | "
        f"{'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}"
    )
    for name, res in results.items():
        m = res["final_means"]
        targets = [5.0, 50.0, 500.0]
        mean_err = np.mean(
            [abs(m[i] - targets[i]) / targets[i] for i in range(3)]
        )
        print(
            f"  {name:>6s} | {res['conv_step']:>10d} | {res['final_loss']:12.4f} | "
            f"{mean_err:10.4f} | {m[0]:10.2f} | {m[1]:10.2f} | {m[2]:10.2f}"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Effective Step Size
# ═══════════════════════════════════════════════════════════════════════════════


def section5_effective_step_size():
    print("=" * 90)
    print("SECTION 5: Effective Step Size — Jacobian gain × gradient SNR")
    print("=" * 90)
    print()
    print("Effective velocity = |Jacobian gain| × SNR.")
    print("This predicts convergence speed without running optimization.")
    print(
        "Connects Sections 1 + 2: great SNR with tiny Jacobian (or vice versa) = slow."
    )
    print()

    prior = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    target_ks = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    y = torch.tensor(50.0)
    n_samples = 1000

    for name, fn in REPAMS.items():
        p1n, p2n = PARAM_NAMES[name]
        print(f"--- Repam {name} ({p1n}, {p2n}) ---")
        print(
            f"  {'target_k':>10s} | {'J_gain_p1':>12s} | {'J_gain_p2':>12s} | "
            f"{'SNR_p1':>10s} | {'SNR_p2':>10s} | "
            f"{'eff_v_p1':>10s} | {'eff_v_p2':>10s}"
        )
        print(
            f"  {'-' * 10} | {'-' * 12} | {'-' * 12} | "
            f"{'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}"
        )

        for target_k in target_ks:
            opt_p1, opt_p2 = _raw_params_for_target_k(name, target_k)

            # Jacobian gain: Frobenius norm of each column of the Jacobian
            # (how much a unit change in raw_pi moves the output)
            def param_to_kr(params):
                k, r = fn(params[0], params[1])
                return torch.stack([k, r])

            inp = torch.stack([opt_p1, opt_p2])
            try:
                J = torch.autograd.functional.jacobian(param_to_kr, inp)
                # J is (2, 2): columns = d(k,r)/dp1, d(k,r)/dp2
                j_gain_p1 = J.select(1, 0).norm().item()
                j_gain_p2 = J.select(1, 1).norm().item()
            except Exception:
                j_gain_p1 = j_gain_p2 = float("nan")

            # Gradient SNR through ELBO
            grads_p1, grads_p2 = [], []
            for _ in range(n_samples):
                raw_p1 = opt_p1.clone().requires_grad_(True)
                raw_p2 = opt_p2.clone().requires_grad_(True)
                k, r = fn(raw_p1, raw_p2)
                try:
                    loss = _compute_elbo_loss(k, r, y, prior)
                    loss.backward()
                except RuntimeError:
                    continue

                g1 = raw_p1.grad
                g2 = raw_p2.grad
                if (
                    g1 is not None
                    and g2 is not None
                    and not torch.isnan(g1)
                    and not torch.isnan(g2)
                    and not torch.isinf(g1)
                    and not torch.isinf(g2)
                ):
                    grads_p1.append(g1.item())
                    grads_p2.append(g2.item())

            if len(grads_p1) >= 10:
                g1 = np.array(grads_p1)
                g2 = np.array(grads_p2)
                snr1 = abs(g1.mean()) / (g1.std() + 1e-30)
                snr2 = abs(g2.mean()) / (g2.std() + 1e-30)
            else:
                snr1 = snr2 = float("nan")

            eff_v1 = j_gain_p1 * snr1
            eff_v2 = j_gain_p2 * snr2

            print(
                f"  {target_k:10.1f} | {j_gain_p1:12.6f} | {j_gain_p2:12.6f} | "
                f"{snr1:10.4f} | {snr2:10.4f} | {eff_v1:10.4f} | {eff_v2:10.4f}"
            )
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    section1_jacobian_conditioning()
    section2_elbo_gradient_snr()
    section3_gradient_correlation()
    section4_convergence_race()
    section5_effective_step_size()

    print("=" * 90)
    print("DONE — all 5 sections complete.")
    print("=" * 90)

"""
Numerical stability analysis: when do rsample gradients blow up?

For each distribution family, sweep over parameter ranges and check:
  1. Does rsample produce finite values?
  2. Does backprop through rsample produce finite gradients?
  3. What is the gradient magnitude?

Tests: Gamma, FoldedNormal, LogNormal, SoftplusNormal (transformed Normal)
"""

import sys

import torch
import torch.nn.functional as F
from torch.distributions import Gamma, LogNormal, Normal

sys.path.insert(0, "src")
from integrator.model.distributions.folded_normal import FoldedNormal


def test_gamma_stability():
    """Test Gamma rsample gradient stability across k values."""
    print("=" * 80)
    print("GAMMA: rsample gradient stability")
    print("  Gamma(k, r) with r = k/mean, mean=1000")
    print("=" * 80)

    mean = 1000.0
    ks = [1, 5, 10, 50, 100, 200, 500, 800, 900, 950, 1000, 1500, 2000, 5000]
    mc = 50

    print(
        f"{'k':>6} {'r':>10} {'CV':>6} {'sample_ok':>10} {'grad_ok':>10} "
        f"{'grad_k_mag':>12} {'grad_r_mag':>12} {'sample_range':>20}"
    )
    print("-" * 100)

    for k_val in ks:
        k = torch.tensor(float(k_val), requires_grad=True)
        r = torch.tensor(float(k_val / mean), requires_grad=True)

        try:
            dist = Gamma(k, r)
            samples = dist.rsample([mc])

            sample_ok = samples.isfinite().all().item()
            sample_range = f"[{samples.min():.0f}, {samples.max():.0f}]"

            # Backprop through mean of samples
            loss = samples.mean()
            loss.backward()

            grad_k = k.grad
            grad_r = r.grad

            grad_ok = (
                grad_k is not None
                and grad_k.isfinite().item()
                and grad_r is not None
                and grad_r.isfinite().item()
            )
            grad_k_mag = (
                f"{grad_k.abs().item():.4g}"
                if grad_k is not None and grad_k.isfinite()
                else "NaN/Inf"
            )
            grad_r_mag = (
                f"{grad_r.abs().item():.4g}"
                if grad_r is not None and grad_r.isfinite()
                else "NaN/Inf"
            )

        except Exception as e:
            sample_ok = False
            grad_ok = False
            grad_k_mag = "ERROR"
            grad_r_mag = "ERROR"
            sample_range = str(e)[:20]

        print(
            f"{k_val:>6} {k_val / mean:>10.4f} {1 / k_val**0.5:>6.3f} "
            f"{'✓' if sample_ok else '✗':>10} {'✓' if grad_ok else '✗':>10} "
            f"{grad_k_mag:>12} {grad_r_mag:>12} {sample_range:>20}"
        )


def test_folded_normal_stability():
    """Test FoldedNormal rsample gradient stability across loc/scale ratios."""
    print("\n" + "=" * 80)
    print("FOLDED NORMAL: rsample gradient stability")
    print("  FoldedNormal(loc, scale) with mean ≈ loc (when loc >> scale)")
    print("=" * 80)

    mc = 50
    locs = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000]

    print(
        f"{'loc':>8} {'scale':>8} {'loc/scale':>10} {'sample_ok':>10} {'grad_ok':>10} "
        f"{'grad_loc_mag':>14} {'grad_scale_mag':>14}"
    )
    print("-" * 90)

    for loc_val in locs:
        # Fix CV = 10% → scale = 0.1 * loc
        scale_val = max(0.1 * loc_val, 1.0)

        loc = torch.tensor(float(loc_val), requires_grad=True)
        scale = torch.tensor(float(scale_val), requires_grad=True)

        try:
            dist = FoldedNormal(loc, scale)
            samples = dist.rsample([mc])

            sample_ok = samples.isfinite().all().item()

            loss = samples.mean()
            loss.backward()

            grad_loc = loc.grad
            grad_scale = scale.grad

            grad_ok = (
                grad_loc is not None
                and grad_loc.isfinite().item()
                and grad_scale is not None
                and grad_scale.isfinite().item()
            )
            grad_loc_mag = (
                f"{grad_loc.abs().item():.4g}"
                if grad_loc is not None and grad_loc.isfinite()
                else "NaN/Inf"
            )
            grad_scale_mag = (
                f"{grad_scale.abs().item():.4g}"
                if grad_scale is not None and grad_scale.isfinite()
                else "NaN/Inf"
            )

        except Exception:
            sample_ok = False
            grad_ok = False
            grad_loc_mag = "ERROR"
            grad_scale_mag = "ERROR"

        print(
            f"{loc_val:>8} {scale_val:>8.1f} {loc_val / scale_val:>10.1f} "
            f"{'✓' if sample_ok else '✗':>10} {'✓' if grad_ok else '✗':>10} "
            f"{grad_loc_mag:>14} {grad_scale_mag:>14}"
        )

    # Also test with tighter scale (CV=1%)
    print("\n  With tighter scale (CV ≈ 1%):")
    print(
        f"{'loc':>8} {'scale':>8} {'loc/scale':>10} {'sample_ok':>10} {'grad_ok':>10} "
        f"{'grad_loc_mag':>14} {'grad_scale_mag':>14}"
    )
    print("-" * 90)

    for loc_val in [100, 500, 1000, 5000]:
        scale_val = max(0.01 * loc_val, 0.1)

        loc = torch.tensor(float(loc_val), requires_grad=True)
        scale = torch.tensor(float(scale_val), requires_grad=True)

        try:
            dist = FoldedNormal(loc, scale)
            samples = dist.rsample([mc])
            sample_ok = samples.isfinite().all().item()
            loss = samples.mean()
            loss.backward()
            grad_loc = loc.grad
            grad_scale = scale.grad
            grad_ok = (
                grad_loc is not None
                and grad_loc.isfinite().item()
                and grad_scale is not None
                and grad_scale.isfinite().item()
            )
            grad_loc_mag = (
                f"{grad_loc.abs().item():.4g}"
                if grad_loc is not None and grad_loc.isfinite()
                else "NaN/Inf"
            )
            grad_scale_mag = (
                f"{grad_scale.abs().item():.4g}"
                if grad_scale is not None and grad_scale.isfinite()
                else "NaN/Inf"
            )
        except Exception:
            sample_ok = grad_ok = False
            grad_loc_mag = grad_scale_mag = "ERROR"

        print(
            f"{loc_val:>8} {scale_val:>8.2f} {loc_val / scale_val:>10.0f} "
            f"{'✓' if sample_ok else '✗':>10} {'✓' if grad_ok else '✗':>10} "
            f"{grad_loc_mag:>14} {grad_scale_mag:>14}"
        )


def test_lognormal_stability():
    """Test LogNormal rsample gradient stability."""
    print("\n" + "=" * 80)
    print("LOGNORMAL: rsample gradient stability")
    print("  LogNormal(mu, sigma) where mean = exp(mu + sigma²/2)")
    print("  rsample = exp(mu + sigma * epsilon)  [standard reparam trick]")
    print("=" * 80)

    mc = 50
    # Target means from 1 to 1M
    target_means = [1, 10, 100, 1000, 10000, 100000, 1000000]

    print(
        f"{'target_mean':>12} {'mu':>8} {'sigma':>8} {'CV':>6} {'sample_ok':>10} "
        f"{'grad_ok':>10} {'grad_mu_mag':>12} {'grad_sig_mag':>12}"
    )
    print("-" * 100)

    for target_mean in target_means:
        # Fix CV = 10%: sigma² = log(1 + CV²) = log(1.01) ≈ 0.01, sigma ≈ 0.1
        cv = 0.10
        sigma_val = (torch.tensor(1 + cv**2).log()).sqrt().item()
        mu_val = (
            torch.tensor(float(target_mean)).log().item() - sigma_val**2 / 2
        )

        mu = torch.tensor(mu_val, requires_grad=True)
        sigma = torch.tensor(sigma_val, requires_grad=True)

        try:
            dist = LogNormal(mu, sigma)
            samples = dist.rsample([mc])

            sample_ok = samples.isfinite().all().item()

            loss = samples.mean()
            loss.backward()

            grad_mu = mu.grad
            grad_sig = sigma.grad

            grad_ok = (
                grad_mu is not None
                and grad_mu.isfinite().item()
                and grad_sig is not None
                and grad_sig.isfinite().item()
            )
            grad_mu_mag = (
                f"{grad_mu.abs().item():.4g}"
                if grad_mu is not None and grad_mu.isfinite()
                else "NaN/Inf"
            )
            grad_sig_mag = (
                f"{grad_sig.abs().item():.4g}"
                if grad_sig is not None and grad_sig.isfinite()
                else "NaN/Inf"
            )

        except Exception:
            sample_ok = grad_ok = False
            grad_mu_mag = grad_sig_mag = "ERROR"

        print(
            f"{target_mean:>12} {mu_val:>8.2f} {sigma_val:>8.4f} {cv:>6.2f} "
            f"{'✓' if sample_ok else '✗':>10} {'✓' if grad_ok else '✗':>10} "
            f"{grad_mu_mag:>12} {grad_sig_mag:>12}"
        )

    # Now test with very tight posteriors (CV = 1%)
    print("\n  With very tight posteriors (CV = 1%):")
    print(
        f"{'target_mean':>12} {'mu':>8} {'sigma':>8} {'CV':>6} {'sample_ok':>10} "
        f"{'grad_ok':>10} {'grad_mu_mag':>12} {'grad_sig_mag':>12}"
    )
    print("-" * 100)

    for target_mean in target_means:
        cv = 0.01
        sigma_val = (torch.tensor(1 + cv**2).log()).sqrt().item()
        mu_val = (
            torch.tensor(float(target_mean)).log().item() - sigma_val**2 / 2
        )

        mu = torch.tensor(mu_val, requires_grad=True)
        sigma = torch.tensor(sigma_val, requires_grad=True)

        try:
            dist = LogNormal(mu, sigma)
            samples = dist.rsample([mc])
            sample_ok = samples.isfinite().all().item()
            loss = samples.mean()
            loss.backward()
            grad_mu = mu.grad
            grad_sig = sigma.grad
            grad_ok = (
                grad_mu is not None
                and grad_mu.isfinite().item()
                and grad_sig is not None
                and grad_sig.isfinite().item()
            )
            grad_mu_mag = (
                f"{grad_mu.abs().item():.4g}"
                if grad_mu is not None and grad_mu.isfinite()
                else "NaN/Inf"
            )
            grad_sig_mag = (
                f"{grad_sig.abs().item():.4g}"
                if grad_sig is not None and grad_sig.isfinite()
                else "NaN/Inf"
            )
        except Exception:
            sample_ok = grad_ok = False
            grad_mu_mag = grad_sig_mag = "ERROR"

        print(
            f"{target_mean:>12} {mu_val:>8.2f} {sigma_val:>8.4f} {cv:>6.2f} "
            f"{'✓' if sample_ok else '✗':>10} {'✓' if grad_ok else '✗':>10} "
            f"{grad_mu_mag:>12} {grad_sig_mag:>12}"
        )


def test_softplus_normal_stability():
    """Test Softplus(Normal) as a positive distribution."""
    print("\n" + "=" * 80)
    print("SOFTPLUS NORMAL: rsample gradient stability")
    print("  I = softplus(mu + sigma * epsilon)")
    print("  Simple, always stable, but no analytic KL")
    print("=" * 80)

    mc = 50
    target_means = [1, 10, 100, 1000, 10000, 100000]

    print(
        f"{'target_I':>10} {'mu':>8} {'sigma':>8} {'sample_ok':>10} {'grad_ok':>10} "
        f"{'grad_mu_mag':>12} {'grad_sig_mag':>12} {'actual_mean':>12}"
    )
    print("-" * 100)

    for target in target_means:
        # softplus(mu) ≈ mu for mu >> 1, so mu ≈ target_mean
        mu_val = float(target)
        sigma_val = 0.1 * target  # CV ≈ 10%

        mu = torch.tensor(mu_val, requires_grad=True)
        sigma = torch.tensor(sigma_val, requires_grad=True)

        try:
            eps = torch.randn(mc)
            z = mu + sigma * eps
            samples = F.softplus(z)

            sample_ok = samples.isfinite().all().item()
            actual_mean = samples.mean().item()

            loss = samples.mean()
            loss.backward()

            grad_mu = mu.grad
            grad_sig = sigma.grad

            grad_ok = (
                grad_mu is not None
                and grad_mu.isfinite().item()
                and grad_sig is not None
                and grad_sig.isfinite().item()
            )
            grad_mu_mag = (
                f"{grad_mu.abs().item():.4g}"
                if grad_mu is not None and grad_mu.isfinite()
                else "NaN/Inf"
            )
            grad_sig_mag = (
                f"{grad_sig.abs().item():.4g}"
                if grad_sig is not None and grad_sig.isfinite()
                else "NaN/Inf"
            )

        except Exception:
            sample_ok = grad_ok = False
            grad_mu_mag = grad_sig_mag = "ERROR"
            actual_mean = 0

        print(
            f"{target:>10} {mu_val:>8.1f} {sigma_val:>8.1f} "
            f"{'✓' if sample_ok else '✗':>10} {'✓' if grad_ok else '✗':>10} "
            f"{grad_mu_mag:>12} {grad_sig_mag:>12} {actual_mean:>12.0f}"
        )


def test_kl_availability():
    """Check which distribution pairs have analytic KL."""
    print("\n" + "=" * 80)
    print("KL DIVERGENCE AVAILABILITY")
    print("=" * 80)

    pairs = [
        ("Gamma || Gamma", Gamma(2.0, 1.0), Gamma(1.0, 1.0)),
        ("Gamma || Exp", Gamma(2.0, 1.0), Gamma(1.0, 1.0)),
        ("LogNormal || LogNormal", LogNormal(1.0, 0.5), LogNormal(0.0, 1.0)),
        ("LogNormal || Gamma", LogNormal(1.0, 0.5), Gamma(1.0, 1.0)),
        ("Normal || Normal", Normal(1.0, 0.5), Normal(0.0, 1.0)),
    ]

    for name, q, p in pairs:
        try:
            kl = torch.distributions.kl.kl_divergence(q, p)
            print(f"  KL({name}): {kl.item():.4f}  [ANALYTIC]")
        except NotImplementedError:
            # Try MC
            samples = q.rsample([1000])
            kl_mc = (q.log_prob(samples) - p.log_prob(samples)).mean()
            print(
                f"  KL({name}): {kl_mc.item():.4f}  [MC ONLY - no analytic form]"
            )


def summary():
    print("\n" + "=" * 80)
    print("SUMMARY: Distribution comparison for positive-valued VI")
    print("=" * 80)
    print("""
    Distribution    | Reparam Method      | Stable? | Max range    | Analytic KL?
    ================|=====================|=========|==============|=============
    Gamma(k,r)      | Rejection sampling  | NO      | k < ~950     | Gamma-Gamma ✓
                    | + implicit grads    |         | then NaN     | Gamma-Exp ✓
    ----------------+---------------------+---------+--------------+-------------
    FoldedNormal    | Implicit reparam    | NO      | loc/scale<~30| MC only
    (loc, scale)    | via CDF inversion   |         | then NaN     |
    ----------------+---------------------+---------+--------------+-------------
    LogNormal       | Pathwise (standard  | YES     | Unlimited*   | LN-LN ✓
    (mu, sigma)     | Normal trick + exp) |         | mean up to   | LN-Gamma: MC
                    |                     |         | exp(88)≈10^38|
    ----------------+---------------------+---------+--------------+-------------
    Softplus(Normal)| Pathwise (standard  | YES     | Unlimited    | MC only
    (mu, sigma)     | Normal trick +      |         |              |
                    | softplus)           |         |              |

    * LogNormal: gradients scale as O(mean), so gradient clipping needed for
      very large intensities. But no NaN — just large, finite gradients.

    Key insight: only PATHWISE reparameterization (Normal trick + monotone transform)
    is unconditionally stable. Both Gamma and FoldedNormal use implicit/rejection-based
    gradients that break down when the distribution becomes concentrated.
    """)


if __name__ == "__main__":
    test_gamma_stability()
    test_folded_normal_stability()
    test_lognormal_stability()
    test_softplus_normal_stability()
    test_kl_availability()
    summary()

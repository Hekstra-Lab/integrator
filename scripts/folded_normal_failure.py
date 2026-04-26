"""
Pinpoint exactly where FoldedNormal gradients blow up.

The implicit reparameterization computes:
    dz/dmu = -dF/dmu / q(z)
    dz/dsigma = -dF/dsigma / q(z)

where q(z) = pdf(z) and dF/dmu = dCDF/dmu.

When loc >> scale, q(z) involves Normal(-loc, scale).pdf(z) + Normal(loc, scale).pdf(z).
For z near loc, the Normal(-loc, scale).pdf(z) term evaluates a normal at 2*loc/scale
standard deviations — this underflows to 0. But dF/dmu has a similar term. The ratio
dF/dmu / q can be 0/0 or tiny/tiny, producing NaN or huge values.

This test sweeps over batched scenarios to find the failure threshold.
"""

import sys

import torch

sys.path.insert(0, "src")
from integrator.model.distributions.folded_normal import FoldedNormal


def test_batched_rsample(
    loc_val, scale_val, batch_size=512, mc_samples=100, n_trials=20
):
    """Run multiple trials to find stochastic failures."""
    n_nan = 0
    max_grad = 0.0

    for trial in range(n_trials):
        loc = torch.full((batch_size,), loc_val, requires_grad=True)
        scale = torch.full((batch_size,), scale_val, requires_grad=True)

        dist = FoldedNormal(loc, scale)
        samples = dist.rsample([mc_samples])  # [mc, batch]

        if not samples.isfinite().all():
            n_nan += 1
            continue

        loss = samples.mean()
        loss.backward()

        if loc.grad is None or not loc.grad.isfinite().all():
            n_nan += 1
            continue

        grad_mag = loc.grad.abs().max().item()
        max_grad = max(max_grad, grad_mag)

    return n_nan, max_grad


print("=" * 90)
print(
    "FOLDED NORMAL: batched rsample gradient stability (512 batch, 100 MC, 20 trials)"
)
print("=" * 90)
print(
    f"{'loc':>8} {'scale':>8} {'loc/sc':>8} {'NaN_trials':>12} {'max_grad':>12} {'status':>8}"
)
print("-" * 70)

ratios_and_values = [
    # loc, scale
    (10, 1),
    (50, 5),
    (100, 10),
    (100, 5),
    (100, 2),
    (100, 1),
    (500, 50),
    (500, 10),
    (500, 5),
    (1000, 100),
    (1000, 50),
    (1000, 10),
    (1000, 5),
    (5000, 500),
    (5000, 100),
    (5000, 50),
    (10000, 1000),
    (10000, 100),
]

for loc_val, scale_val in ratios_and_values:
    n_nan, max_grad = test_batched_rsample(float(loc_val), float(scale_val))
    ratio = loc_val / scale_val
    status = (
        "✗ FAIL" if n_nan > 0 else ("⚠ WARN" if max_grad > 100 else "✓ OK")
    )
    print(
        f"{loc_val:>8} {scale_val:>8} {ratio:>8.0f} {n_nan:>10}/20 "
        f"{max_grad:>12.4g} {status:>8}"
    )


# Now check the internal quantities that blow up
print("\n" + "=" * 90)
print("INTERNAL DIAGNOSTICS: q(z) and dF/dmu at failure points")
print("=" * 90)

for loc_val, scale_val in [(100, 1), (1000, 10), (5000, 50)]:
    loc = torch.tensor(float(loc_val))
    scale = torch.tensor(float(scale_val))
    dist = FoldedNormal(loc, scale)

    # Sample
    samples = dist.sample([1000])
    q = dist.pdf(samples)
    dFdmu = dist.dcdfdmu(samples)
    dFdsig = dist.dcdfdsigma(samples)

    ratio_mu = (dFdmu / (q + 1e-4)).abs()
    ratio_sig = (dFdsig / (q + 1e-4)).abs()

    print(
        f"\nloc={loc_val}, scale={scale_val}, loc/scale={loc_val / scale_val:.0f}:"
    )
    print(
        f"  q(z):     min={q.min():.4e}, max={q.max():.4e}, "
        f"num_below_1e-4={int((q < 1e-4).sum())}/1000"
    )
    print(f"  dF/dmu:   min={dFdmu.min():.4e}, max={dFdmu.max():.4e}")
    print(f"  dF/dsig:  min={dFdsig.min():.4e}, max={dFdsig.max():.4e}")
    print(
        f"  |dF/dmu / (q+eps)|:  max={ratio_mu.max():.4e}, "
        f"num_above_10={int((ratio_mu > 10).sum())}"
    )
    print(
        f"  |dF/dsig / (q+eps)|: max={ratio_sig.max():.4e}, "
        f"num_above_10={int((ratio_sig > 10).sum())}"
    )

    # What's the Normal(-loc, scale).pdf(z) term?
    mirror_log_prob = torch.distributions.Normal(-loc, scale).log_prob(samples)
    print(
        f"  Normal(-loc,scale).log_prob(z): min={mirror_log_prob.min():.1f}, "
        f"max={mirror_log_prob.max():.1f}"
    )
    print(
        f"  (at loc/scale={loc_val / scale_val}, the mirror term evaluates Normal "
        f"at ~{2 * loc_val / scale_val:.0f} sigma away)"
    )

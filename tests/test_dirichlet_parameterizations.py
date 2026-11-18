"""
Comparison tests for Dirichlet concentration parameterizations.

Tests three parameterizations (A, C, LogClamp) for numerical stability, KL magnitude,
gradient health, and optimization convergence under the conditions that
cause NaN with the current approach (concentration=0.01, 441-dim Dirichlet).
"""

import math

import pytest
import torch
import torch.nn as nn
from torch.distributions import Dirichlet

from integrator.model.distributions.dirichlet import DirichletDistribution

# ---------------------------------------------------------------------------
# Parameterization modules (C and LogClamp are test-only)
# ---------------------------------------------------------------------------

IN_FEATURES = 64
SBOX_SHAPE = (21, 21)
N_PIXELS = 21 * 21  # 441


class _DirichletLogClamp(nn.Module):
    """Test-only parameterization: exp(clamp(linear(x), log(1e-3), log(1e3)))."""

    def __init__(self, in_features=IN_FEATURES, sbox_shape=SBOX_SHAPE):
        super().__init__()
        n_pixels = 1
        for s in sbox_shape:
            n_pixels *= s
        self.n_pixels = n_pixels
        self.alpha_layer = nn.Linear(in_features, n_pixels)
        self.min_log_alpha = math.log(1e-3)
        self.max_log_alpha = math.log(1e3)

    def forward(self, x):
        log_alpha = self.alpha_layer(x)
        log_alpha = torch.clamp(
            log_alpha, self.min_log_alpha, self.max_log_alpha
        )
        alpha = torch.exp(log_alpha)
        return Dirichlet(alpha)


class DirichletDistributionC(nn.Module):
    """Parameterization C: sigmoid_total * softmax(linear(x)), clamped."""

    def __init__(
        self,
        in_features=IN_FEATURES,
        sbox_shape=SBOX_SHAPE,
        alpha_min=0.05,
        alpha_max=30.0,
        total_min=10.0,
        total_max=200.0,
    ):
        super().__init__()
        n_pixels = 1
        for s in sbox_shape:
            n_pixels *= s
        self.n_pixels = n_pixels
        self.alpha_layer = nn.Linear(in_features, n_pixels, bias=False)
        self.total_layer = nn.Linear(in_features, 1, bias=True)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.total_min = total_min
        self.total_max = total_max

    def forward(self, x):
        logits = self.alpha_layer(x)
        pi = torch.softmax(logits, dim=-1)
        total_raw = self.total_layer(x)
        s = torch.sigmoid(total_raw)
        s = self.total_min + (self.total_max - self.total_min) * s
        alpha = s * pi
        alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        return Dirichlet(alpha)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PARAM_NAMES = ["A", "C", "LogClamp"]


def make_parameterization(name: str) -> nn.Module:
    if name == "A":
        return DirichletDistribution(
            in_features=IN_FEATURES, sbox_shape=SBOX_SHAPE
        )
    elif name == "C":
        return DirichletDistributionC()
    elif name == "LogClamp":
        return _DirichletLogClamp()
    else:
        raise ValueError(f"Unknown parameterization: {name}")


def kl_divergence_dirichlet(q: Dirichlet, p: Dirichlet) -> torch.Tensor:
    """KL(q || p) for Dirichlet distributions."""
    return torch.distributions.kl_divergence(q, p)


def make_prior(concentration: float = 0.01, n: int = N_PIXELS) -> Dirichlet:
    return Dirichlet(torch.full((n,), concentration))


# ---------------------------------------------------------------------------
# Test 1: Forward + backward stability
# ---------------------------------------------------------------------------


class TestForwardBackwardStability:
    """Pass random inputs through each parameterization, compute KL with
    Dirichlet(0.01) prior, and check for NaN/Inf."""

    @pytest.fixture(params=PARAM_NAMES)
    def param_name(self, request):
        return request.param

    def _run_stability(self, param_name, inputs):
        torch.manual_seed(42)
        model = make_parameterization(param_name)
        prior = make_prior(0.01)

        q = model(inputs)
        alphas = q.concentration

        kl = kl_divergence_dirichlet(q, prior).mean()
        kl.backward()

        # Collect gradient stats
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad)

        has_nan_alpha = torch.isnan(alphas).any().item()
        has_inf_alpha = torch.isinf(alphas).any().item()
        has_nan_kl = torch.isnan(kl).item()
        has_inf_kl = torch.isinf(kl).item()
        has_nan_grad = any(torch.isnan(g).any().item() for g in grads)
        has_inf_grad = any(torch.isinf(g).any().item() for g in grads)

        return {
            "has_nan_alpha": has_nan_alpha,
            "has_inf_alpha": has_inf_alpha,
            "has_nan_kl": has_nan_kl,
            "has_inf_kl": has_inf_kl,
            "has_nan_grad": has_nan_grad,
            "has_inf_grad": has_inf_grad,
            "kl": kl.item()
            if not (torch.isnan(kl) or torch.isinf(kl))
            else float("nan"),
        }

    def test_normal_inputs(self, param_name):
        """Standard encoder-like inputs."""
        torch.manual_seed(0)
        inputs = torch.randn(128, IN_FEATURES)
        result = self._run_stability(param_name, inputs)
        print(
            f"\n[{param_name}] normal inputs: KL={result['kl']:.2f}, "
            f"NaN_alpha={result['has_nan_alpha']}, NaN_KL={result['has_nan_kl']}, "
            f"NaN_grad={result['has_nan_grad']}"
        )
        # D should be fully stable; others may have issues
        if param_name == "A":
            assert not result["has_nan_alpha"], "A: NaN in alphas"
            assert not result["has_nan_kl"], "A: NaN in KL"
            assert not result["has_nan_grad"], "A: NaN in gradients"

    def test_extreme_inputs(self, param_name):
        """Large/small values simulating early vs late training."""
        torch.manual_seed(0)
        inputs = torch.randn(128, IN_FEATURES) * 10.0  # 10x scale
        result = self._run_stability(param_name, inputs)
        print(
            f"\n[{param_name}] extreme inputs (10x): KL={result['kl']:.2f}, "
            f"NaN_alpha={result['has_nan_alpha']}, NaN_KL={result['has_nan_kl']}, "
            f"NaN_grad={result['has_nan_grad']}"
        )
        if param_name == "A":
            assert not result["has_nan_alpha"], (
                "A: NaN in alphas with extreme inputs"
            )
            assert not result["has_nan_kl"], "A: NaN in KL with extreme inputs"
            assert not result["has_nan_grad"], (
                "A: NaN in gradients with extreme inputs"
            )


# ---------------------------------------------------------------------------
# Test 2: KL magnitude at initialization
# ---------------------------------------------------------------------------


class TestInitialKLMagnitude:
    """Measure KL(q || Dirichlet(0.01)) at default initialization."""

    def test_initial_kl_comparison(self):
        prior = make_prior(0.01)
        results = {}

        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            inputs = torch.zeros(1, IN_FEATURES)  # zero input = default bias
            q = model(inputs)
            kl = kl_divergence_dirichlet(q, prior).mean()
            kl_val = (
                kl.item()
                if not (torch.isnan(kl) or torch.isinf(kl))
                else float("inf")
            )
            results[name] = kl_val

        print("\n=== Initial KL(q || Dirichlet(0.01)) at default init ===")
        for name in PARAM_NAMES:
            print(f"  {name}: {results[name]:.4f}")
        best = min(results, key=results.get)
        print(f"  Best (lowest): {best}")

        # D should have a finite, reasonable KL
        assert math.isfinite(results["A"]), "A has non-finite initial KL"


# ---------------------------------------------------------------------------
# Test 3: Gradient health
# ---------------------------------------------------------------------------


class TestGradientHealth:
    """Check gradient magnitudes and dead gradients for each parameterization."""

    def test_gradient_health_comparison(self):
        prior = make_prior(0.01)
        results = {}

        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            inputs = torch.randn(128, IN_FEATURES)
            q = model(inputs)
            kl = kl_divergence_dirichlet(q, prior).mean()
            kl.backward()

            all_grads = []
            for p in model.parameters():
                if p.grad is not None:
                    all_grads.append(p.grad.flatten())

            if all_grads:
                grads = torch.cat(all_grads)
                max_grad = grads.abs().max().item()
                mean_grad = grads.abs().mean().item()
                pct_zero = (grads == 0).float().mean().item() * 100
            else:
                max_grad = mean_grad = 0.0
                pct_zero = 100.0

            results[name] = {
                "max_grad": max_grad,
                "mean_grad": mean_grad,
                "pct_zero": pct_zero,
                "explosion_risk": max_grad > 1e4,
                "stuck_risk": pct_zero > 50.0,
            }

        print("\n=== Gradient Health ===")
        print(
            f"  {'Name':<6} {'max|g|':>12} {'mean|g|':>12} {'%zero':>8} {'explode?':>10} {'stuck?':>8}"
        )
        for name in PARAM_NAMES:
            r = results[name]
            print(
                f"  {name:<6} {r['max_grad']:>12.4f} {r['mean_grad']:>12.6f} "
                f"{r['pct_zero']:>7.1f}% {'YES' if r['explosion_risk'] else 'no':>10} "
                f"{'YES' if r['stuck_risk'] else 'no':>8}"
            )

        # D should have no explosion or stuck risk
        assert not results["A"]["explosion_risk"], "A: gradient explosion risk"
        assert not results["A"]["stuck_risk"], "A: stuck gradient risk"


# ---------------------------------------------------------------------------
# Test 4: Optimization convergence
# ---------------------------------------------------------------------------


class TestOptimizationConvergence:
    """Optimize each parameterization to match a target Dirichlet."""

    def test_convergence_comparison(self):
        torch.manual_seed(42)
        n_steps = 200
        lr = 1e-3

        # Target: center-peaked profile
        target_alpha = torch.ones(N_PIXELS) * 0.1
        center = N_PIXELS // 2
        target_alpha[center] = 5.0
        # Also make neighbors higher
        h, w = SBOX_SHAPE
        cy, cx = h // 2, w // 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                idx = (cy + dy) * w + (cx + dx)
                if 0 <= idx < N_PIXELS:
                    dist = (dy**2 + dx**2) ** 0.5
                    target_alpha[idx] = max(
                        target_alpha[idx].item(), 5.0 * math.exp(-dist)
                    )
        target = Dirichlet(target_alpha)

        results = {}
        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            # Use a fixed input (as if encoder output is constant)
            fixed_input = torch.randn(1, IN_FEATURES)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            losses = []
            nan_step = None
            for step in range(n_steps):
                optimizer.zero_grad()
                q = model(fixed_input)
                kl = kl_divergence_dirichlet(q, target).mean()
                if torch.isnan(kl) or torch.isinf(kl):
                    nan_step = step
                    break
                kl.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                losses.append(kl.item())

            results[name] = {
                "final_kl": losses[-1] if losses else float("inf"),
                "nan_step": nan_step,
                "n_steps_completed": len(losses),
                "losses": losses,
            }

        print(
            "\n=== Optimization Convergence (200 steps, target=center-peaked) ==="
        )
        print(
            f"  {'Name':<6} {'Final KL':>12} {'NaN at step':>14} {'Steps done':>12}"
        )
        for name in PARAM_NAMES:
            r = results[name]
            nan_str = (
                str(r["nan_step"]) if r["nan_step"] is not None else "none"
            )
            print(
                f"  {name:<6} {r['final_kl']:>12.4f} {nan_str:>14} {r['n_steps_completed']:>12}"
            )

        # D should complete all steps without NaN
        assert results["A"]["nan_step"] is None, (
            f"A: NaN at step {results['A']['nan_step']}"
        )
        assert results["A"]["n_steps_completed"] == n_steps, (
            f"A: only completed {results['A']['n_steps_completed']}/{n_steps} steps"
        )


# ---------------------------------------------------------------------------
# Test 5: Optimization with sparse prior as regularizer
# ---------------------------------------------------------------------------


class TestSparsePriorOptimization:
    """Optimize to fit Poisson counts with Dirichlet(0.01) KL regularizer."""

    def test_sparse_prior_convergence(self):
        torch.manual_seed(42)
        n_steps = 500
        lr = 1e-3
        kl_weight = 0.1

        # Create realistic data
        h, w = SBOX_SHAPE
        true_profile = torch.zeros(N_PIXELS)
        cy, cx = h // 2, w // 2
        for y in range(h):
            for x in range(w):
                dist = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
                true_profile[y * w + x] = math.exp(-dist / 3.0)
        true_profile = true_profile / true_profile.sum()

        true_intensity = 100.0
        true_bg = 0.5
        expected = true_intensity * true_profile + true_bg
        counts = torch.poisson(
            expected.unsqueeze(0).expand(32, -1)
        )  # batch of 32

        prior = make_prior(0.01)

        results = {}
        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            fixed_input = torch.randn(1, IN_FEATURES).expand(32, -1)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            losses = []
            nan_step = None
            for step in range(n_steps):
                optimizer.zero_grad()
                q = model(fixed_input)

                # E[log p(counts | z_I * z_p + z_bg)] approximated via samples
                profile_sample = q.rsample()  # (32, 441)
                rate = true_intensity * profile_sample + true_bg
                log_lik = (
                    torch.distributions.Poisson(rate)
                    .log_prob(counts)
                    .sum(-1)
                    .mean()
                )

                kl = kl_divergence_dirichlet(q, prior).mean()

                loss = -log_lik + kl_weight * kl

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_step = step
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                losses.append(loss.item())

            results[name] = {
                "final_loss": losses[-1] if losses else float("inf"),
                "nan_step": nan_step,
                "n_steps_completed": len(losses),
            }

        print(
            "\n=== Sparse Prior Optimization (500 steps, Poisson likelihood + KL) ==="
        )
        print(
            f"  {'Name':<6} {'Final Loss':>12} {'NaN at step':>14} {'Steps done':>12}"
        )
        for name in PARAM_NAMES:
            r = results[name]
            nan_str = (
                str(r["nan_step"]) if r["nan_step"] is not None else "none"
            )
            print(
                f"  {name:<6} {r['final_loss']:>12.4f} {nan_str:>14} {r['n_steps_completed']:>12}"
            )

        # D should complete all steps
        assert results["A"]["nan_step"] is None, (
            f"A: NaN at step {results['A']['nan_step']}"
        )
        assert results["A"]["n_steps_completed"] == n_steps, (
            f"A: only completed {results['A']['n_steps_completed']}/{n_steps} steps"
        )


# ---------------------------------------------------------------------------
# Summary (runs as a standalone test)
# ---------------------------------------------------------------------------


class TestSummary:
    """Print a combined summary table ranking all parameterizations."""

    def test_print_summary(self):
        torch.manual_seed(42)
        prior = make_prior(0.01)
        summary = {name: {} for name in PARAM_NAMES}

        # --- Stability ---
        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            nan_count = 0
            for scale in [1.0, 5.0, 10.0]:
                inputs = torch.randn(128, IN_FEATURES) * scale
                try:
                    q = model(inputs)
                    kl = kl_divergence_dirichlet(q, prior).mean()
                    kl.backward()
                    if torch.isnan(kl) or torch.isinf(kl):
                        nan_count += 1
                except RuntimeError:
                    nan_count += 1
                model.zero_grad()
            summary[name]["nan_count"] = nan_count

        # --- Initial KL ---
        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            q = model(torch.zeros(1, IN_FEATURES))
            kl = kl_divergence_dirichlet(q, prior).mean()
            kl_val = kl.item() if math.isfinite(kl.item()) else float("inf")
            summary[name]["init_kl"] = kl_val

        # --- Gradient health ---
        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            q = model(torch.randn(128, IN_FEATURES))
            kl = kl_divergence_dirichlet(q, prior).mean()
            kl.backward()
            grads = torch.cat(
                [
                    p.grad.flatten()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
            summary[name]["max_grad"] = grads.abs().max().item()
            summary[name]["mean_grad"] = grads.abs().mean().item()
            summary[name]["pct_zero"] = (
                grads == 0
            ).float().mean().item() * 100

        # --- Convergence (quick version, 100 steps) ---
        target_alpha = torch.ones(N_PIXELS) * 0.5
        target_alpha[N_PIXELS // 2] = 5.0
        target = Dirichlet(target_alpha)
        for name in PARAM_NAMES:
            torch.manual_seed(42)
            model = make_parameterization(name)
            fixed_input = torch.randn(1, IN_FEATURES)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            steps_done = 0
            final_kl = float("inf")
            for step in range(100):
                optimizer.zero_grad()
                q = model(fixed_input)
                kl = kl_divergence_dirichlet(q, target).mean()
                if torch.isnan(kl) or torch.isinf(kl):
                    break
                kl.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                steps_done += 1
                final_kl = kl.item()
            summary[name]["conv_steps"] = steps_done
            summary[name]["conv_final_kl"] = final_kl

        # --- Print summary ---
        print("\n" + "=" * 80)
        print("DIRICHLET PARAMETERIZATION COMPARISON SUMMARY")
        print("=" * 80)
        print(
            f"  {'':>6} {'NaN/3':>7} {'Init KL':>12} {'max|g|':>12} {'mean|g|':>12} "
            f"{'%zero':>7} {'Conv steps':>11} {'Final KL':>12}"
        )
        print("-" * 80)
        for name in PARAM_NAMES:
            s = summary[name]
            print(
                f"  {name:>6} {s['nan_count']:>5}/3 {s['init_kl']:>12.2f} "
                f"{s['max_grad']:>12.4f} {s['mean_grad']:>12.6f} "
                f"{s['pct_zero']:>6.1f}% {s['conv_steps']:>9}/100 "
                f"{s['conv_final_kl']:>12.4f}"
            )
        print("=" * 80)

        # Just verify D is stable
        assert summary["A"]["nan_count"] == 0, (
            "A should have zero NaN occurrences"
        )

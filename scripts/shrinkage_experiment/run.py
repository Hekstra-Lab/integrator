"""Shrinkage bias experiment — main orchestrator.

Demonstrates that ELBO shrinkage bias depends on I_true per-reflection
and is non-multiplicative. Runs both analytical (Case 1, B=0) and
numerical (Case 2, B>0) experiments with direct optimization and
neural network encoder.

Usage:
    uv run python scripts/shrinkage_experiment/run.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from case1 import run_case1
from case2_direct import run_case2_direct
from case2_encoder import run_case2_encoder
from case3_full import run_case3_full
from plotting import make_linearity_plot, make_plots
from quadrature import batch_log_evidence

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes/dirichlet_profile/"
N_REFL = 1000
SEED = 42

PRIORS = [
    {"alpha0": 2.0, "beta0": 0.1, "label": "strong (μ₀=20, w≈0.091)"},
    {"alpha0": 2.0, "beta0": 0.02, "label": "weak (μ₀=100, w≈0.020)"},
]

# Case 2 direct optimization
MC_SAMPLES = 100
N_STEPS = 3000
LR_DIRECT = 0.01

# Case 2 encoder
N_EPOCHS = 500
BATCH_SIZE = 128
LR_ENCODER = 1e-3

# Quadrature
N_QUAD = 30000

OUTPUT_DIR = str(Path(__file__).parent)


def load_and_subsample(data_dir: str, n_refl: int, seed: int) -> dict:
    """Load data and subsample stratified by I_true."""
    print("[1] Loading data...")
    profiles_all = torch.load(f"{data_dir}/profiles.pt", weights_only=False)
    reference = torch.load(f"{data_dir}/reference.pt", weights_only=False)
    I_true_all = reference["intensity"]
    B_true_all = reference["background"]

    # Stratified subsample: sort by I_true, take evenly spaced
    sorted_idx = torch.argsort(I_true_all)
    step = len(I_true_all) // n_refl
    idx = sorted_idx[::step][:n_refl]

    profiles = profiles_all[idx]
    I_true = I_true_all[idx]
    B_true = B_true_all[idx]

    print(
        f"    {n_refl} reflections, I_true range: "
        f"[{I_true.min():.1f}, {I_true.max():.1f}], "
        f"B range: [{B_true.min():.3f}, {B_true.max():.3f}]"
    )
    return {"profiles": profiles, "I_true": I_true, "B_true": B_true}


def run_prior_experiment(
    data: dict,
    alpha0: float,
    beta0: float,
    label: str,
    seed: int,
) -> dict:
    """Run all experiments for a single prior setting."""
    mu0 = alpha0 / beta0
    w = beta0 / (beta0 + 1.0)
    profiles = data["profiles"]
    I_true = data["I_true"]
    B_true = data["B_true"]

    print(f"\n{'=' * 60}")
    print(f"Prior: Gamma({alpha0}, {beta0}), μ₀={mu0:.1f}, w={w:.4f}")
    print(f"{'=' * 60}")

    # ── Case 1: Analytical ──
    print("\n[2] Case 1: Analytical conjugate posterior (B=0)...")
    c1 = run_case1(profiles, I_true, alpha0, beta0, seed=seed)
    print(f"    Mean |bias| = {c1['bias'].abs().mean():.2f}")
    print(
        f"    Predicted mean |bias| = {c1['predicted_bias'].abs().mean():.4f}"
    )

    # ── Case 2 direct: Per-reflection optimization ──
    print(f"\n[3] Case 2 direct: ELBO optimization (B>0, {N_STEPS} steps)...")
    c2d = run_case2_direct(
        profiles,
        I_true,
        B_true,
        alpha0,
        beta0,
        mc_samples=MC_SAMPLES,
        n_steps=N_STEPS,
        lr=LR_DIRECT,
        seed=seed,
    )
    print(f"    Mean |bias| = {c2d['bias'].abs().mean():.2f}")

    # ── Case 2 encoder: Neural network ──
    # Use same counts as case2_direct for fair comparison
    print(f"\n[4] Case 2 encoder: Training CNN ({N_EPOCHS} epochs)...")
    c2e = run_case2_encoder(
        profiles,
        I_true,
        B_true,
        alpha0,
        beta0,
        counts=c2d["counts"],  # same counts!
        mc_samples=MC_SAMPLES,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR_ENCODER,
    )
    print(f"    Mean |bias| = {c2e['bias'].abs().mean():.2f}")

    # ── Case 3: Full model (unknown bg, profile, intensity) ──
    print(f"\n[4b] Case 3 full model: Training CNN ({N_EPOCHS} epochs)...")
    c3 = run_case3_full(
        profiles,
        I_true,
        B_true,
        alpha0,
        beta0,
        counts=c2d["counts"],  # same counts!
        mc_samples=MC_SAMPLES,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR_ENCODER,
    )
    print(f"    Mean |bias| = {c3['bias'].abs().mean():.2f}")

    # ── Quadrature: exact log-evidence ──
    print(f"\n[5] Quadrature: log p(X) for Case 1 ({N_QUAD} grid points)...")
    q_c1 = batch_log_evidence(
        c1["counts"],
        profiles,
        torch.zeros_like(B_true),
        alpha0,
        beta0,
        n_quad=N_QUAD,
    )
    kl_gap_c1 = q_c1 - c1["elbo"]
    print(f"    Case 1 KL gap (should be ~0): mean={kl_gap_c1.mean():.6f}")

    print(f"\n[6] Quadrature: log p(X) for Case 2 ({N_QUAD} grid points)...")
    q_c2 = batch_log_evidence(
        c2d["counts"],
        profiles,
        B_true,
        alpha0,
        beta0,
        n_quad=N_QUAD,
    )
    kl_gap_c2 = q_c2 - c2d["elbo"]
    print(f"    Case 2 KL gap: mean={kl_gap_c2.mean():.4f}")

    # Attach S to encoder/full results for shrinkage weight computation
    c2e["S"] = c2d["S"]
    c3["S"] = c2d["S"]

    return {
        "label": label,
        "alpha0": alpha0,
        "beta0": beta0,
        "I_true": I_true,
        "B_true": B_true,
        "case1": c1,
        "case2_direct": c2d,
        "case2_encoder": c2e,
        "case3_full": c3,
        "quadrature_case1": q_c1,
        "quadrature_case2": q_c2,
    }


def save_results(all_results: list[dict], output_dir: str) -> None:
    """Save per-reflection results as parquet files (one per prior)."""
    out = Path(output_dir)
    for r in all_results:
        alpha0 = r["alpha0"]
        beta0 = r["beta0"]
        mu0 = alpha0 / beta0
        w = beta0 / (beta0 + 1.0)
        I_true = r["I_true"]
        c1 = r["case1"]
        c2d = r["case2_direct"]
        c2e = r["case2_encoder"]
        q_c1 = r["quadrature_case1"]
        q_c2 = r["quadrature_case2"]

        c3 = r["case3_full"]

        B_true = r["B_true"]
        df = pl.DataFrame(
            {
                # Ground truth
                "I_true": I_true.numpy(),
                "B_true": B_true.numpy(),
                # Case 1
                "case1_S": c1["S"].numpy(),
                "case1_alpha": c1["alpha_star"].numpy(),
                "case1_beta": c1["beta_star"].numpy(),
                "case1_mu": c1["mu"].numpy(),
                "case1_bias": c1["bias"].numpy(),
                "case1_predicted_bias": c1["predicted_bias"].numpy(),
                "case1_elbo": c1["elbo"].numpy(),
                "case1_log_evidence": q_c1.numpy(),
                "case1_kl_gap": (q_c1 - c1["elbo"]).numpy(),
                # Case 2 direct
                "case2d_S": c2d["S"].numpy(),
                "case2d_alpha": c2d["alpha"].numpy(),
                "case2d_beta": c2d["beta"].numpy(),
                "case2d_mu": c2d["mu"].numpy(),
                "case2d_bias": c2d["bias"].numpy(),
                "case2d_elbo": c2d["elbo"].numpy(),
                "case2d_log_evidence": q_c2.numpy(),
                "case2d_kl_gap": (q_c2 - c2d["elbo"]).numpy(),
                # Case 2 encoder
                "case2e_alpha": c2e["alpha"].numpy(),
                "case2e_beta": c2e["beta"].numpy(),
                "case2e_mu": c2e["mu"].numpy(),
                "case2e_bias": c2e["bias"].numpy(),
                "case2e_elbo": c2e["elbo"].numpy(),
                "case2e_kl_gap": (q_c2 - c2e["elbo"]).numpy(),
                # Case 3 full
                "case3_alpha": c3["alpha"].numpy(),
                "case3_beta": c3["beta"].numpy(),
                "case3_mu": c3["mu"].numpy(),
                "case3_bias": c3["bias"].numpy(),
                "case3_elbo": c3["elbo"].numpy(),
                "case3_mu_B": c3["mu_B"].numpy(),
                "case3_bias_B": c3["bias_B"].numpy(),
                # Derived: effective shrinkage
                "w_case1": [w] * len(I_true),
                "mu0": [mu0] * len(I_true),
            }
        )

        # Add effective shrinkage weights
        denom_d = mu0 - c2d["S"].numpy()
        denom_e = mu0 - c2d["S"].numpy()
        df = df.with_columns(
            pl.when(pl.lit(np.abs(denom_d) > max(50.0, 0.05 * mu0)))
            .then(pl.col("case2d_bias") / pl.lit(denom_d))
            .otherwise(None)
            .alias("case2d_w_eff"),
            pl.when(pl.lit(np.abs(denom_e) > max(50.0, 0.05 * mu0)))
            .then(pl.col("case2e_bias") / pl.lit(denom_e))
            .otherwise(None)
            .alias("case2e_w_eff"),
        )

        tag = f"a{alpha0}_b{beta0}".replace(".", "p")
        path = out / f"results_{tag}.parquet"
        df.write_parquet(str(path))
        print(f"  Saved: {path} ({len(df)} rows)")


def main():
    print("=" * 60)
    print("SHRINKAGE BIAS EXPERIMENT")
    print(f"N_REFL={N_REFL}, MC_SAMPLES={MC_SAMPLES}")
    print(f"Direct: {N_STEPS} steps, Encoder: {N_EPOCHS} epochs")
    print("=" * 60)

    data = load_and_subsample(DATA_DIR, N_REFL, SEED)

    all_results = []
    for prior_cfg in PRIORS:
        result = run_prior_experiment(
            data,
            alpha0=prior_cfg["alpha0"],
            beta0=prior_cfg["beta0"],
            label=prior_cfg["label"],
            seed=SEED,
        )
        all_results.append(result)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        alpha0 = r["alpha0"]
        beta0 = r["beta0"]
        mu0 = alpha0 / beta0
        w = beta0 / (beta0 + 1.0)
        c1 = r["case1"]
        c2d = r["case2_direct"]
        c2e = r["case2_encoder"]
        q_c1 = r["quadrature_case1"]
        q_c2 = r["quadrature_case2"]

        c3 = r["case3_full"]

        print(f"\nPrior: Gamma({alpha0}, {beta0}), μ₀={mu0:.1f}, w={w:.4f}")
        print(
            f"  Case 1 mean |bias| (actual):    {c1['bias'].abs().mean():.2f}"
        )
        print(
            f"  Case 1 mean |bias| (predicted): {c1['predicted_bias'].abs().mean():.4f}"
        )
        print(
            f"  Case 2 direct mean |bias|:      {c2d['bias'].abs().mean():.2f}"
        )
        print(
            f"  Case 2 encoder mean |bias|:     {c2e['bias'].abs().mean():.2f}"
        )
        print(
            f"  Case 3 full mean |bias|:        {c3['bias'].abs().mean():.2f}"
        )
        print(
            f"  Case 1 KL gap (mean):           {(q_c1 - c1['elbo']).mean():.6f}"
        )
        print(
            f"  Case 2 direct KL gap (mean):    {(q_c2 - c2d['elbo']).mean():.4f}"
        )
        print(
            f"  Case 2 encoder KL gap (mean):   {(q_c2 - c2e['elbo']).mean():.4f}"
        )
        print(
            f"  Amortization gap (mean):        {(c2e['elbo'] - c2d['elbo']).mean():.4f}"
        )
        print(
            f"  Case 2 direct final loss:       {c2d['loss_history'][-1]:.1f}"
        )
        print(
            f"  Case 2 encoder final loss:      {c2e['loss_history'][-1]:.2f}"
        )
        print(
            f"  Case 3 full final loss:         {c3['loss_history'][-1]:.2f}"
        )

    # ── Save results ──
    print("\n[7] Saving results...")
    save_results(all_results, OUTPUT_DIR)

    # ── Plots ──
    print("\n[8] Generating plots...")
    make_plots(all_results, OUTPUT_DIR)
    make_linearity_plot(all_results, OUTPUT_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()

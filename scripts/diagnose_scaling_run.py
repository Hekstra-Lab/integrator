"""Diagnose a scaling model training run.

Extracts key model parameters from the last checkpoint and reports
table statistics, scale function behavior, loss parameters, and
learned dispersion.

Usage
-----
    python scripts/diagnose_scaling_run.py --run-dir <run_dir>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose a scaling model run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specific epoch to analyze (default: last checkpoint)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    config_path = meta["config"]
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb_info = meta["wandb"]
    log_dir = Path(wandb_info["log_dir"])

    checkpoints = sorted(log_dir.glob("**/epoch*.ckpt"))
    if not checkpoints:
        logger.error("No checkpoints found in %s", log_dir)
        return

    if args.epoch is not None:
        ckpt = [c for c in checkpoints if f"epoch={args.epoch:04d}" in c.name or f"epoch={args.epoch}" in c.name]
        if not ckpt:
            logger.error("No checkpoint found for epoch %d", args.epoch)
            return
        ckpt = ckpt[0]
    else:
        ckpt = checkpoints[-1]

    logger.info("=" * 60)
    logger.info("SCALING MODEL DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info("Run dir: %s", run_dir)
    logger.info("Checkpoint: %s", ckpt.name)

    state = torch.load(ckpt, weights_only=False, map_location="cpu")
    sd = state["state_dict"]

    # ── Config summary ──
    int_args = config["integrator"]["args"]
    loss_args = config["loss"]["args"]
    logger.info("")
    logger.info("── Config ──")
    logger.info("  integrator: %s", config["integrator"]["name"])
    logger.info("  loss: %s", config["loss"]["name"])
    logger.info("  n_hkl: %s", int_args.get("n_hkl"))
    logger.info("  scaling_amplitude: %s", int_args.get("scaling_amplitude", "gamma"))
    logger.info("  scaling_lr: %s", int_args.get("scaling_lr"))
    logger.info("  pi_weight: %s", loss_args.get("pi_weight"))
    logger.info("  scale_spatial: %s", int_args.get("scale_spatial", False))
    logger.info("  init_dispersion: %s", loss_args.get("init_dispersion", "None (Poisson)"))

    # ── Table type detection ──
    has_fano = "hkl_table.raw_fano.weight" in sd
    has_sigma = "hkl_table.raw_sigma.weight" in sd
    table_type = "gamma" if has_fano else ("amplitude" if has_sigma else "unknown")

    logger.info("")
    logger.info("── HKL Table (%s) ──", table_type)

    if table_type == "gamma":
        raw_mu = sd["hkl_table.raw_mu.weight"].squeeze(-1)
        raw_fano = sd["hkl_table.raw_fano.weight"].squeeze(-1)

        mu = torch.exp(raw_mu)
        fano = F.softplus(raw_fano) + 1e-6
        rate = 1.0 / fano
        k = mu * rate + 0.1

        I_mean = (k / rate).numpy()
        I_var = (k / rate.pow(2)).numpy()

        logger.info("  I = Gamma(k, rate)")
        logger.info("  k:    min=%.3f  median=%.3f  max=%.3f", k.min(), k.median(), k.max())
        logger.info("  rate: min=%.3f  median=%.3f  max=%.3f", rate.min(), rate.median(), rate.max())
        logger.info("  I_mean: min=%.3f  median=%.3f  max=%.1f", I_mean.min(), np.median(I_mean), I_mean.max())
        logger.info("  fano:   min=%.3f  median=%.3f  max=%.3f", fano.min(), fano.median(), fano.max())

    elif table_type == "amplitude":
        raw_mu = sd["hkl_table.raw_mu.weight"].squeeze(-1)
        raw_sigma = sd["hkl_table.raw_sigma.weight"].squeeze(-1)

        mu = raw_mu
        sigma = F.softplus(raw_sigma) + 1e-6

        I_mean = (mu.pow(2) + sigma.pow(2)).numpy()

        logger.info("  X ~ N(mu, sigma^2), F = |X|, F^2 = X^2")
        logger.info("  mu:    min=%.3f  median=%.3f  max=%.3f", mu.min(), mu.median(), mu.max())
        logger.info("  sigma: min=%.3f  median=%.3f  max=%.3f", sigma.min(), sigma.median(), sigma.max())
        logger.info("  mu/sigma (SNR): min=%.2f  median=%.2f  max=%.2f",
                     (mu / sigma).min(), (mu / sigma).median(), (mu / sigma).max())
        logger.info("  E[F^2]: min=%.3f  median=%.3f  max=%.1f", I_mean.min(), np.median(I_mean), I_mean.max())

    # ── Wilson G and B ──
    logger.info("")
    logger.info("── Wilson Parameters ──")
    if "loss.raw_G" in sd:
        G = F.softplus(sd["loss.raw_G"]).item()
        logger.info("  G (overall scale): %.4f", G)
    if "loss.raw_B" in sd:
        B = F.softplus(sd["loss.raw_B"]).item()
        b_min = loss_args.get("b_min", 0.0)
        logger.info("  B (B-factor): %.4f (+ b_min=%.1f = %.4f)", B, b_min, B + b_min)

    # ── Dispersion (NB) ──
    if "loss.raw_dispersion" in sd:
        r = F.softplus(sd["loss.raw_dispersion"]).item()
        logger.info("")
        logger.info("── Negative Binomial Dispersion ──")
        logger.info("  r (dispersion): %.4f", r)
        logger.info("  Interpretation: Var = mu + mu^2/%.1f", r)
        if r > 1000:
            logger.info("  -> Effectively Poisson (r very large)")
        elif r > 50:
            logger.info("  -> Mild overdispersion")
        elif r > 10:
            logger.info("  -> Moderate overdispersion")
        else:
            logger.info("  -> Heavy overdispersion - model sees significant extra variance")

    # ── Chebyshev Scale ──
    logger.info("")
    logger.info("── Scale Function ──")

    # Frame-only or spatial
    if "scale_fn.c" in sd:
        c = sd["scale_fn.c"]
        if c.dim() == 1:
            logger.info("  Type: frame-only Chebyshev (degree %d)", len(c) - 1)
            logger.info("  Coefficients: %s", c.numpy().round(4).tolist())
            s_const = F.softplus(c[0]).item()
            logger.info("  s(mid-frame) ≈ %.4f", s_const)
        elif c.dim() == 2:
            d_frame, d_radius = c.shape
            logger.info("  Type: spatial Chebyshev (frame deg %d * radius deg %d = %d params)",
                         d_frame - 1, d_radius - 1, c.numel())
            logger.info("  Coefficient norms per frame degree:")
            for i in range(d_frame):
                logger.info("    T_%d(frame): %s", i, c[i].numpy().round(4).tolist())

    # ── Learned concentration ──
    if "loss.log_alpha_per_group" in sd:
        alpha = F.softplus(sd["loss.log_alpha_per_group"])
        logger.info("")
        logger.info("── Learned Concentration (per group) ──")
        logger.info("  alpha: min=%.3f  median=%.3f  max=%.3f", alpha.min(), alpha.median(), alpha.max())

    # ── Parameter counts ──
    logger.info("")
    logger.info("── Parameter Counts ──")
    total = 0
    groups = {}
    for name, param in sd.items():
        if not name.endswith(".weight") and not name.endswith(".bias") and "raw" not in name:
            continue
        prefix = name.split(".")[0]
        n = param.numel()
        groups[prefix] = groups.get(prefix, 0) + n
        total += n
    for prefix, n in sorted(groups.items()):
        logger.info("  %s: %d", prefix, n)
    logger.info("  TOTAL: %d", total)

    logger.info("")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

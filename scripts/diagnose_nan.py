"""
Numerical investigation: why do NaN values appear during training?

This script traces through every stage of the forward pass and loss
computation, checking for NaN/Inf at each step and measuring the
magnitudes of loss components and gradients.

Run: python scripts/diagnose_nan.py
"""

import math
import os

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, Exponential, Gamma, Poisson

# ── 0.  Configuration (mirrors the YAML) ────────────────────────────────

DATA_DIR = "/n/hekstra_lab/people/aldama/integrator_data/simulated_data"
STATS_FILE = "stats_anscombe.pt"
COUNTS_FILE = "counts.pt"
MASKS_FILE = "masks.pt"
CONC_FILE = "concentration.pt"

MC_SAMPLES = 100
BATCH_SIZE = 512
H, W = 21, 21
N_PIXELS = H * W  # 441

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1.  Check the data ──────────────────────────────────────────────────
print("=" * 70)
print("STAGE 1: DATA INSPECTION")
print("=" * 70)

stats = torch.load(os.path.join(DATA_DIR, STATS_FILE), weights_only=False)
counts_all = torch.load(
    os.path.join(DATA_DIR, COUNTS_FILE), weights_only=False
).squeeze(-1)
masks_all = torch.load(
    os.path.join(DATA_DIR, MASKS_FILE), weights_only=False
).squeeze(-1)

print(f"\nstats type: {type(stats)}")
if isinstance(stats, (list, tuple)):
    for i, s in enumerate(stats):
        if isinstance(s, torch.Tensor):
            print(f"  stats[{i}]: shape={s.shape}, dtype={s.dtype}")
            print(
                f"    min={s.min().item():.6g}, max={s.max().item():.6g}, "
                f"mean={s.mean().item():.6g}"
            )
            print(
                f"    has_nan={s.isnan().any().item()}, has_inf={s.isinf().any().item()}"
            )
            print(
                f"    num_zeros={(s == 0).sum().item()}, num_negative={(s < 0).sum().item()}"
            )
        else:
            print(f"  stats[{i}]: {s}")
elif isinstance(stats, torch.Tensor):
    print(f"  stats: shape={stats.shape}, dtype={stats.dtype}")
    print(f"    min={stats.min().item():.6g}, max={stats.max().item():.6g}")
    print(
        f"    has_nan={stats.isnan().any().item()}, has_inf={stats.isinf().any().item()}"
    )

print(f"\ncounts: shape={counts_all.shape}, dtype={counts_all.dtype}")
print(
    f"  min={counts_all.min().item():.6g}, max={counts_all.max().item():.6g}"
)
print(
    f"  has_nan={counts_all.isnan().any().item()}, has_inf={counts_all.isinf().any().item()}"
)
print(f"  num_zeros={(counts_all == 0).sum().item()} / {counts_all.numel()}")

print(f"\nmasks: shape={masks_all.shape}, dtype={masks_all.dtype}")

# ── 2.  Check standardization ───────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 2: STANDARDIZATION CHECK")
print("=" * 70)

# NOTE: SimulatedShoeboxLoader has self.anscombe = False HARDCODED (line 610)
# So even though your YAML says anscombe: true, the raw-count normalization
# is used with the anscombe stats file. Let's check both paths.

if isinstance(stats, (list, tuple)):
    stat_0, stat_1 = stats[0], stats[1]
else:
    stat_0, stat_1 = stats[0], stats[1]

print(
    f"\nstat_0 (mean): shape={stat_0.shape if isinstance(stat_0, torch.Tensor) else 'scalar'}"
)
print(
    f"stat_1 (var):  shape={stat_1.shape if isinstance(stat_1, torch.Tensor) else 'scalar'}"
)

if isinstance(stat_1, torch.Tensor):
    sqrt_var = stat_1.sqrt()
    print("\nsqrt(stat_1):")
    print(
        f"  min={sqrt_var.min().item():.6g}, max={sqrt_var.max().item():.6g}"
    )
    print(f"  num_zeros={(sqrt_var == 0).sum().item()}")
    print(f"  num_near_zero (< 1e-6)={(sqrt_var < 1e-6).sum().item()}")

    if (sqrt_var == 0).any():
        print(
            "\n  *** CRITICAL: sqrt(variance) has ZEROS → division will produce Inf! ***"
        )
        zero_positions = (sqrt_var == 0).nonzero(as_tuple=True)
        print(f"  Zero positions: {zero_positions}")

# Reproduce the ACTUAL standardization (SimulatedShoeboxLoader, anscombe=False)
batch_counts = counts_all[:BATCH_SIZE]
batch_masks = masks_all[:BATCH_SIZE]

standardized = ((batch_counts * batch_masks) - stat_0) / stat_1.sqrt()
print(f"\nStandardized counts (non-anscombe path, batch of {BATCH_SIZE}):")
print(
    f"  min={standardized.min().item():.6g}, max={standardized.max().item():.6g}"
)
print(
    f"  mean={standardized.mean().item():.6g}, std={standardized.std().item():.6g}"
)
print(
    f"  has_nan={standardized.isnan().any().item()}, has_inf={standardized.isinf().any().item()}"
)
n_inf = standardized.isinf().sum().item()
n_nan = standardized.isnan().sum().item()
if n_inf > 0 or n_nan > 0:
    print(
        f"\n  *** CRITICAL: standardized_counts has {n_inf} Inf and {n_nan} NaN values! ***"
    )
    print(
        "  *** These propagate through encoder → surrogate → loss → NaN everywhere ***"
    )

# Also check the anscombe path (what SHOULD be used if anscombe: true worked)
ans = 2 * torch.sqrt(batch_counts + 3.0 / 8.0)
standardized_anscombe = (ans - stat_1) / stat_1.sqrt()
print("\nStandardized counts (anscombe path, for comparison):")
print(
    f"  min={standardized_anscombe.min().item():.6g}, max={standardized_anscombe.max().item():.6g}"
)
print(
    f"  has_nan={standardized_anscombe.isnan().any().item()}, has_inf={standardized_anscombe.isinf().any().item()}"
)

# ── 3.  Dirichlet prior check ───────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: DIRICHLET PRIOR ANALYSIS")
print("=" * 70)

conc_file = os.path.join(DATA_DIR, CONC_FILE)
if os.path.exists(conc_file):
    loaded = torch.load(conc_file, weights_only=False)
    print(
        f"\nconcentration.pt: shape={loaded.shape}, sum={loaded.sum().item():.6g}"
    )
    print(
        f"  before processing: min={loaded.min().item():.6g}, max={loaded.max().item():.6g}"
    )

    # Reproduce the processing in _get_dirichlet_prior
    loaded_proc = loaded.clone()
    loaded_proc[loaded_proc > 2] *= 40
    loaded_proc /= loaded_proc.sum()
    alpha_p = loaded_proc.reshape(-1)

    print(f"  after processing: sum={alpha_p.sum().item():.6g}")
    print(f"    min={alpha_p.min().item():.10g}")
    print(f"    max={alpha_p.max().item():.6g}")
    print(f"    mean={alpha_p.mean().item():.6g}")
    print(f"    num_components={alpha_p.numel()}")
    print(f"    num < 0.01: {(alpha_p < 0.01).sum().item()}")
    print(f"    num < 0.001: {(alpha_p < 0.001).sum().item()}")

    # What does digamma/lgamma look like for these?
    print(
        f"\n  lgamma(alpha_p): min={torch.lgamma(alpha_p).min().item():.4f}, max={torch.lgamma(alpha_p).max().item():.4f}"
    )
    print(
        f"  digamma(alpha_p): min={torch.digamma(alpha_p).min().item():.4f}, max={torch.digamma(alpha_p).max().item():.4f}"
    )

    # Compute KL at initialization: alpha_q ≈ softplus(0) + eps ≈ 0.693
    alpha_q_init = torch.full_like(alpha_p, 0.693)
    q_dir = Dirichlet(alpha_q_init)
    p_dir = Dirichlet(alpha_p)
    kl_dir = torch.distributions.kl.kl_divergence(q_dir, p_dir)
    print("\n  KL(Dirichlet(0.693) || Dirichlet(prior)) at initialization:")
    print(f"    KL = {kl_dir.item():.2f}")
    print(f"    KL * weight(0.005) = {kl_dir.item() * 0.005:.4f}")
    print(
        f"    is_nan={kl_dir.isnan().item()}, is_inf={kl_dir.isinf().item()}"
    )

    # Check: what KL values arise as alpha_q gets smaller?
    print("\n  KL sensitivity to alpha_q (all components equal):")
    for aq_val in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5, 1e-6]:
        alpha_q_test = torch.full_like(alpha_p, aq_val)
        try:
            kl_test = torch.distributions.kl.kl_divergence(
                Dirichlet(alpha_q_test), p_dir
            )
            print(
                f"    alpha_q={aq_val:.1e}: KL={kl_test.item():.2f}, "
                f"nan={kl_test.isnan().item()}, inf={kl_test.isinf().item()}"
            )
        except Exception as e:
            print(f"    alpha_q={aq_val:.1e}: ERROR: {e}")
else:
    print(f"\n  concentration.pt not found at {conc_file}")

# ── 4.  Gamma/Exponential KL check ──────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 4: GAMMA KL vs EXPONENTIAL(0.001) PRIOR")
print("=" * 70)

p_exp = Gamma(torch.tensor(1.0), torch.tensor(0.001))  # Exponential(0.001)

print("\nKL(Gamma(k, r) || Exponential(0.001)) for various k, r:")
print(f"{'k':>10} {'r':>10} {'mean':>12} {'KL':>14} {'nan':>5} {'inf':>5}")
print("-" * 60)
for k_val in [0.01, 0.1, 0.5, 1.0, 5.0, 50.0, 500.0]:
    for r_val in [1e-6, 1e-4, 0.01, 0.1, 1.0, 10.0]:
        k_t = torch.tensor(k_val)
        r_t = torch.tensor(r_val)
        q_g = Gamma(k_t, r_t)
        kl_g = torch.distributions.kl.kl_divergence(q_g, p_exp)
        mean_g = k_val / r_val
        print(
            f"{k_val:10.4f} {r_val:10.6f} {mean_g:12.2f} {kl_g.item():14.2f} "
            f"{kl_g.isnan().item():>5} {kl_g.isinf().item():>5}"
        )

# ── 5.  Poisson log-prob check with extreme rates ────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: POISSON LOG-PROB WITH EXTREME RATES")
print("=" * 70)

print("\nPoisson(rate).log_prob(counts=50) for various rates:")
for rate_val in [
    1e-6,
    0.01,
    1.0,
    50.0,
    1e3,
    1e6,
    1e9,
    1e20,
    1e38,
    float("inf"),
]:
    rate_t = torch.tensor(rate_val)
    try:
        ll = Poisson(rate_t).log_prob(torch.tensor(50.0))
        print(
            f"  rate={rate_val:.1e}: log_prob={ll.item():.4g}, "
            f"nan={ll.isnan().item()}, inf={ll.isinf().item()}"
        )
    except Exception as e:
        print(f"  rate={rate_val:.1e}: ERROR: {e}")

# ── 6.  Gamma rsample overflow check ────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 6: GAMMA RSAMPLE OVERFLOW CHECK")
print("=" * 70)

print("\nGamma(k, r).rsample() statistics (1000 samples each):")
print(
    f"{'k':>10} {'r':>10} {'mean':>12} {'max_sample':>14} {'has_inf':>8} {'has_nan':>8}"
)
print("-" * 70)
for k_val in [0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]:
    for r_val in [1e-6, 1e-4, 0.01, 1.0]:
        k_t = torch.tensor(k_val)
        r_t = torch.tensor(r_val)
        try:
            samples = Gamma(k_t, r_t).rsample([1000])
            print(
                f"{k_val:10.4f} {r_val:10.6f} {k_val / r_val:12.2f} "
                f"{samples.max().item():14.4g} "
                f"{samples.isinf().any().item():>8} "
                f"{samples.isnan().any().item():>8}"
            )
        except Exception as e:
            print(f"{k_val:10.4f} {r_val:10.6f} {'ERROR':>12}: {e}")

# ── 7.  MC KL estimator check (FoldedNormal → Exponential) ──────────────
print("\n" + "=" * 70)
print("STAGE 7: MC KL ESTIMATOR (FoldedNormal → Exponential)")
print("=" * 70)

from integrator.model.distributions.folded_normal import FoldedNormal

p_exp_dist = Exponential(torch.tensor(0.001))
print(
    "\nKL(FoldedNormal(loc, scale) || Exponential(0.001)) via MC (100 samples):"
)
print(f"{'loc':>12} {'scale':>8} {'KL':>14} {'nan':>5} {'inf':>5}")
print("-" * 50)
for loc_val in [0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6, 1e10]:
    for scale_val in [0.1, 1.0, 10.0]:
        loc_t = torch.tensor(loc_val)
        scale_t = torch.tensor(scale_val)
        fn = FoldedNormal(loc_t, scale_t)
        try:
            samples = fn.rsample(torch.Size([100]))
            log_q = fn.log_prob(samples)
            log_p = p_exp_dist.log_prob(samples)
            kl_mc = (log_q - log_p).mean()
            print(
                f"{loc_val:12.1e} {scale_val:8.1f} {kl_mc.item():14.4g} "
                f"{kl_mc.isnan().item():>5} {kl_mc.isinf().item():>5}"
            )
        except Exception as e:
            print(f"{loc_val:12.1e} {scale_val:8.1f} {'ERROR':>14}: {e}")

# ── 8.  FoldedNormal exp(raw_loc) overflow ───────────────────────────────
print("\n" + "=" * 70)
print("STAGE 8: exp(raw_loc) OVERFLOW IN FoldedNormalA")
print("=" * 70)

print("\nfloat32 exp overflow threshold: raw_loc ≈ 88.7")
print(f"{'raw_loc':>10} {'exp(raw_loc)':>14} {'is_inf':>7}")
print("-" * 35)
for raw_val in [0, 10, 20, 50, 80, 85, 87, 88, 88.7, 89, 90, 100]:
    val = torch.tensor(raw_val, dtype=torch.float32).exp()
    print(f"{raw_val:10.1f} {val.item():14.4g} {val.isinf().item():>7}")

print("\nWith encoder_out=64 and weights ~U(-0.125, 0.125):")
print("  Max initial linear output: ~64 * 0.125 * max_encoder_out")
print("  After ReLU, encoder outputs are in [0, ~3]")
print("  Initial raw_loc range: ~[-0.4, 0.4]")
print(f"  exp(0.4) = {math.exp(0.4):.3f}")
print("\nWith trained weights reaching ±1.5:")
print("  raw_loc could reach: 64 * 1.5 * 2 = 192 → exp(192) = Inf")
print("  raw_loc = 89 (much less extreme) is already enough for Inf")

# ── 9.  Full forward+loss simulation at init ─────────────────────────────
print("\n" + "=" * 70)
print("STAGE 9: SIMULATED FORWARD+LOSS AT INITIALIZATION")
print("=" * 70)

torch.manual_seed(42)

# Simulate encoder output (64-dim, after ReLU, typical initialization)
x_enc = torch.randn(BATCH_SIZE, 64).relu().to(DEVICE)
print(f"\nSimulated encoder output: shape={x_enc.shape}")
print(
    f"  min={x_enc.min().item():.4f}, max={x_enc.max().item():.4f}, mean={x_enc.mean().item():.4f}"
)

# GammaA surrogate (qi)
linear_k = torch.nn.Linear(64, 1).to(DEVICE)
linear_r = torch.nn.Linear(64, 1).to(DEVICE)
eps = 1e-6

raw_k = linear_k(x_enc)
k = F.softplus(raw_k) + eps
raw_r = linear_r(x_enc)
r = F.softplus(raw_r) + eps

print("\nGammaA qi at init:")
print(
    f"  k: min={k.min().item():.6f}, max={k.max().item():.6f}, mean={k.mean().item():.6f}"
)
print(
    f"  r: min={r.min().item():.6f}, max={r.max().item():.6f}, mean={r.mean().item():.6f}"
)
print(f"  mean(k/r) = {(k / r).mean().item():.4f}")

qi = Gamma(k.flatten(), r.flatten())
zI = qi.rsample([MC_SAMPLES]).unsqueeze(-1).permute(1, 0, 2)
print(
    f"  zI: shape={zI.shape}, min={zI.min().item():.6g}, max={zI.max().item():.6g}"
)
print(
    f"  zI has_nan={zI.isnan().any().item()}, has_inf={zI.isinf().any().item()}"
)

# Dirichlet surrogate (qp)
alpha_layer = torch.nn.Linear(64, N_PIXELS).to(DEVICE)
alpha_raw = alpha_layer(x_enc)
alpha = F.softplus(alpha_raw) + eps
print("\nDirichlet qp at init:")
print(f"  alpha: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}")
print(f"  alpha sum per sample: mean={alpha.sum(-1).mean().item():.2f}")

qp = Dirichlet(alpha)
zp = qp.rsample([MC_SAMPLES]).permute(1, 0, 2)
print(
    f"  zp: shape={zp.shape}, min={zp.min().item():.6g}, max={zp.max().item():.6g}"
)
print(
    f"  zp has_nan={zp.isnan().any().item()}, has_inf={zp.isinf().any().item()}"
)
print(f"  zp has_zeros={(zp == 0).sum().item()} / {zp.numel()}")

# GammaA surrogate (qbg)
linear_k2 = torch.nn.Linear(64, 1).to(DEVICE)
linear_r2 = torch.nn.Linear(64, 1).to(DEVICE)
raw_k2 = linear_k2(x_enc)
k2 = F.softplus(raw_k2) + eps
raw_r2 = linear_r2(x_enc)
r2 = F.softplus(raw_r2) + eps
qbg = Gamma(k2.flatten(), r2.flatten())
zbg = qbg.rsample([MC_SAMPLES]).unsqueeze(-1).permute(1, 0, 2)
print("\nGammaA qbg at init:")
print(f"  zbg: min={zbg.min().item():.6g}, max={zbg.max().item():.6g}")

# Rate
rate = zI * zp + zbg
print("\nRate = zI * zp + zbg:")
print(f"  min={rate.min().item():.6g}, max={rate.max().item():.6g}")
print(
    f"  has_nan={rate.isnan().any().item()}, has_inf={rate.isinf().any().item()}"
)

# Poisson log-prob
fake_counts = batch_counts[:BATCH_SIZE].to(DEVICE)
ll = Poisson(rate + eps).log_prob(fake_counts.unsqueeze(1))
print("\nPoisson log-prob:")
print(f"  min={ll.min().item():.6g}, max={ll.max().item():.6g}")
print(
    f"  has_nan={ll.isnan().any().item()}, has_inf={ll.isinf().any().item()}"
)

neg_ll = (-torch.mean(ll, dim=1)).sum(1)
print(
    f"  neg_ll per sample: mean={neg_ll.mean().item():.2f}, max={neg_ll.max().item():.2f}"
)

# KL terms
if os.path.exists(conc_file):
    loaded_kl = torch.load(conc_file, weights_only=False)
    loaded_kl[loaded_kl > 2] *= 40
    loaded_kl /= loaded_kl.sum()
    p_dir_kl = Dirichlet(loaded_kl.reshape(-1).to(DEVICE))

    kl_prf_vals = torch.distributions.kl.kl_divergence(qp, p_dir_kl)
    print("\nKL(qp || p_dirichlet):")
    print(
        f"  per sample: mean={kl_prf_vals.mean().item():.2f}, max={kl_prf_vals.max().item():.2f}"
    )
    print(
        f"  has_nan={kl_prf_vals.isnan().any().item()}, has_inf={kl_prf_vals.isinf().any().item()}"
    )
    print(f"  * weight(0.005) = {kl_prf_vals.mean().item() * 0.005:.4f}")

p_i = Gamma(torch.tensor(1.0).to(DEVICE), torch.tensor(0.001).to(DEVICE))
kl_i_vals = torch.distributions.kl.kl_divergence(qi, p_i)
print("\nKL(qi || Exponential(0.001)):")
print(
    f"  per sample: mean={kl_i_vals.mean().item():.4f}, max={kl_i_vals.max().item():.4f}"
)
print(
    f"  has_nan={kl_i_vals.isnan().any().item()}, has_inf={kl_i_vals.isinf().any().item()}"
)
print(f"  * weight(0.5) = {kl_i_vals.mean().item() * 0.5:.4f}")

kl_bg_vals = torch.distributions.kl.kl_divergence(qbg, p_i)
print("\nKL(qbg || Exponential(0.001)):")
print(
    f"  per sample: mean={kl_bg_vals.mean().item():.4f}, max={kl_bg_vals.max().item():.4f}"
)

# Total loss
total_loss = neg_ll.mean() + kl_i_vals.mean() * 0.5 + kl_bg_vals.mean() * 0.5
if os.path.exists(conc_file):
    total_loss += kl_prf_vals.mean() * 0.005
print(f"\nTotal loss at initialization: {total_loss.item():.2f}")
print(
    f"  Breakdown: neg_ll={neg_ll.mean().item():.2f}, "
    f"kl_i={kl_i_vals.mean().item() * 0.5:.4f}, "
    f"kl_bg={kl_bg_vals.mean().item() * 0.5:.4f}, "
    f"kl_prf={kl_prf_vals.mean().item() * 0.005:.4f}"
    if os.path.exists(conc_file)
    else ""
)

# ── 10.  Gradient check ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 10: GRADIENT MAGNITUDE CHECK")
print("=" * 70)

total_loss.backward()

for name, param in [
    ("linear_k.weight", linear_k.weight),
    ("linear_k.bias", linear_k.bias),
    ("linear_r.weight", linear_r.weight),
    ("linear_r.bias", linear_r.bias),
    ("alpha_layer.weight", alpha_layer.weight),
    ("alpha_layer.bias", alpha_layer.bias),
]:
    if param.grad is not None:
        g = param.grad
        print(
            f"  {name:25s}: norm={g.norm().item():.4g}, "
            f"max_abs={g.abs().max().item():.4g}, "
            f"nan={g.isnan().any().item()}, inf={g.isinf().any().item()}"
        )

# ── 11.  SimulatedShoeboxLoader anscombe bug check ───────────────────────
print("\n" + "=" * 70)
print("STAGE 11: SimulatedShoeboxLoader ANSCOMBE BUG")
print("=" * 70)
print("""
SimulatedShoeboxLoader.__init__ (line 610) hardcodes:
    self.anscombe = False

This means even though your YAML says 'anscombe: true', the loader
IGNORES it and uses the NON-anscombe standardization formula:
    standardized_counts = ((counts * masks) - stats[0]) / stats[1].sqrt()

But you're loading 'stats_anscombe.pt', which contains Anscombe-transformed
statistics. The mismatch between raw counts and Anscombe stats may produce
incorrectly scaled inputs.

If stats[1] (variance) has any zeros → division by zero → Inf → NaN cascade.
""")

# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

issues = []

if isinstance(stat_1, torch.Tensor) and (stat_1.sqrt() == 0).any():
    issues.append(
        "CRITICAL: stats variance has zeros → Inf in standardized counts"
    )

if isinstance(stat_1, torch.Tensor) and (stat_1.sqrt() < 1e-6).any():
    issues.append(
        "WARNING: stats variance has near-zero values → very large standardized counts"
    )

if standardized.isinf().any():
    issues.append("CRITICAL: standardized_counts contains Inf values")

if standardized.isnan().any():
    issues.append("CRITICAL: standardized_counts contains NaN values")

if os.path.exists(conc_file) and kl_prf_vals.isnan().any():
    issues.append("CRITICAL: Dirichlet KL produces NaN")

if os.path.exists(conc_file) and kl_prf_vals.isinf().any():
    issues.append("CRITICAL: Dirichlet KL produces Inf")

if kl_i_vals.isnan().any():
    issues.append("CRITICAL: Gamma KL produces NaN")

for name, param in [
    ("linear_k", linear_k),
    ("linear_r", linear_r),
    ("alpha_layer", alpha_layer),
]:
    for pname, p in param.named_parameters():
        if p.grad is not None and p.grad.isnan().any():
            issues.append(f"CRITICAL: {name}.{pname} gradient is NaN")

if not issues:
    issues.append("No immediate NaN/Inf found at initialization.")
    issues.append(
        "The NaN likely develops during training as parameters drift."
    )
    issues.append(
        "Run with torch.autograd.set_detect_anomaly(True) to find the exact step."
    )

print()
for issue in issues:
    print(f"  • {issue}")
print()

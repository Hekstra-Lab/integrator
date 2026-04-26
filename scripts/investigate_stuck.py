"""
Investigate: why does training get stuck when all prior weights = 1.0?

Compares three settings:
  A) weights = (0.005, 0.5, 0.5) — current working config
  B) weights = (1.0, 1.0, 1.0)  — proper ELBO
  C) weights = (1.0, 1.0, 1.0) + warm-up schedule (KL annealing)

Traces loss components, gradient flow per module, and distribution params.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Gamma, Poisson

torch.manual_seed(42)

DATA_DIR = (
    "/Users/luis/master/notebooks/integrator_notes/code/simulating_shoeboxes"
)
BATCH_SIZE = 128
MC_SAMPLES = 10
H, W = 21, 21
N_PIXELS = H * W
LR = 0.001
GRAD_CLIP = 1.0
N_STEPS = 1500

# ── Load data ────────────────────────────────────────────────────────────
counts_all = torch.load(f"{DATA_DIR}/counts.pt", weights_only=False).squeeze(
    -1
)
masks_all = torch.load(f"{DATA_DIR}/masks.pt", weights_only=False).squeeze(-1)
stats = torch.load(f"{DATA_DIR}/stats_anscombe.pt", weights_only=False)
conc_raw = torch.load(f"{DATA_DIR}/concentration.pt", weights_only=False)

# Correct anscombe standardization
ans = 2 * torch.sqrt(counts_all + 3.0 / 8.0)
standardized = (ans - stats[0]) / stats[1].sqrt()

# Dirichlet prior (scaled to sum = N_PIXELS)
conc_prior = conc_raw.clone()
conc_prior[conc_prior > 2] *= 40
n_components = conc_prior.numel()
conc_prior = conc_prior / conc_prior.sum() * n_components
alpha_prior = conc_prior.reshape(-1)

print(
    f"Data: {counts_all.shape[0]} samples, counts range [{counts_all.min():.0f}, {counts_all.max():.0f}]"
)
print(
    f"Standardized: range [{standardized.min():.2f}, {standardized.max():.2f}], std={standardized.std():.2f}"
)
print(
    f"Dirichlet prior: sum={alpha_prior.sum():.1f}, min={alpha_prior.min():.4f}, max={alpha_prior.max():.2f}"
)


class SimpleEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 16)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)
        self.norm2 = nn.GroupNorm(4, 32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm3 = nn.GroupNorm(8, 64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        return F.relu(self.fc(x))


class GammaHead(nn.Module):
    def __init__(self, in_features=64, eps=1e-6):
        super().__init__()
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(self, x):
        k = F.softplus(self.linear_k(x)) + self.eps
        r = F.softplus(self.linear_r(x)) + self.eps
        return Gamma(k.flatten(), r.flatten())


class DirichletHead(nn.Module):
    def __init__(self, in_features=64, n_pixels=441, eps=1e-6):
        super().__init__()
        self.alpha_layer = nn.Linear(in_features, n_pixels)
        self.eps = eps

    def forward(self, x):
        return Dirichlet(F.softplus(self.alpha_layer(x)) + self.eps)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_profile = SimpleEncoder(64)
        self.enc_intensity = SimpleEncoder(64)
        self.qp_head = DirichletHead()
        self.qi_head = GammaHead()
        self.qbg_head = GammaHead()

    def forward(self, shoebox):
        b = shoebox.shape[0]
        x = shoebox.reshape(b, 1, H, W)
        x_profile = self.enc_profile(x)
        x_intensity = self.enc_intensity(x)
        return (
            self.qp_head(x_profile),
            self.qi_head(x_intensity),
            self.qbg_head(x_intensity),
        )


def compute_loss(
    qp,
    qi,
    qbg,
    counts,
    mask,
    alpha_prior,
    w_prf=1.0,
    w_i=1.0,
    w_bg=1.0,
    mc_samples=10,
    eps=1e-6,
):
    zI = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zp = qp.rsample([mc_samples]).permute(1, 0, 2)
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    rate = zI * zp + zbg

    ll = Poisson(rate + eps).log_prob(counts.unsqueeze(1))
    ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
    neg_ll = (-ll_mean).sum(1)

    p_dir = Dirichlet(alpha_prior)
    kl_prf = torch.distributions.kl.kl_divergence(qp, p_dir)
    p_i = Gamma(torch.tensor(1.0), torch.tensor(0.001))
    kl_i = torch.distributions.kl.kl_divergence(qi, p_i)
    p_bg = Gamma(torch.tensor(1.0), torch.tensor(1.0))
    kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg)

    kl = kl_prf * w_prf + kl_i * w_i + kl_bg * w_bg
    loss = (neg_ll + kl).mean()

    return {
        "loss": loss,
        "neg_ll": neg_ll.mean().item(),
        "kl_prf": kl_prf.mean().item(),
        "kl_i": kl_i.mean().item(),
        "kl_bg": kl_bg.mean().item(),
        "kl_total": kl.mean().item(),
        "rate_mean": rate.mean().item(),
        "rate_max": rate.max().item(),
    }


def grad_norm_by_module(model):
    """Compute gradient L2 norm for each named module."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            module = name.rsplit(".", 1)[0]
            if module not in norms:
                norms[module] = 0.0
            norms[module] += param.grad.norm().item() ** 2
    return {k: v**0.5 for k, v in norms.items()}


def run_experiment(name, w_prf, w_i, w_bg, kl_warmup_steps=0, seed=42):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(
        f"  weights: prf={w_prf}, i={w_i}, bg={w_bg}, kl_warmup={kl_warmup_steps}"
    )
    print(f"{'=' * 70}")

    torch.manual_seed(seed)
    model = MiniModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    n_samples = len(counts_all)

    history = []

    for step in range(N_STEPS):
        start = (step * BATCH_SIZE) % n_samples
        end = start + BATCH_SIZE
        if end > n_samples:
            start = 0
            end = BATCH_SIZE

        batch_std = standardized[start:end]
        batch_counts = counts_all[start:end]
        batch_masks = masks_all[start:end]

        qp, qi, qbg = model(batch_std)

        # KL annealing: linearly ramp weights from 0 to target over warmup steps
        if kl_warmup_steps > 0 and step < kl_warmup_steps:
            beta = step / kl_warmup_steps
        else:
            beta = 1.0

        loss_dict = compute_loss(
            qp,
            qi,
            qbg,
            batch_counts,
            batch_masks,
            alpha_prior,
            w_prf=w_prf * beta,
            w_i=w_i * beta,
            w_bg=w_bg * beta,
            mc_samples=MC_SAMPLES,
        )

        loss = loss_dict["loss"]
        if loss.isnan():
            print(f"  Step {step}: NaN loss!")
            break

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )
        optimizer.step()

        if step % 50 == 0:
            # Get per-module gradient norms
            mod_grads = grad_norm_by_module(model)
            enc_i_grad = mod_grads.get("enc_intensity.fc", 0) + mod_grads.get(
                "enc_intensity.conv3", 0
            )
            qi_grad = mod_grads.get("qi_head.linear_k", 0) + mod_grads.get(
                "qi_head.linear_r", 0
            )
            qp_grad = mod_grads.get("qp_head.alpha_layer", 0)

            k_mean = qi.concentration.mean().item()
            r_mean = qi.rate.mean().item()
            qi_mean_intensity = (qi.concentration / qi.rate).mean().item()
            alpha_sum = qp.concentration.sum(-1).mean().item()

            record = {
                "step": step,
                "loss": loss.item(),
                "neg_ll": loss_dict["neg_ll"],
                "kl_prf": loss_dict["kl_prf"],
                "kl_i": loss_dict["kl_i"],
                "kl_bg": loss_dict["kl_bg"],
                "kl_total": loss_dict["kl_total"],
                "grad_norm": grad_norm.item(),
                "qi_k_mean": k_mean,
                "qi_r_mean": r_mean,
                "qi_mean_I": qi_mean_intensity,
                "rate_mean": loss_dict["rate_mean"],
                "rate_max": loss_dict["rate_max"],
                "alpha_sum": alpha_sum,
                "beta": beta,
                "enc_i_grad": enc_i_grad,
                "qi_grad": qi_grad,
                "qp_grad": qp_grad,
            }
            history.append(record)

            if step % 150 == 0:
                kl_frac = loss_dict["kl_total"] / (
                    loss_dict["neg_ll"] + loss_dict["kl_total"] + 1e-8
                )
                print(
                    f"  Step {step:4d}: loss={loss.item():>8.1f}  "
                    f"nll={loss_dict['neg_ll']:>8.1f}  "
                    f"kl_tot={loss_dict['kl_total']:>7.1f} ({kl_frac:.0%} of loss)  "
                    f"k/r={qi_mean_intensity:>8.1f}  "
                    f"rate_mean={loss_dict['rate_mean']:>6.1f}  "
                    f"β={beta:.2f}  "
                    f"grad={grad_norm.item():>6.1f}"
                )

    return history


# ── Run experiments ──────────────────────────────────────────────────────
hist_A = run_experiment(
    "A) Downweighted (prf=0.005, i=0.5, bg=0.5)",
    w_prf=0.005,
    w_i=0.5,
    w_bg=0.5,
)

hist_B = run_experiment(
    "B) Full ELBO (all weights = 1.0)",
    w_prf=1.0,
    w_i=1.0,
    w_bg=1.0,
)

hist_C = run_experiment(
    "C) Full ELBO + KL warmup (500 steps)",
    w_prf=1.0,
    w_i=1.0,
    w_bg=1.0,
    kl_warmup_steps=500,
)

hist_D = run_experiment(
    "D) Full ELBO + KL warmup (1000 steps)",
    w_prf=1.0,
    w_i=1.0,
    w_bg=1.0,
    kl_warmup_steps=1000,
)

# ── Summary comparison ───────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY COMPARISON (last recorded step)")
print(f"{'=' * 70}")
print(
    f"{'Experiment':<45} {'loss':>8} {'nll':>8} {'kl_tot':>8} {'kl%':>5} {'k/r':>8} {'rate':>8}"
)
print("-" * 95)
for label, hist in [
    ("A) Downweighted", hist_A),
    ("B) Full ELBO", hist_B),
    ("C) Warmup 500", hist_C),
    ("D) Warmup 1000", hist_D),
]:
    if hist:
        h = hist[-1]
        kl_frac = h["kl_total"] / (h["neg_ll"] + h["kl_total"] + 1e-8)
        print(
            f"{label:<45} {h['loss']:>8.1f} {h['neg_ll']:>8.1f} {h['kl_total']:>8.1f} "
            f"{kl_frac:>4.0%} {h['qi_mean_I']:>8.1f} {h['rate_mean']:>8.1f}"
        )

# ── Detailed trace for stuck detection ───────────────────────────────────
print(f"\n{'=' * 70}")
print("LOSS TRAJECTORY (every 150 steps)")
print(f"{'=' * 70}")
for label, hist in [
    ("A) Downweighted", hist_A),
    ("B) Full ELBO", hist_B),
    ("C) Warmup 500", hist_C),
    ("D) Warmup 1000", hist_D),
]:
    print(f"\n{label}:")
    print(
        f"  {'step':>5} {'loss':>8} {'nll':>8} {'kl_tot':>8} {'kl%':>5} {'qi_mean_I':>10} {'grad':>8}"
    )
    for h in hist:
        if h["step"] % 150 == 0:
            kl_frac = h["kl_total"] / (h["neg_ll"] + h["kl_total"] + 1e-8)
            print(
                f"  {h['step']:>5} {h['loss']:>8.1f} {h['neg_ll']:>8.1f} {h['kl_total']:>8.1f} "
                f"{kl_frac:>4.0%} {h['qi_mean_I']:>10.1f} {h['grad_norm']:>8.1f}"
            )

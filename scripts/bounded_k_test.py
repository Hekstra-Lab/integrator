"""
Test sigmoid-bounded k parameterization.

The core insight: KL(Gamma(k,r) || any Gamma prior) grows only as O(log k),
which is far too slow to prevent unbounded k growth. So we bound k via the
parameterization instead: k = K_MAX * sigmoid(raw_k) + eps.

Tests:
  1) Unbounded baseline (softplus) — k grows without limit
  2) K_MAX = 100  — CV floor = 10%
  3) K_MAX = 200  — CV floor = 7%
  4) K_MAX = 500  — CV floor = 4.5%
  5) K_MAX = 1000 — CV floor = 3.2% (just below NaN boundary)

All use weights = 1.0 and Gamma(2, 0.002) prior (mean=1000).
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
N_STEPS = 3000

# ── Load data ────────────────────────────────────────────────────────────
counts_all = torch.load(f"{DATA_DIR}/counts.pt", weights_only=False).squeeze(
    -1
)
masks_all = torch.load(f"{DATA_DIR}/masks.pt", weights_only=False).squeeze(-1)
stats = torch.load(f"{DATA_DIR}/stats_anscombe.pt", weights_only=False)
conc_raw = torch.load(f"{DATA_DIR}/concentration.pt", weights_only=False)

ans = 2 * torch.sqrt(counts_all + 3.0 / 8.0)
standardized = (ans - stats[0]) / stats[1].sqrt()

conc_prior = conc_raw.clone()
conc_prior[conc_prior > 2] *= 40
n_components = conc_prior.numel()
conc_prior = conc_prior / conc_prior.sum() * n_components
alpha_prior = conc_prior.reshape(-1)

print(f"Data: {counts_all.shape[0]} samples")


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
    def __init__(self, in_features=64, eps=1e-6, k_max=None):
        super().__init__()
        self.linear_k = nn.Linear(in_features, 1)
        self.linear_r = nn.Linear(in_features, 1)
        self.eps = eps
        self.k_max = k_max

    def forward(self, x):
        if self.k_max is not None:
            k = self.k_max * torch.sigmoid(self.linear_k(x)) + self.eps
        else:
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
    def __init__(self, k_max_i=None, k_max_bg=None):
        super().__init__()
        self.enc_profile = SimpleEncoder(64)
        self.enc_intensity = SimpleEncoder(64)
        self.qp_head = DirichletHead()
        self.qi_head = GammaHead(k_max=k_max_i)
        self.qbg_head = GammaHead(k_max=k_max_bg)

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
    qp, qi, qbg, counts, mask, alpha_prior, p_i, p_bg, mc_samples=10, eps=1e-6
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
    kl_i = torch.distributions.kl.kl_divergence(qi, p_i)
    kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg)

    kl = kl_prf + kl_i + kl_bg  # all weights = 1.0
    loss = (neg_ll + kl).mean()

    return {
        "loss": loss,
        "neg_ll": neg_ll.mean().item(),
        "kl_prf": kl_prf.mean().item(),
        "kl_i": kl_i.mean().item(),
        "kl_bg": kl_bg.mean().item(),
        "kl_total": kl.mean().item(),
        "rate_mean": rate.mean().item(),
    }


def run_experiment(name, k_max_i=None, k_max_bg=None, seed=42):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(
        f"  k_max_i={'unbounded' if k_max_i is None else k_max_i}, "
        f"k_max_bg={'unbounded' if k_max_bg is None else k_max_bg}"
    )
    print(f"{'=' * 80}")

    torch.manual_seed(seed)
    model = MiniModel(k_max_i=k_max_i, k_max_bg=k_max_bg)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    p_i = Gamma(torch.tensor(2.0), torch.tensor(0.002))
    p_bg = Gamma(torch.tensor(2.0), torch.tensor(0.5))
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

        loss_dict = compute_loss(
            qp,
            qi,
            qbg,
            batch_counts,
            batch_masks,
            alpha_prior,
            p_i=p_i,
            p_bg=p_bg,
            mc_samples=MC_SAMPLES,
        )

        loss = loss_dict["loss"]
        if loss.isnan() or loss.isinf():
            print(f"  Step {step}: loss {'NaN' if loss.isnan() else 'Inf'}!")
            history.append({"step": step, "nan": True})
            break

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        if torch.isnan(grad_norm):
            print(f"  Step {step}: gradient NaN!")
            history.append({"step": step, "nan": True})
            break

        optimizer.step()

        if step % 100 == 0:
            k_mean = qi.concentration.mean().item()
            k_max_val = qi.concentration.max().item()
            k_min_val = qi.concentration.min().item()
            qi_mean_I = (qi.concentration / qi.rate).mean().item()
            qi_cv = (1.0 / qi.concentration.sqrt()).mean().item()

            # Background k stats
            bg_k_mean = qbg.concentration.mean().item()

            record = {
                "step": step,
                "loss": loss.item(),
                "neg_ll": loss_dict["neg_ll"],
                "kl_i": loss_dict["kl_i"],
                "kl_bg": loss_dict["kl_bg"],
                "kl_total": loss_dict["kl_total"],
                "grad_norm": grad_norm.item(),
                "qi_k_mean": k_mean,
                "qi_k_max": k_max_val,
                "qi_k_min": k_min_val,
                "qi_mean_I": qi_mean_I,
                "qi_cv_mean": qi_cv,
                "bg_k_mean": bg_k_mean,
            }
            history.append(record)

            if step % 300 == 0:
                print(
                    f"  Step {step:4d}: loss={loss.item():>8.1f}  "
                    f"nll={loss_dict['neg_ll']:>8.1f}  "
                    f"kl_i={loss_dict['kl_i']:>6.1f}  "
                    f"k=[{k_min_val:.1f},{k_max_val:.1f}]  "
                    f"CV={qi_cv:.2f}  "
                    f"I={qi_mean_I:>7.0f}  "
                    f"bg_k={bg_k_mean:.1f}  "
                    f"grad={grad_norm.item():>6.1f}"
                )

    if history and "nan" not in history[-1]:
        h = history[-1]
        print(
            f"  Final: loss={h['loss']:.1f}, nll={h['neg_ll']:.1f}, "
            f"k=[{h['qi_k_min']:.1f},{h['qi_k_max']:.1f}], CV={h['qi_cv_mean']:.3f}"
        )

    return {"name": name, "k_max_i": k_max_i, "history": history}


# ── Run experiments ──────────────────────────────────────────────────────

results = []

results.append(
    run_experiment(
        "1) Unbounded (softplus) — baseline",
        k_max_i=None,
        k_max_bg=None,
    )
)

results.append(
    run_experiment(
        "2) K_MAX=100 (CV floor=10%)",
        k_max_i=100,
        k_max_bg=100,
    )
)

results.append(
    run_experiment(
        "3) K_MAX=200 (CV floor=7%)",
        k_max_i=200,
        k_max_bg=200,
    )
)

results.append(
    run_experiment(
        "4) K_MAX=500 (CV floor=4.5%)",
        k_max_i=500,
        k_max_bg=500,
    )
)

results.append(
    run_experiment(
        "5) K_MAX=1000 (CV floor=3.2%)",
        k_max_i=1000,
        k_max_bg=1000,
    )
)


# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("SUMMARY — bounded k with weights=1.0, Gamma(2,0.002) prior, 3000 steps")
print(f"{'=' * 100}")
print(
    f"{'Config':<40} {'status':>7} {'loss':>7} {'nll':>7} {'kl_i':>6} {'k_mean':>7} {'k_max':>7} {'CV':>6}"
)
print("-" * 100)

for r in results:
    h_list = [h for h in r["history"] if "nan" not in h]
    if not h_list:
        print(f"{r['name']:<40} {'NaN':>7}")
    else:
        h = h_list[-1]
        print(
            f"{r['name']:<40} {'  OK':>7} {h['loss']:>7.0f} {h['neg_ll']:>7.0f} "
            f"{h['kl_i']:>6.1f} {h['qi_k_mean']:>7.1f} {h['qi_k_max']:>7.1f} {h['qi_cv_mean']:>6.3f}"
        )


# ── K saturation analysis ────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("K SATURATION: does k hit the boundary?")
print(f"{'=' * 100}")

for r in results:
    h_list = [h for h in r["history"] if "nan" not in h]
    if not h_list:
        continue
    k_max_bound = r["k_max_i"]
    print(f"\n{r['name']}:")
    print(
        f"  {'step':>5} {'k_mean':>8} {'k_max':>8} {'k_min':>8} {'utilization':>12} {'loss':>8} {'nll':>8}"
    )
    for h in h_list:
        if h["step"] % 600 == 0:
            if k_max_bound is not None:
                util = f"{h['qi_k_max'] / k_max_bound:.0%}"
            else:
                util = "n/a"
            print(
                f"  {h['step']:>5} {h['qi_k_mean']:>8.1f} {h['qi_k_max']:>8.1f} "
                f"{h['qi_k_min']:>8.1f} {util:>12} {h['loss']:>8.1f} {h['neg_ll']:>8.1f}"
            )


# ── Loss quality comparison ──────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("LOSS TRAJECTORY COMPARISON (every 600 steps)")
print(f"{'=' * 100}")
print(f"{'step':>5}", end="")
for r in results:
    label = r["name"][:15]
    print(f"  {label:>15}", end="")
print()
print("-" * (5 + 17 * len(results)))

all_steps = set()
for r in results:
    for h in r["history"]:
        if "nan" not in h:
            all_steps.add(h["step"])

for step in sorted(all_steps):
    if step % 600 != 0:
        continue
    print(f"{step:>5}", end="")
    for r in results:
        found = False
        for h in r["history"]:
            if "nan" not in h and h["step"] == step:
                print(f"  {h['neg_ll']:>15.1f}", end="")
                found = True
                break
        if not found:
            print(f"  {'—':>15}", end="")
    print()

print("\n(Showing NLL only — lower is better. KL adds 20-30 to total loss.)")

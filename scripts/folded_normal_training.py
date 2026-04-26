"""
Test FoldedNormal in actual training to confirm it's stable after the softplus fix.

Compares:
  A) FoldedNormalA with exp(raw_loc) — original (should NaN)
  B) FoldedNormalA with softplus(raw_loc) — fixed
  C) GammaRepamA unbounded — for comparison (will grow k)
  D) LogNormal — gold standard (should always work)

All with weights = 1.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Gamma, LogNormal, Poisson

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

import sys

sys.path.insert(0, "src")
from integrator.model.distributions.folded_normal import FoldedNormal


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


class DirichletHead(nn.Module):
    def __init__(self, in_features=64, n_pixels=441, eps=1e-6):
        super().__init__()
        self.alpha_layer = nn.Linear(in_features, n_pixels)
        self.eps = eps

    def forward(self, x):
        return Dirichlet(F.softplus(self.alpha_layer(x)) + self.eps)


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


class FoldedNormalHeadExp(nn.Module):
    """Original: exp(raw_loc) — OVERFLOWS."""

    def __init__(self, in_features=64, eps=0.1):
        super().__init__()
        self.linear_loc = nn.Linear(in_features, 1)
        self.linear_scale = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(self, x):
        loc = (torch.exp(self.linear_loc(x)) + self.eps).squeeze()
        scale = (F.softplus(self.linear_scale(x)) + self.eps).squeeze()
        return FoldedNormal(loc, scale)


class FoldedNormalHeadSoftplus(nn.Module):
    """Fixed: softplus(raw_loc)."""

    def __init__(self, in_features=64, eps=0.1):
        super().__init__()
        self.linear_loc = nn.Linear(in_features, 1)
        self.linear_scale = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(self, x):
        loc = (F.softplus(self.linear_loc(x)) + self.eps).squeeze()
        scale = (F.softplus(self.linear_scale(x)) + self.eps).squeeze()
        return FoldedNormal(loc, scale)


class LogNormalHead(nn.Module):
    """LogNormal: pathwise reparameterization, always stable."""

    def __init__(self, in_features=64, eps=1e-3):
        super().__init__()
        self.linear_mu = nn.Linear(in_features, 1)
        self.linear_sigma = nn.Linear(in_features, 1)
        self.eps = eps

    def forward(self, x):
        mu = self.linear_mu(x).flatten()  # unconstrained
        sigma = F.softplus(self.linear_sigma(x)).flatten() + self.eps
        return LogNormal(mu, sigma)


class MiniModel(nn.Module):
    def __init__(self, head_cls, head_bg_cls=None):
        super().__init__()
        self.enc_profile = SimpleEncoder(64)
        self.enc_intensity = SimpleEncoder(64)
        self.qp_head = DirichletHead()
        self.qi_head = head_cls(in_features=64)
        self.qbg_head = (head_bg_cls or head_cls)(in_features=64)

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


def kl_mc(q, p, mc_samples=50):
    """MC estimate of KL(q || p)."""
    samples = q.rsample([mc_samples])
    return (q.log_prob(samples) - p.log_prob(samples)).mean(dim=0)


def compute_loss(
    qp,
    qi,
    qbg,
    counts,
    mask,
    alpha_prior,
    mc_samples=10,
    eps=1e-6,
    prior_type="gamma",
):
    zI = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zp = qp.rsample([mc_samples]).permute(1, 0, 2)
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    rate = zI * zp + zbg

    ll = Poisson(rate + eps).log_prob(counts.unsqueeze(1))
    ll_mean = torch.mean(ll, dim=1) * mask.squeeze(-1)
    neg_ll = (-ll_mean).sum(1)

    # Profile KL
    p_dir = Dirichlet(alpha_prior)
    kl_prf = torch.distributions.kl.kl_divergence(qp, p_dir)

    # Intensity / background KL depends on distribution type
    if prior_type == "gamma":
        p_i = Gamma(torch.tensor(2.0), torch.tensor(0.002))
        p_bg = Gamma(torch.tensor(2.0), torch.tensor(0.5))
        kl_i = torch.distributions.kl.kl_divergence(qi, p_i)
        kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg)
    elif prior_type == "lognormal":
        # LogNormal prior: mean ≈ exp(mu_p) ≈ 1000 → mu_p ≈ 6.9
        p_i = LogNormal(6.9, 2.0)  # broad prior centered near 1000
        p_bg = LogNormal(0.0, 2.0)  # broad prior centered near 1
        kl_i = torch.distributions.kl.kl_divergence(qi, p_i)
        kl_bg = torch.distributions.kl.kl_divergence(qbg, p_bg)
    elif prior_type == "folded_normal":
        # MC KL against Gamma prior (no analytic form)
        p_i = Gamma(torch.tensor(2.0), torch.tensor(0.002))
        p_bg = Gamma(torch.tensor(2.0), torch.tensor(0.5))
        kl_i = kl_mc(qi, p_i)
        kl_bg = kl_mc(qbg, p_bg)
    else:
        raise ValueError(prior_type)

    kl = kl_prf + kl_i + kl_bg  # all weights = 1.0
    loss = (neg_ll + kl).mean()

    return {
        "loss": loss,
        "neg_ll": neg_ll.mean().item(),
        "kl_i": kl_i.mean().item(),
        "kl_bg": kl_bg.mean().item(),
        "kl_total": kl.mean().item(),
    }


def get_dist_stats(dist):
    """Get mean and 'precision' info from any distribution."""
    if isinstance(dist, Gamma):
        return (
            dist.concentration.mean().item(),
            dist.concentration.max().item(),
            "k",
        )
    elif isinstance(dist, FoldedNormal):
        loc_scale = (dist.loc / dist.scale).mean().item()
        return (
            dist.loc.mean().item(),
            (dist.loc / dist.scale).max().item(),
            "loc/sc",
        )
    elif isinstance(dist, LogNormal):
        # mean = exp(mu + sigma²/2)
        mean = (dist.loc + dist.scale**2 / 2).exp().mean().item()
        inv_sigma = (1 / dist.scale).max().item()
        return mean, inv_sigma, "1/σ"
    return 0, 0, "?"


def run_experiment(name, head_cls, prior_type, seed=42):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")

    torch.manual_seed(seed)
    model = MiniModel(head_cls)
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

        loss_dict = compute_loss(
            qp,
            qi,
            qbg,
            batch_counts,
            batch_masks,
            alpha_prior,
            mc_samples=MC_SAMPLES,
            prior_type=prior_type,
        )

        loss = loss_dict["loss"]
        if loss.isnan() or loss.isinf():
            print(f"  Step {step}: loss {'NaN' if loss.isnan() else 'Inf'}!")
            qi_mean, qi_prec, label = get_dist_stats(qi)
            print(f"    qi: mean={qi_mean:.1f}, {label}={qi_prec:.1f}")
            history.append({"step": step, "nan": True})
            break

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        if torch.isnan(grad_norm):
            print(f"  Step {step}: gradient NaN!")
            qi_mean, qi_prec, label = get_dist_stats(qi)
            print(f"    qi: mean={qi_mean:.1f}, {label}={qi_prec:.1f}")
            # Find which params have NaN grads
            for pname, p in model.named_parameters():
                if p.grad is not None and p.grad.isnan().any():
                    print(f"    NaN grad: {pname}")
            history.append({"step": step, "nan": True})
            break

        optimizer.step()

        if step % 100 == 0:
            qi_mean, qi_prec, label = get_dist_stats(qi)
            record = {
                "step": step,
                "loss": loss.item(),
                "neg_ll": loss_dict["neg_ll"],
                "kl_i": loss_dict["kl_i"],
                "kl_total": loss_dict["kl_total"],
                "qi_mean": qi_mean,
                "qi_prec": qi_prec,
                "prec_label": label,
                "grad_norm": grad_norm.item(),
            }
            history.append(record)

            if step % 300 == 0:
                print(
                    f"  Step {step:4d}: loss={loss.item():>8.1f}  "
                    f"nll={loss_dict['neg_ll']:>8.1f}  "
                    f"kl_i={loss_dict['kl_i']:>6.1f}  "
                    f"mean_I={qi_mean:>8.0f}  "
                    f"{label}={qi_prec:>8.1f}  "
                    f"grad={grad_norm.item():>6.1f}"
                )

    return {"name": name, "history": history}


# ── Run experiments ──────────────────────────────────────────────────────
results = []

results.append(
    run_experiment(
        "A) FoldedNormal + exp(raw_loc) — ORIGINAL",
        FoldedNormalHeadExp,
        "folded_normal",
    )
)

results.append(
    run_experiment(
        "B) FoldedNormal + softplus(raw_loc) — FIXED",
        FoldedNormalHeadSoftplus,
        "folded_normal",
    )
)

results.append(
    run_experiment(
        "C) Gamma (unbounded) — comparison",
        GammaHead,
        "gamma",
    )
)

results.append(
    run_experiment(
        "D) LogNormal — gold standard",
        LogNormalHead,
        "lognormal",
    )
)


# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("SUMMARY — all weights = 1.0, 3000 steps")
print(f"{'=' * 90}")
for r in results:
    h_list = [h for h in r["history"] if "nan" not in h]
    if not h_list:
        nan_step = [h for h in r["history"] if "nan" in h]
        step = nan_step[0]["step"] if nan_step else "?"
        print(f"  {r['name']:<50} NaN at step {step}")
    else:
        h = h_list[-1]
        print(
            f"  {r['name']:<50} loss={h['loss']:>7.0f}  nll={h['neg_ll']:>7.0f}  "
            f"kl_i={h['kl_i']:>6.1f}  {h['prec_label']}={h['qi_prec']:>7.1f}"
        )

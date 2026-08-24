"""Train the per-image-Wilson integrator on the SFX sim and track ground-truth recovery.

Uses the REAL production architecture -- the same 5-encoder hierarchical stack
(`ProfileEncoder` + 4x `IntensityEncoder`), the same `build_gamma(shape_rate)` surrogates
for q(I)/q(bg), the same learned-basis `ProfileSurrogate` for q(profile), and the exact
`CountLikelihood` Poisson/Normal classes -- on JUNGFRAU detector data. The ONE deliberate
difference is the prior: the Wilson G is per-image (an `Embedding` indexed by image_id)
rather than one global scalar, which is the thing under study.

Experiment knobs (compose freely):
    --likelihood  poisson | normal_coupled | normal_free   int vs real data + likelihood
    --profile     learned | known                           learned-basis q(profile) vs oracle
    --scale       per_image | global                        per-image Wilson G vs one shared G
    --per-image-B                                            per-image B vs a single global B

Model, per shoebox (observation j on image i) -- identical to HierarchicalIntegrator:
    5 encoders(asinh counts) -> q(I)=Gamma(k_i,r_i), q(bg)=Gamma(k_bg,r_bg),
                                q(profile)=learned-basis ProfileSurrogate
    prior  I_j ~ Exp(1/mu),  mu = G_i exp(-2 B_i s_j^2)
    recon  rate = zI * zp + zbg,  CountLikelihood(rate, counts)
    ELBO   E_q[-log p(counts|rate)] + KL(q(I)||Exp) + KL(q(bg)||Gamma) + profile KL

For long cluster runs: history.json is flushed every eval (crash-safe), checkpoints are
written every eval, and --resume continues from the latest checkpoint. Data lives on the
host; minibatches move to the device, so large sims do not have to fit in GPU memory.

Run:  uv run python scripts/jungfrau_sim/sfx_experiment.py --likelihood poisson --profile learned
See scripts/jungfrau_sim/run_sfx_experiments.sh for the full matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Gamma, kl_divergence

# The REAL production components, so the architecture is identical to the integrator's:
# the 5-encoder hierarchical stack, the learned-basis profile surrogate, the Gamma
# intensity/background surrogates, and the exact Poisson/Normal pixel likelihood.
from integrator.model.distributions.gamma import build_gamma
from integrator.model.distributions.profile_surrogates import ProfileSurrogate
from integrator.model.encoders import IntensityEncoder, ProfileEncoder
from integrator.model.loss.count_likelihood import CountLikelihood
from integrator.model.loss.kl_helpers import compute_profile_kl
from integrator.utils.factory_utils import _apply_encoder_preset

torch.set_default_dtype(torch.float32)

DATA_FILE = {  # Normal reads real-valued counts; Poisson reads the rounded integers.
    "poisson": "counts_poisson",
    "normal_coupled": "counts_real",
    "normal_free": "counts_real",
}


def run_tag(args) -> str:
    t = f"{args.likelihood}_{args.profile}_{args.scale}"
    return t + ("_Bi" if args.per_image_B else "")


# ── data ───────────────────────────────────────────────────────────────────────


class SFXData:
    """Loads a `sfx_generate.py` directory. Tensors stay on the HOST; move batches later."""

    def __init__(self, data_dir: Path, likelihood: str):
        d = data_dir
        self.counts = torch.tensor(np.load(d / f"{DATA_FILE[likelihood]}.npy").astype(np.float32))
        self.profiles = torch.tensor(np.load(d / "profiles.npy").astype(np.float32))
        self.image_id = torch.tensor(np.load(d / "image_id.npy")).long()
        self.s_sq = torch.tensor(np.load(d / "s_sq.npy").astype(np.float32))
        self.i_true = torch.tensor(np.load(d / "intensity_true.npy").astype(np.float32))
        self.bg_true = torch.tensor(np.load(d / "background_true.npy").astype(np.float32))
        self.g_true = torch.tensor(np.load(d / "g_true_per_image.npy").astype(np.float32))
        self.b_true = torch.tensor(np.load(d / "b_true_per_image.npy").astype(np.float32))
        self.n_obs, self.n_pix = self.counts.shape
        self.n_images = int(self.image_id.max()) + 1
        self.manifest = json.loads((d / "sim.json").read_text())
        h = self.manifest["geometry"]["h"]
        self.hw = (h, self.manifest["geometry"]["w"])

    def image_mean_intensity(self, profile_known: bool) -> torch.Tensor:
        """Per-image mean intensity anchor -- the data-driven init for the G embedding."""
        prof = self.profiles if profile_known else None
        i_hat, _ = count_anchors(self.counts, prof)
        num = torch.bincount(self.image_id, weights=i_hat, minlength=self.n_images)
        den = torch.bincount(self.image_id, minlength=self.n_images).clamp_min(1)
        return (num / den).clamp_min(1.0)


def count_anchors(
    counts: torch.Tensor, profile: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust per-shoebox (intensity, background) scale anchors from raw counts.

    The background dominates the raw total (bg ~0.7 ph/px x 400 px vs a mean intensity
    ~34), so the intensity anchor must subtract it. Two modes:

    - profile KNOWN: background from the low-profile pixels, then a profile matched filter
      sum_p profile_p (counts_p - bg) / sum_p profile_p^2  (near-optimal given the profile).
    - profile UNKNOWN (learned experiment): a profile-free estimate -- background from the
      low-COUNT pixels, then the background-subtracted TOTAL, which is unbiased for I for
      ANY profile. The model must recover profile-fitting quality via its learned q(p),
      so it may not peek at the true profile through the anchor.

    The floor is 0.05, not 1: clamping weak high-resolution reflections at 1 photon would
    hide the Wilson resolution falloff and make B unlearnable.
    """
    if profile is not None:
        low = profile < profile.median(dim=1, keepdim=True).values
        bg = (counts * low).sum(1) / low.sum(1).clamp_min(1)
        resid = counts - bg.unsqueeze(1)
        i_hat = (profile * resid).sum(1) / (profile * profile).sum(1)
    else:
        # background = mean of the lower 60% of pixels by value (signal is concentrated).
        k = max(1, int(0.6 * counts.shape[1]))
        bg = counts.sort(dim=1).values[:, :k].mean(1)
        i_hat = (counts - bg.unsqueeze(1)).sum(1)                    # bg-subtracted total
    return i_hat.clamp_min(0.05), bg.clamp_min(1e-2)


def asinh_std(counts: torch.Tensor) -> torch.Tensor:
    """The asinh encoder-input transform added to the real data module."""
    return torch.asinh(counts)


# ── model ──────────────────────────────────────────────────────────────────────


class SFXIntegrator(nn.Module):
    """The real 5-encoder hierarchical stack + a per-image Wilson prior.

    Architecture is IDENTICAL to the production `HierarchicalIntegrator`: the same
    `ProfileEncoder` + 4x `IntensityEncoder` (k_i, r_i, k_bg, r_bg), the same
    `build_gamma(shape_rate)` surrogates for q(I)/q(bg), the same learned-basis
    `ProfileSurrogate` for q(profile), and the same `rate = zI * zp + zbg` composition.
    What differs is only the prior: G is per-image (an Embedding indexed by image_id)
    instead of one global scalar, which is the thing under study.

    `--profile known` swaps zp for the true profile as an oracle control, to isolate
    scale/intensity recovery from profile learning. `--profile learned` (the default)
    uses the real learned-basis surrogate.
    """

    def __init__(self, data: SFXData, args):
        super().__init__()
        self.mc = args.mc
        self.scale_per_image = args.scale == "per_image"
        self.per_image_B = args.per_image_B
        self.profile_known = args.profile == "known"
        self.hw = data.hw
        n_pix = data.n_pix
        eo = args.encoder_out

        # 5 encoders, built through the factory's own 2d presets so they are constructed
        # exactly as `resolve_config` would from a YAML config (kernel/pool sizes etc.).
        prof_args = _apply_encoder_preset("profile_encoder", {"data_dim": "2d"})
        prof_args.update(encoder_out=eo, input_shape=tuple(data.hw))
        int_args = _apply_encoder_preset("intensity_encoder", {"data_dim": "2d"})
        int_args.update(encoder_out=eo)

        self.enc_profile = ProfileEncoder(**prof_args)
        self.enc_k_i, self.enc_r_i, self.enc_k_bg, self.enc_r_bg = (
            IntensityEncoder(**int_args) for _ in range(4)
        )
        self.qi = build_gamma("shape_rate", in_features=eo, eps=1e-6, k_min=0.01)
        self.qbg = build_gamma("shape_rate", in_features=eo, eps=1e-6, k_min=0.01)
        self.qp = ProfileSurrogate(
            input_dim=eo, latent_dim=args.latent_dim, output_dim=n_pix,
            init_std=0.5, prior_scale=3.0,
        )
        self.profile_kl_weight = 1.0

        init = torch.log(data.image_mean_intensity(self.profile_known))
        if self.scale_per_image:
            self.log_G = nn.Embedding(data.n_images, 1)
            with torch.no_grad():
                self.log_G.weight.copy_(init.unsqueeze(1))
        else:
            self.log_G = nn.Parameter(init.mean().reshape(1))

        n_b = data.n_images if self.per_image_B else 1
        self.raw_B = (nn.Embedding(data.n_images, 1) if self.per_image_B
                      else nn.Parameter(torch.tensor(2.0)))
        if self.per_image_B:
            with torch.no_grad():
                self.raw_B.weight.fill_(2.0)
        self.bg_prior_rate = 1.0 / max(data.manifest["bg_mean"], 1e-3)
        self.prof_prior_conc = 1.0

    def gathered_logG(self, image_id):
        return self.log_G(image_id).squeeze(1) if self.scale_per_image \
            else self.log_G.expand(len(image_id))

    def gathered_B(self, image_id):
        raw = self.raw_B(image_id).squeeze(1) if self.per_image_B \
            else self.raw_B.expand(len(image_id))
        return torch.nn.functional.softplus(raw)

    def hyper_params(self):
        return [self.log_G.weight if self.scale_per_image else self.log_G,
                self.raw_B.weight if self.per_image_B else self.raw_B]

    def encode(self, counts):
        """Run the 5 encoders on the standardized shoebox; returns the surrogates."""
        sbox = asinh_std(counts).reshape(-1, 1, *self.hw)
        q_i = self.qi(self.enc_k_i(sbox), self.enc_r_i(sbox))
        q_bg = self.qbg(self.enc_k_bg(sbox), self.enc_r_bg(sbox))
        q_p = self.qp(self.enc_profile(sbox), mc_samples=self.mc)
        return q_i, q_bg, q_p

    def forward(self, counts, profile_true, image_id, s_sq, likelihood, kl_weight):
        q_i, q_bg, q_p = self.encode(counts)

        # Same composition as HierarchicalIntegrator._forward_impl: (B, S, .) throughout.
        zI = q_i.rsample([self.mc]).unsqueeze(-1).permute(1, 0, 2)     # (B,S,1)
        zbg = q_bg.rsample([self.mc]).unsqueeze(-1).permute(1, 0, 2)   # (B,S,1)
        if self.profile_known:
            zp = profile_true.unsqueeze(1)                              # (B,1,P) oracle
            kl_p = torch.zeros(len(image_id), device=counts.device)
        else:
            zp = q_p.zp.permute(1, 0, 2)                                # (B,S,P)
            kl_p = compute_profile_kl(
                q_p, q_p.prior_scale, self.profile_kl_weight, counts.device
            )
        rate = zI * zp + zbg
        mask = torch.ones_like(counts).unsqueeze(-1)
        neg_ll = likelihood.neg_ll(rate, counts, mask)

        mu = torch.exp(self.gathered_logG(image_id)) * torch.exp(
            -2.0 * self.gathered_B(image_id) * s_sq
        )
        p_i = Gamma(torch.ones_like(mu), (1.0 / mu.clamp_min(1e-6)).clamp_min(1e-8))
        kl_i = kl_divergence(q_i, p_i).squeeze(-1)
        p_bg = Gamma(torch.ones(len(image_id), device=counts.device),
                     torch.full((len(image_id),), self.bg_prior_rate, device=counts.device))
        kl_bg = kl_divergence(q_bg, p_bg).squeeze(-1)

        loss = (neg_ll + kl_weight * (kl_i + kl_bg + kl_p)).mean()
        return {"loss": loss, "neg_ll": neg_ll.mean().detach()}


# ── recovery evaluation ────────────────────────────────────────────────────────


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def recover(model, data, device, batch=8192) -> dict:
    model.eval()
    i_hat = torch.empty(data.n_obs)
    bg_hat = torch.empty(data.n_obs)
    prof_corr = []
    for s in range(0, data.n_obs, batch):
        e = slice(s, min(s + batch, data.n_obs))
        c = data.counts[e].to(device)
        prof = data.profiles[e].to(device)
        q_i, q_bg, q_p = model.encode(c)
        i_hat[e] = q_i.mean.squeeze(-1).cpu()
        bg_hat[e] = q_bg.mean.squeeze(-1).cpu()
        if not model.profile_known:
            prof_corr.append(
                torch.nn.functional.cosine_similarity(
                    q_p.mean_profile, prof, dim=1
                ).mean().cpu()
            )

    g = torch.exp((model.log_G.weight.squeeze(1) if model.scale_per_image
                   else model.log_G).detach().cpu())
    b_all = model.gathered_B(torch.arange(data.n_images, device=device)).detach().cpu()
    li, lt = torch.log(i_hat.clamp_min(1e-3)), torch.log(data.i_true.clamp_min(1e-3))
    lg, lgt = torch.log(g.clamp_min(1e-6)), torch.log(data.g_true)
    model.train()
    m = {
        "corr_logI": _corr(li, lt), "rmse_logI": float((li - lt).pow(2).mean().sqrt()),
        "corr_logG": _corr(lg, lgt) if model.scale_per_image else 0.0,
        "B_mean": float(b_all.mean()), "B_err": float((b_all - data.b_true).abs().mean()),
        "corr_bg": _corr(bg_hat, data.bg_true),
    }
    if prof_corr:
        m["profile_cos"] = float(torch.stack(prof_corr).mean())
    return m


# ── training ───────────────────────────────────────────────────────────────────


def build_likelihood(name, device):
    base = "normal" if name.startswith("normal") else "poisson"
    kw = {"variance": "coupled", "read_noise": 0.024} if name == "normal_coupled" \
        else ({"variance": "free"} if name == "normal_free" else {})
    return CountLikelihood(base, **kw).to(device)


def resolve_device(spec):
    if spec != "auto":
        return spec
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(args) -> None:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    data = SFXData(Path(args.data), args.likelihood)
    likelihood = build_likelihood(args.likelihood, device)
    model = SFXIntegrator(data, args).to(device)

    hyper = model.hyper_params()
    enc = [p for p in model.parameters() if all(p is not h for h in hyper)]
    opt = torch.optim.Adam([
        {"params": enc + list(likelihood.parameters()), "lr": args.lr},
        {"params": [hyper[0]], "lr": args.lr * 10},   # per-image / global G
        {"params": [hyper[1]], "lr": args.lr * 40},   # B: weak KL-only gradient
    ])

    out = Path(args.out) / run_tag(args)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    history, start = [], 1
    if args.resume:
        ck = sorted((out / "checkpoints").glob("epoch_*.pt"))
        if ck:
            st = torch.load(ck[-1], map_location=device)
            model.load_state_dict(st["model"])
            likelihood.load_state_dict(st["likelihood"])
            opt.load_state_dict(st["opt"])
            start = st["epoch"] + 1
            if (out / "history.json").exists():
                history = json.loads((out / "history.json").read_text())
            print(f"resumed from epoch {st['epoch']}")

    idx = torch.arange(data.n_obs)
    print(f"[{run_tag(args)}] {data.n_obs} obs, {data.n_images} images, device={device}")
    for epoch in range(start, args.epochs + 1):
        model.train()
        perm = idx[torch.randperm(data.n_obs)]
        kl_w = min(1.0, epoch / max(args.kl_warmup, 1))
        for s in range(0, data.n_obs, args.batch):
            b = perm[s:s + args.batch]
            out_d = model(
                data.counts[b].to(device), data.profiles[b].to(device),
                data.image_id[b].to(device), data.s_sq[b].to(device), likelihood, kl_w,
            )
            opt.zero_grad()
            out_d["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = recover(model, data, device)
            m["epoch"] = epoch
            m["loss"] = float(out_d["loss"].detach())
            history.append(m)
            torch.save({"model": model.state_dict(), "likelihood": likelihood.state_dict(),
                        "opt": opt.state_dict(), "epoch": epoch, "metrics": m, "args": vars(args)},
                       out / "checkpoints" / f"epoch_{epoch:04d}.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2))  # crash-safe
            extra = f"  prof {m['profile_cos']:.3f}" if "profile_cos" in m else ""
            print(f"  ep {epoch:4d}  loss {m['loss']:8.1f}  corr(logI) {m['corr_logI']:.3f}  "
                  f"corr(logG) {m['corr_logG']:.3f}  B {m['B_mean']:5.1f}  "
                  f"corr(bg) {m['corr_bg']:.3f}{extra}", flush=True)

    print(f"  -> {len(history)} evals + history.json in {out}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--likelihood", choices=list(DATA_FILE), default="poisson")
    ap.add_argument("--profile", choices=["known", "learned"], default="learned")
    ap.add_argument("--scale", choices=["per_image", "global"], default="per_image")
    ap.add_argument("--per-image-B", action="store_true")
    ap.add_argument("--data", default="data/sfx_sim")
    ap.add_argument("--out", default="data/sfx_runs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--mc", type=int, default=8)
    ap.add_argument("--encoder-out", type=int, default=32,
                    help="encoder embedding width (production default 32)")
    ap.add_argument("--latent-dim", type=int, default=12,
                    help="learned-basis profile latent dim (production default 12)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--kl-warmup", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()

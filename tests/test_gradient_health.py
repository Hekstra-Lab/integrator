"""Every sim_truth config trains every parameter it builds, except the ones its config says it cannot reach.

Each config is built through `construct_integrator` and trained four Adam steps
on a synthetic batch. The loss and every gradient must stay finite, and the set
of parameters that received no gradient in any of the four steps must equal the
set the config predicts. Four steps, not one: a head whose last layer starts at
zero (the geometry heads of `geometric_basis_profile`) passes no gradient to
its first layer at step 0 and does from step 1.

The parameters a config cannot reach, and why:

- `surrogates.qi`, `surrogates.qbg`: the `svae` family builds q(I) and q(b) by a
  conjugate update, so the amortized heads are built and never called; q(I)
  uses its head only under `intensity_posterior: amortized`.
- `encoders.signal`, `gate_head`: plain `svae` with a computed gate
  (`responsibility`, `joint`) never reads them (svae_profile_encoder_dead,
  CLAUDE.md "Testing an architecture"). `svae_bg` and `svae_conjugate` give the
  encoder a job.
- `encoders.profile`: the profile takes no per-reflection input --
  `residual: none` / `False` on the streak families, `shape_posterior: none`
  as well on `streak_kmix_v2_profile`, `latent_gain: 0` on
  `geometric_basis_profile` unless `gate_extent` feeds the encoder to the extent
  residual; `latent_gain: 0` always strands its `loc_head` and `scale_head`.

Anything else with no gradient fails the test, and so does an expected group
that does receive gradient, so the list cannot go stale. Learned prior
parameters (Wilson G and B, the background map) are in the loss and must train.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest
import torch

from integrator.registry import REGISTRY
from integrator.utils.factory_utils import construct_integrator

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "sim_truth").glob("*.yaml"))
B, SIDE = 6, 25
P = SIDE * SIDE
STEPS = 4

SVAE_FAMILY = {"svae", "svae_bg", "svae_conjugate", "svae_calib"}


def _batch():
    torch.manual_seed(0)
    x = torch.tensor([600.0, 3500.0, 2074.5, 900.0, 3000.0, 1500.0])
    y = torch.tensor([700.0, 900.0, 3600.0, 3400.0, 2181.5, 1200.0])
    counts = torch.poisson(torch.full((B, P), 20.0))
    counts[:, P // 2] += 400.0
    metadata = {
        "group_label": torch.zeros(B, dtype=torch.long),
        "xyzcal.px.0": x,
        "xyzcal.px.1": y,
        "bbox.0": torch.floor(x) - SIDE // 2,
        "bbox.2": torch.floor(y) - SIDE // 2,
        "d": torch.tensor([1.8, 2.0, 2.5, 3.0, 4.0, 6.0]),
        "wavelength": torch.tensor([1.03, 1.05, 1.08, 1.11, 1.14, 1.17]),
        "image_num": torch.tensor([1.0, 2.0, 3.0, 90.0, 179.0, 180.0]),
    }
    return counts, counts.clone().unsqueeze(-1), torch.ones(B, P, 1), metadata


def _arg(cfg_args: dict, cls, key: str):
    """A surrogate argument from the config, else the class default."""
    if key in cfg_args:
        return cfg_args[key]
    p = inspect.signature(cls.__init__).parameters.get(key)
    return None if p is None or p.default is inspect.Parameter.empty else p.default


def expected_unreached(cfg: dict) -> set[str]:
    """Module prefixes the config cannot reach, by the rules in the module docstring."""
    out: set[str] = set()
    integ = cfg["integrator"]["name"]
    ia = cfg["integrator"].get("args", {}) or {}
    if integ in SVAE_FAMILY:
        out.add("surrogates.qbg")
        if ia.get("intensity_posterior", "conjugate") != "amortized":
            out.add("surrogates.qi")
    if integ == "svae" and ia.get("gate", "learned") in ("responsibility", "joint"):
        out |= {"encoders.signal", "gate_head"}

    qp = cfg.get("surrogates", {}).get("qp", {})
    name, a = qp.get("name"), qp.get("args", {}) or {}
    cls = REGISTRY["surrogates"].get(name)
    if cls is not None:
        res = _arg(a, cls, "residual")
        if name in ("streak_kmix_profile", "streak_mixture_profile") and res == "none":
            out.add("encoders.profile")
        elif name == "streak_base_profile" and res is False:
            out.add("encoders.profile")
        elif name == "streak_kmix_v2_profile" and res == "none" and _arg(a, cls, "shape_posterior") == "none":
            out.add("encoders.profile")
        elif name == "geometric_basis_profile" and _arg(a, cls, "latent_gain") == 0.0:
            out |= {"surrogates.qp.loc_head", "surrogates.qp.scale_head"}
            # the encoder still feeds the extent residual when that is on
            if not (_arg(a, cls, "gate_extent") and _arg(a, cls, "steer")):
                out.add("encoders.profile")
    return out


def _load(path: pathlib.Path) -> dict:
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(path.read_text())
    # a warm-start file on the cluster is not here; the architecture does not depend on it
    la = cfg.get("loss", {}).get("args", {}) or {}
    if la.get("spectrum_init_from") and not pathlib.Path(la["spectrum_init_from"]).exists():
        la["spectrum_init_from"] = None
    ia = cfg.get("integrator", {}).get("args", {}) or {}
    if ia.get("init_from_checkpoint") and not pathlib.Path(ia["init_from_checkpoint"]).exists():
        ia["init_from_checkpoint"] = None
    return cfg


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_every_built_parameter_trains_unless_the_config_says_otherwise(path):
    cfg = _load(path)
    torch.manual_seed(0)
    model = construct_integrator(cfg).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    reached = {n: False for n, p in model.named_parameters() if p.requires_grad}
    for _ in range(STEPS):
        opt.zero_grad()
        out = model._step(_batch(), step="train")
        loss = out["loss"] if isinstance(out, dict) else out
        assert torch.isfinite(loss), f"loss {float(loss)}"
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            assert torch.isfinite(p.grad).all(), f"non-finite gradient in {n}"
            if float(p.grad.abs().sum()) > 0:
                reached[n] = True
        opt.step()
    for n, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite parameter {n} after {STEPS} steps"

    expected = expected_unreached(cfg)
    unreached = sorted(n for n, ok in reached.items() if not ok)
    unexpected = [n for n in unreached if not n.startswith(tuple(expected))]
    assert not unexpected, f"parameters with no gradient in {STEPS} steps: {unexpected}"
    stale = sorted(g for g in expected if not any(n.startswith(g) for n in unreached))
    assert not stale, f"expected unreached but trained (rule out of date): {stale}"


def test_loss_parameters_are_never_on_the_unreached_list():
    """The learned prior lives in the loss; no rule may excuse it."""
    for path in CONFIGS:
        assert not any(g.startswith("loss") for g in expected_unreached(_load(path))), path.stem

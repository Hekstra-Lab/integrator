"""The prior a conjugate q is built from is the prior its KL is taken against.

The `svae` family builds q(I) and q(b) by a conjugate update, prior plus data,
so the prior appears twice: inside q, and in the loss's KL(q || p). The two are
built by different code -- the integrator's `_intensity_prior` and
`_background_prior`, and the loss's `forward` -- and nothing forced them to agree.
They did not for the background until 3cfe2000: q(b) read the fixed
`bg_concentration`, `bg_rate` while `polychromatic_wilson_bgmap` learned a prior
that only the KL saw, so the learned map could never move q(b), and the map's
parameters still received gradient (through the KL), so a gradient check passed.

For every sim_truth config in the family, one training step records both and
requires them equal element by element. A second test replaces the loss's prior
by a strong one with a different mean and requires q to move (q(b) towards the new mean; q(I) at all,
since `svae_conjugate` also builds its gate from it), which catches an
integrator that stops reading the prior at all.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

import integrator.model.loss.wilson_loss as wilson_loss
from integrator.utils.factory_utils import construct_integrator

from test_gradient_health import SVAE_FAMILY, _batch

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(path: pathlib.Path) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text())


CONFIGS = [
    p for p in sorted((ROOT / "configs" / "sim_truth").glob("*.yaml"))
    if _load(p)["integrator"]["name"] in SVAE_FAMILY
]
IDS = [p.stem for p in CONFIGS]


def _record(model, monkeypatch):
    """Wrap the integrator's prior and q builders and the loss's KL; return the log."""
    log = {"prior": {}, "q": {}, "kl": []}

    def wrap(name, key, slot):
        orig = getattr(model, name)

        def f(*a, **k):
            out = orig(*a, **k)
            log[slot][key] = out
            return out
        monkeypatch.setattr(model, name, f)

    wrap("_intensity_prior", "i", "prior")
    wrap("_background_prior", "bg", "prior")
    wrap("_get_qi", "i", "q")
    wrap("_get_qbg", "bg", "q")
    orig_kl = wilson_loss.kl_divergence

    def kl(q, p):
        log["kl"].append((q, p))
        return orig_kl(q, p)
    monkeypatch.setattr(wilson_loss, "kl_divergence", kl)
    return log


def _kl_prior_for(log, q):
    hits = [p for qq, p in log["kl"] if qq is q]
    assert len(hits) == 1, f"KL against this q taken {len(hits)} times"
    return hits[0]


def _same(a, b, shape):
    a = torch.as_tensor(a).expand(shape)
    b = torch.as_tensor(b).expand(shape)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=0)


@pytest.mark.parametrize("path", CONFIGS, ids=IDS)
def test_q_and_its_kl_use_the_same_prior(path, monkeypatch):
    torch.manual_seed(0)
    model = construct_integrator(_load(path)).train()
    log = _record(model, monkeypatch)
    model._step(_batch(), step="train")

    assert "bg" in log["prior"], "q(b) was built without the integrator's background prior"
    qbg = log["q"]["bg"]
    p = _kl_prior_for(log, qbg)
    conc, rate = log["prior"]["bg"]
    _same(conc, p.concentration, qbg.concentration.shape)
    _same(rate, p.rate, qbg.concentration.shape)

    if "i" in log["prior"]:  # absent only under intensity_posterior: amortized
        qi = log["q"]["i"]
        p = _kl_prior_for(log, qi)
        conc, rate = log["prior"]["i"]
        shape = torch.broadcast_shapes(torch.as_tensor(conc).shape, p.concentration.shape)
        _same(conc, p.concentration, shape)
        _same(rate, p.rate, shape)


@pytest.mark.parametrize("path", CONFIGS, ids=IDS)
def test_q_moves_when_the_loss_prior_moves(path, monkeypatch):
    torch.manual_seed(0)
    model = construct_integrator(_load(path)).eval()
    counts, shoebox, mask, metadata = _batch()
    with torch.no_grad():
        base = model(counts, shoebox, mask, metadata)
        loss = model.loss
        # a background prior with mean 50 photons/px and 1e4 pseudo-pixels
        monkeypatch.setattr(loss, "_background_prior",
                            lambda g, md, dev: (torch.full((counts.shape[0],), 1e4), torch.full((counts.shape[0],), 200.0)),
                            raising=False)
        moved_bg = model(counts, shoebox, mask, metadata)
        assert torch.all((moved_bg["qbg"].mean - 50).abs() < (base["qbg"].mean - 50).abs())

        if model.cfg.intensity_posterior != "amortized":
            orig = loss.intensity_prior_params

            def strong(md, dev):
                c, r = orig(md, dev)
                return c * 1e4, r * 1e4 / 1e-3  # mean 1e-3 photons, 1e4 pseudo-counts
            monkeypatch.setattr(loss, "intensity_prior_params", strong)
            moved_i = model(counts, shoebox, mask, metadata)
            # a change, not a direction: `svae_conjugate` computes its gate from
            # the same prior, so a low prior can hand some pixels to the signal
            assert not torch.allclose(moved_i["qi"].mean, base["qi"].mean, rtol=1e-3)

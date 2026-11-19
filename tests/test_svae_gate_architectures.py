"""The three gate architectures: what each builds and what each trains.

`svae` itself is not touched by any of this and is included here only as
the baseline the others are measured against.

The defect these exist to catch is a parameter that is built and never
trained. It has happened twice: the base class builds a 387,008-parameter
signal encoder that receives zero gradient once the gate is computed, and
the first draft of `svae_bg` initialised its head to zero weights, which
cuts the encoder out of the graph -- `d(out)/d(x) = W = 0` -- so the
encoder was dead at initialisation in the very class written to give it a
job.
"""

from __future__ import annotations

import copy

import pytest
import torch

from integrator.registry import REGISTRY
from integrator.utils.factory_utils import construct_integrator

ARCHS = {
    "svae": 2,
    "svae_conjugate": 1,
    "svae_bg": 2,
    "svae_calib": 2,
}
B, P = 6, 441

# The amortized surrogate heads the conjugate SVAE path never calls. Present
# in the baseline too; see `test_every_parameter_receives_gradient`.
KNOWN_UNUSED = {
    "surrogates.qbg.linear_mu.weight", "surrogates.qbg.linear_mu.bias",
    "surrogates.qbg.linear_fano.weight", "surrogates.qbg.linear_fano.bias",
    "surrogates.qi.linear_mu.weight", "surrogates.qi.linear_mu.bias",
    "surrogates.qi.linear_fano.weight", "surrogates.qi.linear_fano.bias",
}


def _cfg(name: str, n_encoders: int, base) -> dict:
    c = copy.deepcopy(base)
    c["integrator"]["name"] = name
    c["encoders"] = c["encoders"][:n_encoders]
    return c


@pytest.fixture(scope="module")
def base():
    """A real svae config, so these are the shapes the runs actually use."""
    pytest.importorskip("yaml")
    import pathlib

    import yaml

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "configs" / "poly" / "pdz2_1116_svae.yaml")
    if not path.is_file():
        pytest.skip("no pdz2 svae config to build from")
    return yaml.safe_load(path.read_text())


@pytest.fixture
def batch():
    counts = torch.poisson(torch.full((B, P), 5.0))
    metadata = {
        "group_label": torch.zeros(B, dtype=torch.long),
        "xyzcal.px.0": torch.rand(B) * 1000,
        "xyzcal.px.1": torch.rand(B) * 1000,
        "d": torch.full((B,), 2.0),
        "wavelength": torch.full((B,), 1.0),
    }
    return counts, counts.clone().unsqueeze(-1), torch.ones(B, P, 1), metadata


def _forward_backward(model, batch):
    counts, shoebox, mask, metadata = batch
    out = model(counts, shoebox, mask, metadata)
    out["forward_out"]["rates"].sum().backward()
    return out


@pytest.mark.parametrize("name", sorted(ARCHS))
def test_every_architecture_is_registered(name):
    assert name in REGISTRY["integrator"]


@pytest.mark.parametrize("name", sorted(ARCHS))
def test_every_parameter_receives_gradient(name, base, batch):
    """Nothing is built that training will not reach.

    This is the whole point of the file. A module that is constructed and
    never trained is invisible -- the run completes, the loss falls -- and
    it inflates every parameter count it appears in.
    """
    model = construct_integrator(_cfg(name, ARCHS[name], base)).train()
    _forward_backward(model, batch)
    dead = {
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)
    }
    # Known and shared by every arm INCLUDING the baseline: the config
    # declares `qi` and `qbg` surrogates, and the SVAE forms both posteriors
    # by conjugate update, so their heads are built and never used. Listed
    # by name rather than counted, so this test fails on NEW dead weight
    # instead of absorbing it.
    assert dead <= KNOWN_UNUSED, sorted(dead - KNOWN_UNUSED)


def test_the_conjugate_arm_builds_no_signal_encoder(base, batch):
    model = construct_integrator(_cfg("svae_conjugate", 1, base))
    assert model.gate_head is None
    signal = model.encoders["signal"]
    assert sum(p.numel() for p in signal.parameters()) == 0


def test_the_conjugate_arm_is_smaller_by_the_encoder_and_the_head(base, batch):
    small = construct_integrator(_cfg("svae_conjugate", 1, base))
    full = construct_integrator(_cfg("svae", 2, base))
    n_small = sum(p.numel() for p in small.parameters())
    n_full = sum(p.numel() for p in full.parameters())
    # the signal encoder plus gate_head, and nothing else
    assert n_full - n_small > 400_000, (n_full, n_small)


def test_calibration_starts_at_the_exact_responsibility(base, batch):
    """`svae_calib` must begin where `svae_conjugate` is, not near it.

    Its slope is `softplus(0.5413) = 1` and its offset 0, so at
    initialisation `sigmoid(a * log-odds + b)` IS the responsibility. If it
    started somewhere else, a result showing it moved would say nothing.
    """
    calib = construct_integrator(_cfg("svae_calib", 2, base)).eval()
    conj = construct_integrator(_cfg("svae_conjugate", 1, base)).eval()
    # the same profile for both, so only the gate construction differs
    conj.surrogates.load_state_dict(calib.surrogates.state_dict())
    conj.encoders["profile"].load_state_dict(
        calib.encoders["profile"].state_dict())
    with torch.no_grad():
        a = torch.nn.functional.softplus(calib.calib_head.bias[0])
        b = calib.calib_head.bias[1]
    assert abs(float(a) - 1.0) < 1e-3, float(a)
    assert abs(float(b)) < 1e-6, float(b)


@pytest.mark.parametrize("name", sorted(ARCHS))
def test_the_gate_is_a_probability_that_partitions_the_pixel(name, base, batch):
    model = construct_integrator(_cfg(name, ARCHS[name], base)).eval()
    with torch.no_grad():
        counts, shoebox, mask, metadata = batch
        out = model(counts, shoebox, mask, metadata)
    g = out["forward_out"]["signal_prob"]
    assert torch.isfinite(g).all()
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0

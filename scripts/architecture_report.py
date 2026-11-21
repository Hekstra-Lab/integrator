"""Architecture of a config as training builds it: modules, parameter counts, and whether each gets gradient.

Builds the integrator through the same sequence as `integrator.cli.train`
(load_config, apply_dataset_defaults, resolve_config, the prior preparation,
construct_data_loader + setup + inject_binning_labels + assign_labels,
construct_integrator), takes ONE training batch from the real dataset, runs
`training_step` and `backward`, and reports per module

    parameters      number of scalars
    grad_norm       L2 norm of the gradient over its parameters after one step
    frac_nonzero    fraction of its scalars with a nonzero gradient

A module with grad_norm 0 is built but cut out of the objective at
initialisation; that has happened twice in this repository (a signal encoder
past the gate, a zero-initialised head) and shows here before any allocation
is spent. One batch can miss a module that is reached only on some batches,
so frac_nonzero near 0 with grad_norm > 0 is worth a second batch.

Writes a Markdown report (the resolved integrator, encoder and surrogate
blocks, the per-module table at depth 1 and 2) to --out and prints it.

    python scripts/architecture_report.py --config configs/sim_truth/mos010_all.yaml --out report.md
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
import yaml


def build(config: str):
    from integrator.utils import (
        apply_dataset_defaults,
        construct_data_loader,
        construct_integrator,
        inject_binning_labels,
        load_config,
        prepare_bg_prior,
        prepare_global_priors,
        prepare_per_bin_priors,
        prepare_profile_basis,
        resolve_config,
    )
    from integrator.callbacks.prediction_writer import assign_labels

    cfg = resolve_config(apply_dataset_defaults(load_config(config)))
    prepare_profile_basis(cfg, events_out=[])
    prepare_bg_prior(cfg, events_out=[])
    prepare_per_bin_priors(cfg)
    prepare_global_priors(cfg)
    dm = construct_data_loader(cfg)
    dm.setup()
    inject_binning_labels(dm, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        assign_labels(dataset=dm, save_dir=tmp)
    return cfg, dm, construct_integrator(cfg)


def module_rows(model, depth: int):
    rows = []
    for name, mod in model.named_modules():
        if not name or name.count(".") != depth - 1:
            continue
        ps = [p for p in mod.parameters() if p.requires_grad]
        n = sum(p.numel() for p in ps)
        if n == 0:
            continue
        g2 = sum(float((p.grad**2).sum()) for p in ps if p.grad is not None)
        nz = sum(int((p.grad != 0).sum()) for p in ps if p.grad is not None)
        rows.append((name, type(mod).__name__, n, g2**0.5, nz / n))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=None, help="override the config's batch size")
    a = ap.parse_args()

    torch.manual_seed(0)
    cfg, dm, model = build(a.config)
    if a.batch_size:
        dm.batch_size = a.batch_size
    batch = next(iter(dm.train_dataloader()))
    model.train()
    out = model.training_step(batch, 0)
    loss = out["loss"] if isinstance(out, dict) else out
    model.zero_grad()
    loss.backward()

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [f"# Architecture: `{a.config}`", "",
             f"Integrator `{type(model).__name__}`, {total:,} trainable parameters; one training batch, "
             f"loss {float(loss):.4g}.", ""]
    for depth in (1, 2):
        lines += [f"## Modules at depth {depth}", "",
                  "| module | class | parameters | grad_norm | frac_nonzero |", "|---|---|---|---|---|"]
        for name, cls, n, g, f in module_rows(model, depth):
            flag = " **NO GRADIENT**" if g == 0 else ""
            lines.append(f"| `{name}` | {cls} | {n:,} | {g:.3g}{flag} | {f:.3f} |")
        lines.append("")
    blocks = {k: cfg.get(k) for k in ("integrator", "encoders", "surrogates", "loss", "optimizer", "trainer")}
    lines += ["## Resolved config blocks", "", "```yaml", yaml.safe_dump(blocks, sort_keys=False).rstrip(), "```", ""]
    text = "\n".join(lines)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()

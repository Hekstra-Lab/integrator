"""Train (or verify) the three-arm JUNGFRAU likelihood comparison.

Arms, all sharing the 5-encoder hierarchical integrator + global_prior loss:

  poisson         Poisson likelihood on rounded integer counts   (data/jf_train_int)
  normal_coupled  Normal, Var = rate + read_noise^2, on real data (data/jf_train_real)
  normal_free     Normal, one learned sigma, on real data         (data/jf_train_real)

They are identical except for the likelihood and the data directory, so any gap
between them is the likelihood choice and nothing else.

Two modes:

  --verify  (default) construct each arm end-to-end and run ONE forward batch to prove
            the config resolves and the loss is finite -- no optimizer step, no training.
            Runs on CPU in seconds. This is the "prep" deliverable.

  --run [ARM ...]  actually train the named arms (default: all three) via the same
            orchestration the `integrator.train` CLI uses. Point `trainer.accelerator`
            at `gpu` (edit the configs) before running this on a cluster.

Prerequisites (run once):
  uv run python scripts/jungfrau_sim/generate.py --n 20000 --out data/jf_sim
  uv run python scripts/jungfrau_sim/prepare_training_data.py --sim data/jf_sim --out data/jf_train

Run:
  uv run python scripts/jungfrau_sim/train_likelihood_comparison.py            # verify all
  uv run python scripts/jungfrau_sim/train_likelihood_comparison.py --run      # train all
  uv run python scripts/jungfrau_sim/train_likelihood_comparison.py --run poisson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from integrator.utils import (
    apply_dataset_defaults,
    construct_data_loader,
    construct_integrator,
    construct_trainer,
    inject_binning_labels,
    load_config,
    prepare_global_priors,
    prepare_per_bin_priors,
    resolve_config,
)

CONFIG_DIR = Path("configs/jungfrau")
ARMS = {
    "poisson": "jf_poisson.yaml",
    "normal_coupled": "jf_normal_coupled.yaml",
    "normal_free": "jf_normal_free.yaml",
}


def _prepare(cfg: dict) -> dict:
    """The construction half of the train CLI: config -> data loader + model.

    Mirrors `integrator.cli.train.main` up to (not including) `trainer.fit`.
    """
    cfg = apply_dataset_defaults(cfg)
    cfg = resolve_config(cfg)
    prepare_per_bin_priors(cfg)   # no-op unless a Wilson loss is selected
    prepare_global_priors(cfg)    # fits the global Gamma priors; writes group_labels_1
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)
    model = construct_integrator(cfg)
    return {"cfg": cfg, "data_loader": data_loader, "model": model}


def verify(arm: str) -> bool:
    """Construct the arm and run one forward batch. Returns True on a finite loss."""
    print(f"\n=== {arm}  ({ARMS[arm]}) ===")
    cfg = load_config(CONFIG_DIR / ARMS[arm])
    built = _prepare(cfg)
    model, data_loader = built["model"], built["data_loader"]

    likelihood = built["cfg"]["loss"]["args"].get("likelihood", "poisson")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  likelihood       : {likelihood}")
    print(f"  data_dir         : {built['cfg']['data_loader']['args']['data_dir']}")
    print(f"  trainable params : {n_params:,}")

    train_loader = data_loader.train_dataloader()
    batch = next(iter(train_loader))
    model.train()
    # One forward + loss only -- no backward, no optimizer step, no epoch.
    with torch.no_grad():
        out = model.training_step(batch, 0)
    loss = out["loss"] if isinstance(out, dict) else out
    finite = bool(torch.isfinite(loss).all())
    print(f"  one-batch loss   : {float(loss):.4f}   finite={finite}")
    return finite


def train(arm: str, max_epochs: int | None) -> None:
    print(f"\n=== training {arm}  ({ARMS[arm]}) ===")
    cfg = load_config(CONFIG_DIR / ARMS[arm])
    if max_epochs is not None:
        cfg.setdefault("trainer", {})["max_epochs"] = max_epochs
    built = _prepare(cfg)
    trainer = construct_trainer(built["cfg"])
    dl = built["data_loader"]
    trainer.fit(
        built["model"],
        train_dataloaders=dl.train_dataloader(),
        val_dataloaders=dl.val_dataloader(),
    )
    print(f"  done: {arm}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", nargs="*", metavar="ARM", default=None,
        help="train the named arms (default: all three). Omit to only verify.",
    )
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override trainer.max_epochs (e.g. a short local run)")
    args = ap.parse_args()

    if args.run is None:
        print("VERIFY MODE -- constructing each arm and running one forward batch.")
        print("(no training; pass --run to train). See docstring for the cluster path.")
        ok = {arm: verify(arm) for arm in ARMS}
        print("\nsummary")
        for arm, good in ok.items():
            print(f"  {arm:15} {'OK' if good else 'FAILED'}")
        print("\nto train on a GPU cluster, set trainer.accelerator: gpu in the configs, then:")
        for fn in ARMS.values():
            print(f"  uv run integrator.train --config {CONFIG_DIR / fn}")
        print("or all three via this script:  ... train_likelihood_comparison.py --run")
        raise SystemExit(0 if all(ok.values()) else 1)

    arms = args.run or list(ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        ap.error(f"unknown arm(s) {unknown}; choose from {list(ARMS)}")
    for arm in arms:
        train(arm, args.max_epochs)


if __name__ == "__main__":
    main()

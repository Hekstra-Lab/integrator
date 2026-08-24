"""Convert simulated JUNGFRAU shoeboxes into integrator-trainable datasets.

Emits two dataset directories from one `generate.py` run, because the Normal and
Poisson arms need different pixel values:

  <out>_real/  counts = counts_real     (real-valued, negative on ~28% of pixels)
                                          -> the Normal-likelihood arms
  <out>_int/   counts = counts_poisson  (rounded, non-negative integers)
                                          -> the Poisson arm

Both carry the SAME latent truth in metadata, so the two arms are scored against
identical ground truth and differ only in (pixel values, likelihood).

Each directory gets the manifest the data module requires:
  counts.npy   (N, h*w) float32   flat-voxel
  masks.npy    (N, h*w) bool      all-valid
  metadata.npy pickled dict       ground truth + the keys the pipeline needs
  dataset.yaml geometry + files + stats, anscombe:false -> `standardization`

Why `standardization`, not Anscombe: Anscombe is 2*sqrt(c + 3/8), NaN for the
negative pixels in counts_real. Standardization ((c-mean)/std) is finite on the real
line, and using it for BOTH arms keeps the encoder input identical so the only
difference between arms is the likelihood.

Metadata keys and why each is here (see the pipeline requirements):
  intensity.prf.variance  REQUIRED -- `_remove_flagged_variance` filters on it; set 0.
  d                       REQUIRED by the prepare step (value unused by global_prior);
                          placeholder ones.
  intensity.sum.value     the global intensity Gamma prior is fit to this -> our I.
  background.mean         the global background Gamma prior is fit to this -> our B.
  refl_ids                stable ids, handy for prediction.
`group_label` is NOT written here -- `prepare_global_priors` + `inject_binning_labels`
generate it at train time.

Run:  uv run python scripts/jungfrau_sim/prepare_training_data.py \
          --sim data/jf_sim --out data/jf_train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from integrator.io.dataset import write_dataset_yaml


def _stats(counts: np.ndarray, mask: np.ndarray) -> dict:
    """[mean, var] over masked pixels, for both transforms the module might select."""
    valid = counts[mask]
    raw = [float(valid.mean()), float(valid.var())]
    # Anscombe is only meaningful where counts >= -3/8; clamp so the number exists even
    # for the real-valued arm, which never actually selects this key (anscombe:false).
    ansc = 2.0 * np.sqrt(np.clip(valid + 0.375, 0.0, None))
    return {"raw": raw, "anscombe": [float(ansc.mean()), float(ansc.var())]}


def write_dataset(
    out: Path,
    counts: np.ndarray,
    sim: dict[str, np.ndarray],
    h: int,
    w: int,
) -> None:
    """Write one dataset directory (counts + masks + metadata + dataset.yaml)."""
    n = counts.shape[0]
    out.mkdir(parents=True, exist_ok=True)

    counts = counts.astype(np.float32)
    mask = np.ones((n, h * w), dtype=bool)

    metadata = {
        # Required by the pipeline (see module docstring).
        "intensity.prf.variance": np.zeros(n, dtype=np.float32),
        "d": np.ones(n, dtype=np.float32),
        # The global Gamma priors are fit to these -> recovers our Exp(mean) generator.
        "intensity.sum.value": sim["intensity"].astype(np.float32),
        "background.mean": sim["background"].astype(np.float32),
        # Ground truth, carried through for scoring predictions later.
        "intensity.true": sim["intensity"].astype(np.float32),
        "background.true": sim["background"].astype(np.float32),
        "refl_ids": np.arange(n, dtype=np.int64),
    }

    np.save(out / "counts.npy", counts)
    np.save(out / "masks.npy", mask)
    np.save(out / "metadata.npy", metadata, allow_pickle=True)

    write_dataset_yaml(
        out,
        geometry={"d": 1, "h": h, "w": w, "data_dim": "2d"},
        n_reflections=n,
        polychromatic=False,
        anscombe=False,  # -> transform "standardization"; finite on negative pixels
        files={"counts": "counts.npy", "masks": "masks.npy", "reference": "metadata.npy"},
        stats=_stats(counts, mask),
    )
    print(f"  wrote {out}/  ({n} shoeboxes, counts range "
          f"[{counts.min():.2f}, {counts.max():.1f}])")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", type=Path, default=Path("data/jf_sim"),
                    help="a generate.py output directory")
    ap.add_argument("--out", type=Path, default=Path("data/jf_train"),
                    help="prefix; writes <out>_real and <out>_int")
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--w", type=int, default=20)
    args = ap.parse_args()

    keys = ["counts_real", "counts_poisson", "intensity", "background"]
    sim = {k: np.load(args.sim / f"{k}.npy") for k in keys}

    print(f"preparing training datasets from {args.sim}/")
    write_dataset(Path(f"{args.out}_real"), sim["counts_real"], sim, args.h, args.w)
    write_dataset(Path(f"{args.out}_int"), sim["counts_poisson"], sim, args.h, args.w)
    print("done. point configs at <out>_real (Normal arms) and <out>_int (Poisson).")


if __name__ == "__main__":
    main()

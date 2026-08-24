"""Compare two refined PDBs against a reference.

Computes per-atom and per-residue RMSD, B-factor correlation,
and coordinate shift statistics. Useful for comparing refinements
from shaken vs unshaken starting models.

Usage
-----
    uv run python scripts/compare_pdbs.py \
        --ref reference.pdb \
        --pdb1 refined_unshaken.pdb \
        --pdb2 refined_shaken.pdb \
        --out comparison.png
"""

import argparse
from pathlib import Path

import gemmi
import matplotlib.pyplot as plt
import numpy as np


def extract_atoms(pdb_path: str) -> dict:
    st = gemmi.read_structure(pdb_path)
    atoms = {}
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    key = (chain.name, str(residue.seqid), atom.name)
                    atoms[key] = {
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z]),
                        "b": atom.b_iso,
                        "element": atom.element.name,
                        "resname": residue.name,
                        "seqid": residue.seqid.num,
                    }
    return atoms


def compare(ref_atoms, test_atoms, label):
    common = set(ref_atoms.keys()) & set(test_atoms.keys())
    if not common:
        print(f"  {label}: no common atoms!")
        return None

    pos_ref = np.array([ref_atoms[k]["pos"] for k in common])
    pos_test = np.array([test_atoms[k]["pos"] for k in common])
    b_ref = np.array([ref_atoms[k]["b"] for k in common])
    b_test = np.array([test_atoms[k]["b"] for k in common])
    seqids = np.array([ref_atoms[k]["seqid"] for k in common])

    dists = np.linalg.norm(pos_ref - pos_test, axis=1)
    rmsd = np.sqrt((dists**2).mean())

    b_corr = np.corrcoef(b_ref, b_test)[0, 1] if b_ref.std() > 0 else 0.0

    print(f"  {label}:")
    print(f"    N atoms: {len(common)}")
    print(f"    RMSD: {rmsd:.4f} A")
    print(f"    Mean shift: {dists.mean():.4f} A")
    print(f"    Max shift: {dists.max():.4f} A (at {max(common, key=lambda k: np.linalg.norm(ref_atoms[k]['pos'] - test_atoms[k]['pos']))})")
    print(f"    B-factor correlation: {b_corr:.4f}")
    print(f"    B ref: mean={b_ref.mean():.1f}, range=[{b_ref.min():.1f}, {b_ref.max():.1f}]")
    print(f"    B test: mean={b_test.mean():.1f}, range=[{b_test.min():.1f}, {b_test.max():.1f}]")

    return {
        "dists": dists,
        "b_ref": b_ref,
        "b_test": b_test,
        "seqids": seqids,
        "rmsd": rmsd,
        "b_corr": b_corr,
        "label": label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, required=True, help="Reference PDB")
    parser.add_argument("--pdb1", type=str, required=True, help="First refined PDB")
    parser.add_argument("--pdb2", type=str, default=None, help="Second refined PDB (optional)")
    parser.add_argument("--out", type=str, default="pdb_comparison.png")
    args = parser.parse_args()

    ref_atoms = extract_atoms(args.ref)
    pdb1_atoms = extract_atoms(args.pdb1)

    print(f"Reference: {args.ref} ({len(ref_atoms)} atoms)")
    r1 = compare(ref_atoms, pdb1_atoms, Path(args.pdb1).stem)

    r2 = None
    if args.pdb2:
        pdb2_atoms = extract_atoms(args.pdb2)
        r2 = compare(ref_atoms, pdb2_atoms, Path(args.pdb2).stem)

        # Compare the two refined models against each other
        compare(pdb1_atoms, pdb2_atoms, f"{Path(args.pdb1).stem} vs {Path(args.pdb2).stem}")

    # Plot
    n_panels = 3 if r2 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    results = [r1] if r2 is None else [r1, r2]

    # Panel 1: coordinate shift histogram
    ax = axes[0]
    for r in results:
        ax.hist(r["dists"], bins=50, alpha=0.6, label=f'{r["label"]} (RMSD={r["rmsd"]:.3f} A)')
    ax.set_xlabel("Coordinate shift from reference (A)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Coordinate shifts")

    # Panel 2: B-factor correlation
    ax = axes[1]
    for r in results:
        ax.scatter(r["b_ref"], r["b_test"], s=1, alpha=0.3, label=f'{r["label"]} (r={r["b_corr"]:.3f})')
    lim = max(r["b_ref"].max(), r["b_test"].max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", linewidth=0.5)
    ax.set_xlabel("B-factor (reference)")
    ax.set_ylabel("B-factor (refined)")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("B-factor correlation")

    # Panel 3: per-residue RMSD (if two models)
    if r2 is not None:
        ax = axes[2]
        for r in results:
            unique_seq = np.unique(r["seqids"])
            per_res = []
            for s in unique_seq:
                mask = r["seqids"] == s
                per_res.append(np.sqrt((r["dists"][mask] ** 2).mean()))
            ax.plot(unique_seq, per_res, alpha=0.7, label=r["label"])
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Per-residue RMSD (A)")
        ax.legend()
        ax.set_title("Per-residue RMSD vs reference")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

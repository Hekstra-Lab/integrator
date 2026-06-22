"""Anomalous, refinement, and scatter plots for the merging checkpoint eval.

Self-contained port of the refltorch plotting conventions (refltorch is not
importable from the integrator envs): a y=x scatter primitive, R-work/R-free vs
epoch, metric-vs-epoch curves, and the anomalous peak height at known anomalous
atom sites (the ANOM/PHANOM difference map sampled at the PDB positions, in
sigma -- the real anomalous metric, vs find_peaks' top peak). matplotlib only.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]


def _palette(keys) -> dict:
    keys = list(dict.fromkeys(keys))
    return {k: _COLORS[i % len(_COLORS)] for i, k in enumerate(keys)}


def save_figure(fig, path, dpi: int = 200) -> Path:
    """Save with the project's standard options and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def plot_scatter_identity(
    x,
    y,
    *,
    xlabel=None,
    ylabel=None,
    title=None,
    log=False,
    figsize=(5, 5),
    s=4,
    alpha=0.3,
):
    """Scatter x against y with a y=x reference line (model-vs-reference)."""
    import numpy as np

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if log:
        keep &= (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=s, alpha=alpha, c="black", edgecolors="none")
    if len(x):
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], c="red", alpha=0.6, lw=1, label="y = x")
        if len(x) > 2:
            xl = np.log(x) if log else x
            yl = np.log(y) if log else y
            cc = float(np.corrcoef(xl, yl)[0, 1])
            ax.set_title(
                (title + "  " if title else "")
                + ("log-CC" if log else "CC")
                + f"={cc:.3f}  n={len(x)}"
            )
    elif title:
        ax.set_title(title)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    return fig, ax


def plot_r_values_vs_epoch(rows, *, ref_vals=None, title=None, figsize=(6, 4)):
    """R-work (solid) and R-free (dashed) vs epoch, one color per variant.

    `rows`: list of dicts with epoch, variant, r_work, r_free.
    `ref_vals`: optional {r_work, r_free} drawn as horizontal reference lines.
    """
    variants = sorted({r["variant"] for r in rows if r.get("variant")})
    pal = _palette(variants)
    fig, ax = plt.subplots(figsize=figsize)
    for v in variants:
        pts = sorted(
            (r["epoch"], r.get("r_work"), r.get("r_free"))
            for r in rows
            if r["variant"] == v
        )
        ep = [p[0] for p in pts]
        for idx, ls, name in ((1, "-", "r_work"), (2, "--", "r_free")):
            ys = [p[idx] for p in pts]
            xy = [(e, y) for e, y in zip(ep, ys) if isinstance(y, (int, float))]
            if xy:
                ax.plot(
                    [a for a, _ in xy], [b for _, b in xy], ls,
                    marker="o", ms=3, color=pal[v], label=f"{v} {name}",
                )
    if ref_vals:
        if ref_vals.get("r_work") is not None:
            ax.axhline(ref_vals["r_work"], color="#888", lw=1, label="ref r_work")
        if ref_vals.get("r_free") is not None:
            ax.axhline(
                ref_vals["r_free"], color="#888", lw=1, ls="--",
                label="ref r_free",
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel("R value")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    return fig, ax


def plot_metric_vs_epoch(
    rows, key, *, ylabel=None, title=None, figsize=(6, 4)
):
    """A per-variant metric (e.g. top_anom_peak) vs epoch."""
    variants = sorted({r["variant"] for r in rows if r.get("variant")})
    pal = _palette(variants)
    fig, ax = plt.subplots(figsize=figsize)
    for v in variants:
        pts = sorted(
            (r["epoch"], r.get(key))
            for r in rows
            if r["variant"] == v and isinstance(r.get(key), (int, float))
        )
        if pts:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=3, color=pal[v], label=v,
            )
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel or key)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def plot_anom_sites_vs_epoch(site_rows, *, title=None, figsize=(6, 4)):
    """Per-site anomalous peak height (sigma) vs epoch.

    `site_rows`: list of dicts {epoch, site, height}.
    """
    sites = sorted({r["site"] for r in site_rows})
    pal = _palette(sites)
    fig, ax = plt.subplots(figsize=figsize)
    for site in sites:
        pts = sorted(
            (r["epoch"], r["height"])
            for r in site_rows
            if r["site"] == site and isinstance(r.get("height"), (int, float))
        )
        if pts:
            ax.plot(
                [p[0] for p in pts], [p[1] for p in pts],
                marker="o", ms=3, color=pal[site], label=site,
            )
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("anomalous peak height (sigma)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig, ax


def get_anom_peak_heights(
    mtz_filename, pdb_filename, atom_sel, f_col=None, phi_col=None
):
    """Anomalous-difference-map peak height (sigma) at each anomalous atom site.

    Builds the anomalous-difference map (normalized to sigma), then samples it at
    each selected atom and its symmetry equivalents (averaged). The amplitude /
    phase column names are auto-detected (phenix names them ANOM/PANOM; older
    versions ANOM/PHANOM), or pass `f_col`/`phi_col` to override. Returns
    `(labels, heights)`. Ported from refltorch's anomalous_peak_heights.
    """
    import gemmi
    import numpy as np

    mtz = gemmi.read_mtz_file(str(mtz_filename))
    st = gemmi.read_pdb(str(pdb_filename))

    labels_present = {c.label for c in mtz.columns}

    def _pick(given, candidates):
        if given and given in labels_present:
            return given
        for c in candidates:
            if c in labels_present:
                return c
        return None

    f_col = _pick(f_col, ("ANOM", "ANOMALOUS", "DANO"))
    phi_col = _pick(phi_col, ("PANOM", "PHANOM", "PHIANOM", "PHANOMALOUS"))
    if f_col is None or phi_col is None:
        raise KeyError(
            "no anomalous amplitude/phase columns in "
            f"{mtz_filename}; have {sorted(labels_present)}"
        )

    grid = mtz.transform_f_phi_to_map(f_col, phi_col, sample_rate=3.0)
    grid.normalize()

    sel = gemmi.Selection(str(atom_sel))
    atoms = list(sel.copy_model_selection(st[0]).all())

    labels, heights = [], []
    ops = grid.spacegroup.operations()
    for cra in atoms:
        vals = []
        for op in ops:
            m = op.apply_to_xyz(st.cell.fractionalize(cra.atom.pos).tolist())
            frac = np.array(m) - np.floor(np.array(m))
            vals.append(
                grid.get_value(
                    round(frac[0] * grid.nu),
                    round(frac[1] * grid.nv),
                    round(frac[2] * grid.nw),
                )
            )
        labels.append(f"{cra.residue.name}{cra.residue.seqid.num}")
        heights.append(round(float(np.average(vals)), 3))
    return labels, heights


def friedel_scatter_from_mtz(mtz_path, *, title=None):
    """F(+) vs F(-) scatter from a merged anomalous MTZ (the anomalous signal)."""
    import reciprocalspaceship as rs

    ds = rs.read_mtz(str(mtz_path))
    cols = {"F(+)", "F(-)"} if {"F(+)", "F(-)"}.issubset(ds.columns) else None
    if cols is None:
        return None, None
    pair = ds[["F(+)", "F(-)"]].dropna()
    return plot_scatter_identity(
        pair["F(+)"].to_numpy(),
        pair["F(-)"].to_numpy(),
        xlabel="F(+)",
        ylabel="F(-)",
        title=title or "Friedel pairs",
        log=True,
    )

"""Check the PyTorch nanoBragg kernel against the C program, pixel for pixel.

The C program must be built from `nanoBragg.c` with a state dump inserted after
`steps` is computed (see `--help`), so that the geometry, sources and mosaic
domains it actually used are read back rather than re-derived. The crystal is the
DHFR simulation's first shot (Umat, Bmat from its `.h5`) on that dataset's
detector, so the test runs at a realistic geometry.

Run:
    uv run python scripts/physics/validate_nanobragg_port.py --nanobragg PATH/nanoBragg_dump
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from integrator.physics.nanobragg import Beam, Crystal, Detector, render_pixels

# DHFR simulation, hkl_data/shot_1_00001.h5 and its CBF header.
UMAT = np.array([0.75829879, -0.32153556, 0.56709596, 0.62673815, 0.12018753, -0.76990535,
                 0.17939408, 0.93923897, 0.29265666]).reshape(3, 3)
BMAT = np.array([3.01592771e-02, 0.0, 0.0, 0.0, 2.27438849e-02, 0.0,
                 0.0, 0.0, 1.04660210e-02]).reshape(3, 3)
PIXEL_MM = 0.08854
N_PIX = 3840
DISTANCE_MM = 200.4980206688427
BEAM_MM = (175.98251986641733, 173.98183043768444)
LAMBDA = 1.05

SHAPE_CODE = {0: "square", 1: "round", 2: "gauss", 3: "tophat"}


def predicted_spot(A: np.ndarray, lam: float, dmin: float = 2.5) -> tuple[int, int]:
    """Pixel of the reflection closest to Bragg at `lam`, on this detector."""
    r = np.arange(-15, 16)
    hkl = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    hkl = hkl[np.abs(hkl).sum(1) > 0]
    q = hkl @ A.T
    d = 1.0 / np.linalg.norm(q, axis=1)
    s0 = np.array([0.0, 0.0, -1.0 / lam])
    s1 = q + s0
    eps = np.abs(np.linalg.norm(s1, axis=1) - 1.0 / lam)
    ok = (d > dmin) & (s1[:, 2] < -0.2 / lam)
    t = -DISTANCE_MM / s1[:, 2]
    f = (s1[:, 0] * t + BEAM_MM[0]) / PIXEL_MM
    s = (BEAM_MM[1] - s1[:, 1] * t) / PIXEL_MM
    ok &= (f > 100) & (f < N_PIX - 100) & (s > 100) & (s < N_PIX - 100)
    i = np.flatnonzero(ok)[np.argmin(eps[ok])]
    return int(round(f[i])), int(round(s[i]))


def run_c(binary: Path, workdir: Path, extra: list[str], roi: tuple[int, int, int, int]):
    A = UMAT @ BMAT
    mat = workdir / "A.mat"
    np.savetxt(mat, A * LAMBDA, fmt="%.17g")
    state = workdir / "nb_state.txt"
    floatfile = workdir / "float.bin"
    cmd = [
        str(binary), "-mat", str(mat), "-default_F", "1", "-nointerpolate",
        "-lambda", str(LAMBDA), "-distance", str(DISTANCE_MM), "-pixel", str(PIXEL_MM),
        "-detpixels", str(N_PIX), "-Xbeam", str(BEAM_MM[0]), "-Ybeam", str(BEAM_MM[1]),
        "-fdet_vector", "1", "0", "0", "-sdet_vector", "0", "-1", "0",
        "-odet_vector", "0", "0", "-1", "-beam_vector", "0", "0", "-1",
        "-roi", *map(str, roi), "-floatfile", str(floatfile), "-nonoise", "-nopgm",
        "-intfile", str(workdir / "int.img"), "-noprogress", *extra,
    ]
    out = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                         env={"NB_STATE": str(state)})
    if out.returncode != 0:
        raise RuntimeError(out.stdout[-2000:] + out.stderr[-2000:])
    image = np.fromfile(floatfile, dtype=np.float32).reshape(N_PIX, N_PIX)
    return image, parse_state(state)


def parse_state(path: Path) -> dict:
    st: dict = {"source": [], "umat": []}
    for line in path.read_text().splitlines():
        key, *vals = line.split()
        nums = [float(v) for v in vals]
        if key in ("source", "umat"):
            st[key].append(nums)
        else:
            st[key] = nums if len(nums) > 1 else nums[0]
    return st


def render_torch(st: dict, roi: tuple[int, int, int, int]) -> np.ndarray:
    t = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    x0, x1, y0, y1 = roi
    ss, ff = np.meshgrid(np.arange(y0, y1 + 1), np.arange(x0, x1 + 1), indexing="ij")
    detector = Detector(
        pix0=t(st["pix0"]), fdet=t(st["fdet"]), sdet=t(st["sdet"]), odet=t(st["odet"]),
        pixel_size=st["pixel_size"],
    )
    src = np.array(st["source"])
    beam = Beam(
        incident=-t(src[:, :3]), wavelength=t(src[:, 3] * 1e10), fluence=st["fluence"],
        kahn=st["polarization"], polar_vector=t(st["polar_vector"]), nopolar=bool(st["nopolar"]),
    )
    cell = t(np.array(st["ap"]).reshape(3, 3) * 1e10)
    umats = t(np.array(st["umat"]).reshape(-1, 3, 3))
    cells = torch.einsum("dij,kj->dki", umats, cell)
    crystal = Crystal(
        cells=cells, recip=t(np.array(st["astar"]).reshape(3, 3)), n_cells=t(st["N"]),
        shape=SHAPE_CODE[int(st["shape"])], fudge=st["fudge"],
    )
    out = render_pixels(
        torch.from_numpy(ff.ravel()), torch.from_numpy(ss.ravel()), detector, beam, crystal,
        oversample=int(st["oversample"]), point_pixel=bool(st["point_pixel"]),
    )
    return out.numpy().reshape(ss.shape)


def write_sources(workdir: Path) -> Path:
    path = workdir / "sources.txt"
    rows = []
    for dx, dy in [(0.0, 0.0), (2e-3, 0.0), (0.0, -1.5e-3)]:
        for dl in (-0.004, 0.0, 0.003):
            rows.append([dx * 10, dy * 10, 10.0, 1.0, LAMBDA * (1 + dl) * 1e-10])
    np.savetxt(path, rows, fmt="%.12g")
    return path


CASES = [
    ("square N=5 no-polar", ["-N", "5", "-nopolar"]),
    ("square N=8,6,4 oversample 3", ["-Na", "8", "-Nb", "6", "-Nc", "4", "-oversample", "3"]),
    ("gauss N=10 kahn 0.9 oversample 2", ["-N", "10", "-gauss_xtal", "-polar", "0.9", "-oversample", "2"]),
    ("round N=6,7,8", ["-Na", "6", "-Nb", "7", "-Nc", "8", "-round_xtal"]),
    ("tophat N=7", ["-N", "7", "-tophat_spots"]),
    ("square N=5 point_pixel", ["-N", "5", "-point_pixel"]),
    ("gauss N=8 mosaic 0.3deg x 25", ["-N", "8", "-gauss_xtal", "-mosaic", "0.3", "-mosaic_domains", "25"]),
    ("square N=6 mosaic 0.2deg x 20 + 9 sources", ["-N", "6", "-mosaic", "0.2", "-mosaic_domains", "20", "SOURCES"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nanobragg", type=Path, required=True)
    ap.add_argument("--half", type=int, default=30, help="ROI half-width in pixels")
    args = ap.parse_args()

    fc, sc = predicted_spot(UMAT @ BMAT, LAMBDA)
    roi = (fc - args.half, fc + args.half, sc - args.half, sc + args.half)
    print(f"ROI centred on predicted spot at fast={fc}, slow={sc}: {roi}")
    print(f"{'case':45s} {'max |rel err|':>14s} {'sum ratio':>12s} {'peak photons':>14s}")
    worst = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name, extra in CASES:
            if "SOURCES" in extra:
                i = extra.index("SOURCES")
                extra = extra[:i] + ["-sourcefile", str(write_sources(work))] + extra[i + 1 :]
            img, st = run_c(args.nanobragg, work, extra, roi)
            c = img[roi[2] : roi[3] + 1, roi[0] : roi[1] + 1].astype(np.float64)
            p = render_torch(st, roi)
            peak = c.max()
            sel = c > 1e-6 * peak
            rel = np.abs(p[sel] - c[sel]) / c[sel]
            worst = max(worst, rel.max())
            print(f"{name:45s} {rel.max():14.3e} {p.sum() / c.sum():12.8f} {peak:14.4g}")
    print(f"\nworst relative error over all cases: {worst:.3e} (C writes float32, so ~1e-7 is exact)")


if __name__ == "__main__":
    main()

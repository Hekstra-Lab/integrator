"""End-to-end check of the simulation shoeboxes against the nanoBragg kernel.

Loads a `make_sim_shoeboxes.py` dataset through `PolychromaticDataModule`, then on
the brightest shoeboxes compares the observed spot with nanoBragg rendered on the
shoebox's own pixels at the true orientation: centroid offset, and the cosine
similarity of the background-subtracted counts with the rendered profile for a few
crystal sizes. Only the target reflection's lattice point is rendered (F_cell = 1
for its hkl, 0 elsewhere), so a neighbour inside the box does not count as signal.

Run:
    uv run python scripts/physics/check_sim_registration.py DATA_DIR
"""

import argparse
import math

import numpy as np
import torch
import yaml

from integrator.data_loaders.poly_data_module import PolychromaticDataModule
from integrator.physics.nanobragg import Beam, Crystal, mosaic_cells, mosaic_domains, render_pixels
from integrator.physics.shoebox_geometry import CrystalModels, detector_from_spec, shoebox_pixel_indices

DT = torch.float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--n-bright", type=int, default=150)
    ap.add_argument("--domains", type=int, default=100)
    ap.add_argument("--n-lambda", type=int, default=25)
    ap.add_argument("--lambda-window", type=float, default=0.015)
    ap.add_argument("--n-cells", type=float, nargs="+", default=[10.0, 20.0, 40.0, 80.0])
    args = ap.parse_args()

    dm = PolychromaticDataModule(data_dir=args.data_dir, batch_size=512, num_workers=0, transform="anscombe")
    dm.setup()
    counts, _, masks, meta = next(iter(dm.train_dataloader()))
    print("batch metadata keys include:", sorted(k for k in meta if k in ("bbox.0", "bbox.2", "image_num", "n_harmonics")))

    spec = yaml.safe_load(open(f"{args.data_dir}/dataset.yaml"))
    physics = spec["physics"]
    h = w = spec["geometry"]["h"]
    det = detector_from_spec(physics)
    models = CrystalModels.load(args.data_dir, physics)
    all_counts = torch.as_tensor(np.load(f"{args.data_dir}/counts.npy")).double()
    all_masks = torch.as_tensor(np.load(f"{args.data_dir}/masks.npy")).double()
    meta_all = {k: torch.as_tensor(v) for k, v in np.load(f"{args.data_dir}/metadata.npy", allow_pickle=True).item().items()}

    keep = (meta_all["n_harmonics"] == 1) & (all_masks.sum(1) == h * w)
    order = torch.argsort(meta_all["intensity.sum.value"] * keep, descending=True)[: args.n_bright]
    fpix, spix = shoebox_pixel_indices({k: v[order] for k, v in meta_all.items()}, h, w)

    beam_spec = physics["beam"]
    normal = torch.tensor(beam_spec["polarization_normal"], dtype=DT)
    s0 = torch.tensor(beam_spec["s0_hat"], dtype=DT)
    e_vec = torch.linalg.cross(normal, s0)
    axes, frac = mosaic_domains(args.domains, torch.Generator().manual_seed(0), dtype=DT)

    ys, xs = torch.meshgrid(torch.arange(h, dtype=DT), torch.arange(w, dtype=DT), indexing="ij")
    rows = []
    for n_cells in args.n_cells:
        cos_list, off_list = [], []
        for j, i in enumerate(order.tolist()):
            img = meta_all["image_num"][i].reshape(1)
            hkl = torch.stack([meta_all[k][i] for k in ("H", "K", "L")]).round()
            lam0 = float(meta_all["wavelength"][i])
            lam = torch.linspace(lam0 * (1 - args.lambda_window), lam0 * (1 + args.lambda_window), args.n_lambda, dtype=DT)
            lam = lam[(lam >= beam_spec["lambda_min"]) & (lam <= beam_spec["lambda_max"])]
            beam = Beam(incident=s0.expand(len(lam), 3), wavelength=lam, kahn=beam_spec["polarization_fraction"],
                        polar_vector=e_vec)
            spread = torch.tensor(math.radians(float(models.mos_spread_deg[models.index(img)][0])), dtype=DT)
            crystal = Crystal(cells=mosaic_cells(models.cells(img)[0], axes, frac, spread),
                              recip=models.recip(img)[0], n_cells=torch.full((3,), n_cells, dtype=DT), shape="gauss")
            prof = render_pixels(fpix[j], spix[j], det, beam, crystal,
                                 f_cell=lambda h0: (h0 == hkl).all(-1).to(DT), domain_chunk=25)
            c = all_counts[i]
            border = torch.ones(h, w, dtype=torch.bool); border[2:-2, 2:-2] = False
            sig = (c - c.reshape(h, w)[border].median()).clamp(min=0)
            cos_list.append(float((prof @ sig) / (prof.norm() * sig.norm() + 1e-30)))
            if n_cells == args.n_cells[0]:
                def centroid(v):
                    v = v.reshape(h, w)
                    return torch.stack([(v * xs).sum() / v.sum(), (v * ys).sum() / v.sum()])
                pred = torch.stack([meta_all["xyzcal.px.0"][i] - meta_all["bbox.0"][i] - 0.5,
                                    meta_all["xyzcal.px.1"][i] - meta_all["bbox.2"][i] - 0.5]).double()
                off_list.append(torch.cat([centroid(sig) - pred, centroid(prof) - centroid(sig)]))
        cos = np.array(cos_list)
        print(f"N_cells={n_cells:4.0f}: cos(rendered, observed) median {np.median(cos):.3f}, p10 {np.percentile(cos, 10):.3f}")
        if off_list:
            off = torch.stack(off_list).numpy()
            print(f"  observed centroid - predicted (px): median |dx| {np.median(np.abs(off[:, 0])):.2f}, "
                  f"|dy| {np.median(np.abs(off[:, 1])):.2f}")
            print(f"  rendered centroid - observed (px):  median |dx| {np.median(np.abs(off[:, 2])):.2f}, "
                  f"|dy| {np.median(np.abs(off[:, 3])):.2f}")


if __name__ == "__main__":
    main()

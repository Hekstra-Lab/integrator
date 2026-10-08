import math

import torch

from integrator.physics.nanobragg import (
    Beam,
    Crystal,
    Detector,
    mosaic_cells,
    mosaic_domains,
    render_pixels,
    sincg,
)

DT = torch.float64


def _setup(shape="gauss", n_dom=6, n_src=3):
    a_star = torch.diag(torch.tensor([1 / 33.0, 1 / 44.0, 1 / 95.0], dtype=DT))
    cell = torch.linalg.inv(a_star)
    det = Detector(
        pix0=torch.tensor([-50.0, 50.0, -100.0], dtype=DT),
        fdet=torch.tensor([1.0, 0, 0], dtype=DT),
        sdet=torch.tensor([0.0, -1, 0], dtype=DT),
        odet=torch.tensor([0.0, 0, -1], dtype=DT),
        pixel_size=0.1,
    )
    axes, frac = mosaic_domains(n_dom, torch.Generator().manual_seed(0), dtype=DT)
    beam = Beam(
        incident=torch.tensor([[0.0, 0, -1]], dtype=DT).expand(n_src, 3),
        wavelength=torch.linspace(1.0, 1.01, n_src, dtype=DT),
    )
    ss, ff = torch.meshgrid(torch.arange(400, 410), torch.arange(600, 610), indexing="ij")
    return cell, a_star, det, axes, frac, beam, ff.reshape(-1), ss.reshape(-1), shape


def test_sincg_limits():
    n = torch.tensor(7.0, dtype=DT)
    x = torch.tensor([0.0, math.pi, 2 * math.pi, 0.3], dtype=DT)
    out = sincg(x, n)
    assert torch.allclose(out[:3].abs(), torch.full((3,), 7.0, dtype=DT))
    assert torch.isclose(out[3], torch.sin(7 * x[3]) / torch.sin(x[3]))


def test_first_domain_is_unrotated():
    axes, frac = mosaic_domains(5, torch.Generator().manual_seed(0), dtype=DT)
    cell = torch.eye(3, dtype=DT) * 30
    cells = mosaic_cells(cell, axes, frac, torch.tensor(0.01, dtype=DT))
    assert torch.allclose(cells[0], cell)
    assert not torch.allclose(cells[1], cell)


def test_equal_weights_match_unweighted():
    cell, a_star, det, axes, frac, beam, ff, ss, shape = _setup()
    xt = Crystal(cells=mosaic_cells(cell, axes, frac, torch.tensor(0.003, dtype=DT)), recip=a_star,
                 n_cells=torch.tensor([6.0, 6, 6], dtype=DT), shape=shape)
    plain = render_pixels(ff, ss, det, beam, xt)
    weighted = render_pixels(
        ff, ss, det, Beam(incident=beam.incident, wavelength=beam.wavelength, weight=torch.full((3,), 5.0, dtype=DT)), xt
    )
    chunked = render_pixels(ff, ss, det, beam, xt, domain_chunk=2)
    assert torch.allclose(plain, weighted)
    assert torch.allclose(plain, chunked)


def test_gradcheck_physical_parameters():
    cell, a_star, det, axes, frac, beam, ff, ss, shape = _setup()

    def f(n_cells, spread):
        xt = Crystal(cells=mosaic_cells(cell, axes, frac, spread), recip=a_star, n_cells=n_cells, shape=shape)
        return render_pixels(ff, ss, det, beam, xt).sum()

    n_cells = torch.tensor([6.0, 7.0, 5.0], dtype=DT, requires_grad=True)
    spread = torch.tensor(0.003, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(f, (n_cells, spread), eps=1e-7, atol=1e-8, rtol=1e-4)

import math

import torch

from integrator.physics.nanobragg import (
    Beam,
    Crystal,
    Detector,
    amorphous_molecules,
    interpolate_stol,
    mosaic_cells,
    mosaic_domains,
    render_background,
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


def test_batched_matches_per_shoebox():
    from integrator.physics.nanobragg import render_shoeboxes, rotation_matrices

    cell, a_star, det, axes, frac, _, _, _, shape = _setup(n_dom=5)
    gen = torch.Generator().manual_seed(3)
    # boxes centred where these reflections diffract at these wavelengths
    spots = [((654, 360), (5, 6, 2), 1.0), ((406, 312), (-3, 8, 2), 1.01), ((566, 128), (2, 15, 6), 1.02)]
    n_box = len(spots)
    rots = rotation_matrices(torch.nn.functional.normalize(torch.randn(n_box, 3, generator=gen, dtype=DT), dim=-1),
                             torch.rand(n_box, generator=gen, dtype=DT) * 2e-4)
    cells_img = torch.einsum("bij,kj->bki", rots, cell)
    recip_img = torch.linalg.inv(cells_img).transpose(-1, -2)
    cells = torch.stack([mosaic_cells(c, axes, frac, torch.tensor(0.004, dtype=DT)) for c in cells_img])
    lam = torch.stack([torch.linspace(l0 * 0.995, l0 * 1.005, 5, dtype=DT) for _, _, l0 in spots])
    wts = torch.rand(n_box, 5, generator=gen, dtype=DT) + 0.1
    ys, xs = torch.meshgrid(torch.arange(-6, 7, dtype=DT), torch.arange(-6, 7, dtype=DT), indexing="ij")
    fpix = torch.stack([xs.reshape(-1) + f0 + 0.3 for (f0, _), _, _ in spots])
    spix = torch.stack([ys.reshape(-1) + s0 for (_, s0), _, _ in spots])
    target = torch.tensor([hkl for _, hkl, _ in spots], dtype=DT)
    n_cells = torch.tensor([6.0, 7.0, 5.0], dtype=DT)
    polar = torch.tensor([1.0, 0, 0], dtype=DT)
    incident = torch.tensor([0.0, 0, -1], dtype=DT)
    kwargs = dict(shape=shape, kahn=0.9, polar_vector=polar, domain_chunk=2, checkpoint=True)

    batched = render_shoeboxes(fpix, spix, det, incident, lam, wts, cells, recip_img, n_cells, target_hkl=target, **kwargs)
    everything = render_shoeboxes(fpix, spix, det, incident, lam, wts, cells, recip_img, n_cells, **kwargs)
    # the target is the spot in each box, not an empty mask
    assert (batched.amax(-1) > 0.5 * everything.amax(-1)).all()
    assert (batched.amax(-1) > 1e3 * batched.amin(-1)).all()
    for b in range(n_box):
        xt = Crystal(cells=cells[b], recip=recip_img[b], n_cells=n_cells, shape=shape)
        beam = Beam(incident=incident.expand(5, 3), wavelength=lam[b], weight=wts[b], kahn=0.9, polar_vector=polar)
        single = render_pixels(fpix[b], spix[b], det, beam, xt,
                               f_cell=lambda h0, t=target[b]: (h0 == t).all(-1).to(DT))
        assert torch.allclose(batched[b], single, rtol=1e-10, atol=1e-30)


def test_batched_gradcheck_with_checkpoint():
    from integrator.physics.nanobragg import render_shoeboxes

    cell, a_star, det, axes, frac, _, ff, ss, shape = _setup(n_dom=4)
    lam = torch.linspace(0.995, 1.005, 3, dtype=DT).expand(2, 3)
    wts = torch.ones(2, 3, dtype=DT)
    # a box on the (5, 6, 2) spot, so the derivatives are of a real profile
    ys, xs = torch.meshgrid(torch.arange(-4, 5, dtype=DT), torch.arange(-4, 5, dtype=DT), indexing="ij")
    ff, ss = xs.reshape(-1) + 654, ys.reshape(-1) + 360

    def f(n_cells, spread, shift):
        cells = mosaic_cells(cell, axes, frac, spread).expand(2, -1, -1, -1)
        return render_shoeboxes(torch.stack([ff, ff]) + shift, torch.stack([ss, ss]), det,
                                torch.tensor([0.0, 0, -1], dtype=DT), lam, wts, cells,
                                a_star.expand(2, 3, 3), n_cells, shape=shape, domain_chunk=2, checkpoint=True).sum()

    args = (torch.tensor([6.0, 7.0, 5.0], dtype=DT, requires_grad=True),
            torch.tensor(0.003, dtype=DT, requires_grad=True),
            torch.tensor(0.2, dtype=DT, requires_grad=True))
    assert torch.autograd.gradcheck(f, args, eps=1e-7, atol=1e-8, rtol=1e-4)


def test_continuous_wavelength_matches_dense_grid():
    """Exact when the spectrum is smooth over a lattice point's crossing (here 1e3 cells, 20% band)."""
    from integrator.physics.nanobragg import render_shoeboxes, render_shoeboxes_continuous, rotation_matrices

    cell, a_star, det, axes, frac, _, _, _, _ = _setup(n_dom=6)
    rot = rotation_matrices(axes, frac * 0.004)
    hkl = torch.tensor([[5.0, 6.0, 2.0]], dtype=DT)
    q_hkl = hkl @ torch.linalg.inv(cell).T
    ys, xs = torch.meshgrid(torch.arange(-6, 7, dtype=DT), torch.arange(-6, 7, dtype=DT), indexing="ij")
    fpix, spix = (xs.reshape(-1) + 654.2)[None], (ys.reshape(-1) + 360)[None]
    n_cells = torch.tensor([10.0, 10.0, 10.0], dtype=DT)
    incident = torch.tensor([0.0, 0, -1], dtype=DT)
    polar = torch.tensor([1.0, 0, 0], dtype=DT)

    def spectrum(lam):
        return torch.exp(-(((lam - 1.003) / 0.2) ** 2))

    cont = render_shoeboxes_continuous(fpix, spix, det, incident, spectrum, rot, q_hkl, n_cells,
                                       kahn=0.9, polar_vector=polar)
    lam = torch.linspace(0.85, 1.15, 30001, dtype=DT)[None]
    cells = torch.einsum("dij,kj->dki", rot, cell)[None]
    dense = render_shoeboxes(fpix, spix, det, incident, lam, spectrum(lam), cells, a_star[None], n_cells,
                             shape="gauss", target_hkl=hkl, kahn=0.9, polar_vector=polar, domain_chunk=2)
    ref = dense * spectrum(lam).sum() * (lam[0, 1] - lam[0, 0])
    sel = ref > 1e-3 * ref.max()
    assert sel.sum() > 20
    assert torch.allclose(cont[sel], ref[sel], rtol=1e-3)


def _bg_setup(n_src=5):
    """The geometry nonBragg printed for the water render these tests pin.

    detector origin 0.200498 0.176022 -0.173968 m, fast axis (0, 0, 1), slow
    axis (0, -1, 0), beam along (1, 0, 0), pixel 8.854e-5 m, five equal sources.
    """
    det = Detector(
        pix0=torch.tensor([0.200498, 0.176022, -0.173968], dtype=DT),
        fdet=torch.tensor([0.0, 0.0, 1.0], dtype=DT),
        sdet=torch.tensor([0.0, -1.0, 0.0], dtype=DT),
        odet=torch.tensor([1.0, 0.0, 0.0], dtype=DT),
        pixel_size=8.854e-05,
    )
    beam = Beam(
        incident=torch.tensor([[1.0, 0.0, 0.0]], dtype=DT).expand(n_src, 3),
        wavelength=torch.linspace(1.02375, 1.07625, n_src, dtype=DT),
        fluence=1.25932e29,
    )
    # a short stand-in for water.stol: the simtbx default table, which rises to
    # the 3.1 A ring and falls away
    curve_stol = torch.tensor(
        [0.0, 0.0365, 0.07, 0.12, 0.162, 0.18, 0.2, 0.216, 0.236, 0.28, 0.3,
         0.345, 0.436, 0.5], dtype=DT
    )
    curve_f = torch.tensor(
        [2.57, 2.58, 2.8, 5.0, 8.0, 7.32, 6.75, 6.75, 6.5, 4.5, 4.3, 4.36,
         3.77, 3.17], dtype=DT
    )
    return det, beam, curve_stol, curve_f


def test_amorphous_molecules_matches_nonbragg():
    """nonBragg prints 1.67282e17 for 0.5 mm of water in a 0.1 mm beam."""
    n = amorphous_molecules(0.5, beam_size_mm=0.1, density_gcm3=1.0, molecular_weight_da=18.0)
    assert math.isclose(n, 1.67282e17, rel_tol=1e-5)


def test_interpolate_stol_is_exact_on_the_table_and_flat_outside():
    _, _, cs, cf = _bg_setup()
    assert torch.allclose(interpolate_stol(cs, cs, cf), cf)
    ends = torch.tensor([-1.0, 0.0, 0.5, 9.0], dtype=DT)
    got = interpolate_stol(ends, cs, cf)
    assert torch.isclose(got[0], cf[0]) and torch.isclose(got[-1], cf[-1])
    mid = interpolate_stol(torch.tensor([0.5 * (0.0365 + 0.07)], dtype=DT), cs, cf)
    assert torch.isclose(mid[0], torch.tensor(0.5 * (2.58 + 2.8), dtype=DT))


def test_background_is_linear_in_molecules_and_fluence():
    """Both enter as plain prefactors, so doubling either doubles the image."""
    det, beam, cs, cf = _bg_setup()
    ff = torch.arange(1900, 1910, dtype=DT)
    ss = torch.full_like(ff, 1900.0)
    one = render_background(ff, ss, det, beam, cs, cf, 1.0e17)
    two = render_background(ff, ss, det, beam, cs, cf, 2.0e17)
    assert torch.allclose(two, 2.0 * one, rtol=1e-12)
    hot = Beam(
        incident=beam.incident, wavelength=beam.wavelength, fluence=2 * beam.fluence
    )
    assert torch.allclose(
        render_background(ff, ss, det, beam, cs, cf, 1.0e17) * 2.0,
        render_background(ff, ss, det, hot, cs, cf, 1.0e17),
        rtol=1e-12,
    )


def test_background_peaks_at_the_water_ring():
    """Walking out from the beam centre, the background must rise to the 3.1 A
    ring and fall again -- a monotone profile would mean stol is wrong."""
    det, beam, cs, cf = _bg_setup()
    ff = torch.arange(1965.0, 3840.0, 40.0, dtype=DT)
    ss = torch.full_like(ff, 1965.0)
    bg = render_background(ff, ss, det, beam, cs, cf, 1.67282e17)
    peak = int(torch.argmax(bg))
    assert 0 < peak < len(bg) - 1, "the maximum must be interior, not at an edge"
    assert bg[peak] > 2.0 * bg[0], "the ring should stand well above forward scattering"
    assert bg[-1] < bg[peak], "the background must fall away past the ring"

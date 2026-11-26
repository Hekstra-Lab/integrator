"""Architecture diagrams of the integrators, profile models, encoders, exports and the planned joint model.

Graphviz with orthogonal edge routing (`splines=ortho`), one diagram per component,
rendered to PDF and SVG in docs/figures/architecture/, and docs/architectures.md
written beside them (the SVGs embedded, a sentence per diagram). The DOT sources
live here, so a change to a model is a change to one string.

Contents follow the code and the configs of the mos010 arms (config_log.yaml of
each run): `svae` with `gate: responsibility`; profiles `learned_basis_profile`
(svae), `streak_kmix_profile` (T0, P2), `streak_kmix_v2_profile` with
`shape_posterior: amortized` (B1) and `streak_kmix_v2_geom` (geometric, trained
with `svae_collapsed`).

    uv run python scripts/figures/fig_architecture_diagrams.py   # needs graphviz `dot`
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures" / "architecture"
DOC = ROOT / "docs" / "architectures.md"

HEAD = """digraph G {
  graph [splines=ortho, rankdir=LR, nodesep=0.35, ranksep=0.55, fontname="Helvetica", fontsize=11, pad=0.2];
  node  [shape=box, style="rounded,filled", fillcolor="#F4F6F8", color="#5A6B7B", fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
  edge  [color="#5A6B7B", arrowsize=0.7, fontname="Helvetica", fontsize=9];
"""
DATA = 'fillcolor="#E8F1FB", color="#2F6DB5"'
NET = 'fillcolor="#FFF3E0", color="#C77700"'
DIST = 'fillcolor="#EAF6EC", color="#2E8B57"'
FIXED = 'fillcolor="#F1EAF7", color="#7B4FA0"'
OUTN = 'fillcolor="#FDECEC", color="#B03A3A"'

DIAGRAMS: dict[str, tuple[str, str]] = {}

DIAGRAMS["legend"] = ("Legend", HEAD + f"""
  data [label="data / metadata", {DATA}];
  net  [label="learned network", {NET}];
  dist [label="distribution / posterior", {DIST}];
  fix  [label="closed form, no parameters", {FIXED}];
  out  [label="output / export", {OUTN}];
  data -> net -> dist -> fix -> out;
}}""")

DIAGRAMS["encoders"] = ("Encoders (ProfileEncoder, IntensityEncoder; 2-D, 25 x 25 window)", HEAD + f"""
  sb   [label="shoebox counts x mask\\n1 x 25 x 25", {DATA}];
  c1   [label="Conv 3x3, 1 -> 16\\nGroupNorm(4), ReLU", {NET}];
  p1   [label="MaxPool 2x2", {NET}];
  c2   [label="Conv 3x3, 16 -> 32\\nGroupNorm(4), ReLU", {NET}];
  fl   [label="flatten -> Linear -> 32\\n(output_activation: none)", {NET}];
  xp   [label="x_profile (32)", {DIST}];
  sb -> c1 -> p1 -> c2 -> fl -> xp;
  ie   [label="IntensityEncoder\\n(same layout) -> x_signal (32)", {NET}];
  note [label="with gate: responsibility the gate is computed,\\nnot emitted: x_signal reaches no loss term\\n(the encoder is built but receives no gradient)", shape=note, fillcolor="#FFFFFF"];
  sb -> ie -> note;
}}""")

DIAGRAMS["svae"] = ("svae integrator, gate: responsibility (all model arms except geometric)", HEAD + f"""
  sb   [label="counts c, mask m\\n25 x 25", {DATA}];
  md   [label="metadata\\n(x, y, lambda, d, group)", {DATA}];
  enc  [label="ProfileEncoder\\n-> x (32)", {NET}];
  qp   [label="profile surrogate q(p)\\n(one of the profile diagrams)", {NET}];
  pbar [label="mean profile p", {DIST}];
  b0   [label="b0 = median of valid pixels", {FIXED}];
  i0   [label="I0 = profile-weighted\\nleast squares given b0", {FIXED}];
  r    [label="responsibility\\nr_k = I0 p_k / (I0 p_k + b0)", {FIXED}];
  wil  [label="Wilson prior on I\\nGamma(alpha, tau(lambda, d))\\nG(lambda) Chebyshev, B", {NET}];
  bgp  [label="background prior\\nGamma per resolution bin", {DIST}];
  qi   [label="q(I) = Gamma(alpha + sum r c,\\n tau + sum p m)", {FIXED}];
  qb   [label="q(b) = Gamma(a + sum (1-r) c,\\n rate + sum m)", {FIXED}];
  rate [label="rate = I p + b\\n(MC samples)", {FIXED}];
  loss [label="loss = Poisson NLL\\n+ KL(q(p)) + KL(q(I)) + KL(q(b))", {OUTN}];
  sb -> enc -> qp; md -> qp; qp -> pbar;
  sb -> b0 -> i0; pbar -> i0; i0 -> r; b0 -> r; pbar -> r;
  r -> qi; wil -> qi; md -> wil; r -> qb; bgp -> qb;
  qi -> rate; qb -> rate; pbar -> rate; rate -> loss; sb -> loss;
}}""")

DIAGRAMS["svae_collapsed"] = ("svae_collapsed integrator (geometric arm): intensity and background integrated out", HEAD + f"""
  sb   [label="counts c, mask m", {DATA}];
  md   [label="metadata", {DATA}];
  enc  [label="ProfileEncoder -> x", {NET}];
  prior[label="shape prior p(eta | m)\\n(geometric law)", {NET}];
  qeta [label="q(eta) full covariance\\nmean, Cholesky from x\\n(off-diagonal head starts at 0)", {NET}];
  draw [label="eta_s ~ q(eta), s = 1..2", {DIST}];
  prof [label="profile p(eta_s)", {FIXED}];
  logz [label="log Z(eta) = log int int p(c | I, b, p) p(I) p(b) dI db\\nI on a grid (64), b by Laplace", {FIXED}];
  wil  [label="Wilson p(I), Gamma p(b)", {NET}];
  bnd  [label="bound = E_q[log Z(eta)] - KL(q(eta) || p(eta))\\ngradient by the envelope theorem", {OUTN}];
  sb -> enc -> qeta; md -> prior; prior -> qeta; qeta -> draw -> prof -> logz; sb -> logz; wil -> logz;
  logz -> bnd; prior -> bnd; qeta -> bnd;
}}""")

DIAGRAMS["profile_learned_basis"] = ("learned_basis_profile (svae arm): a per-reflection latent decoded by a learned linear basis", HEAD + f"""
  x    [label="x (32)\\nfrom ProfileEncoder", {DATA}];
  loc  [label="Linear -> loc(x) (12)", {NET}];
  sc   [label="Linear, softplus -> scale(x) (12)", {NET}];
  qh   [label="q(h | x) = N(loc, diag scale^2)\\nprior N(0, 3^2)", {DIST}];
  dec  [label="W h + b\\nW: 625 x 12 learned basis", {NET}];
  sm   [label="softmax over valid pixels", {FIXED}];
  p    [label="profile p (625)\\nno geometry input", {OUTN}];
  x -> loc -> qh; x -> sc -> qh; qh -> dec -> sm -> p;
}}""")

DIAGRAMS["profile_streak_kmix"] = ("streak_kmix_profile (T0, P2): shape from geometry only", HEAD + f"""
  md   [label="metadata: spot x, y; lambda", {DATA}];
  bc   [label="beam centre (config)", {DATA}];
  geo  [label="geometry_features: r, r^2,\\ncos phi, sin phi, cos 2phi, sin 2phi,\\nlambda~, lambda~^2, r lambda~", {FIXED}];
  phi  [label="azimuth phi\\n(analytic)", {FIXED}];
  mlp  [label="MLP f: 9 -> hidden -> theta\\nhidden 32 (T0), 128 (P2)", {NET}];
  th0  [label="theta0 (trainable)\\nstart: width grid x init_grid_scale\\n(1 for T0, 0.5 for P2)", {NET}];
  th   [label="theta = theta0 + f(m):\\nK = 3 along / across widths,\\nweights, centre shift", {FIXED}];
  rot  [label="rotate pixel grid by phi\\n(u along, v across)", {FIXED}];
  mix  [label="g = sum_k w_k N(u; a_k) N(v; c_k)\\nwidths capped at half window", {FIXED}];
  p    [label="profile p = normalised g\\nresidual: none -> no pixel input", {OUTN}];
  md -> geo; bc -> geo; md -> phi; bc -> phi; geo -> mlp -> th; th0 -> th; th -> mix; phi -> rot -> mix; mix -> p;
}}""")

DIAGRAMS["profile_streak_kmix_v2"] = ("streak_kmix_v2_profile, shape_posterior: amortized (B1): per-reflection shape posterior", HEAD + f"""
  md   [label="metadata", {DATA}];
  x    [label="x (32) from ProfileEncoder", {DATA}];
  base [label="prior mean (theta0 + f(m), psi(m))\\nas streak_kmix, plus tilt psi(m)", {NET}];
  dlt  [label="delta(x) = 3 tanh(Linear(x) / 3)\\ns(x) = softplus(Linear(x))", {NET}];
  sig  [label="sigma (trainable,\\nbounded by shape_sd_max)", {NET}];
  eta  [label="eta = mean + sigma (delta + s eps)\\nKL = 1/2 sum(delta^2 + s^2 - 1 - 2 log s)", {DIST}];
  mix  [label="K = 3 aligned Gaussians\\n(rotated by phi + psi)", {FIXED}];
  p    [label="profile p (per reflection)", {OUTN}];
  md -> base -> eta; x -> dlt -> eta; sig -> eta; eta -> mix -> p;
}}""")

DIAGRAMS["profile_geom"] = ("streak_kmix_v2_geom (geometric arm): the shape network replaced by a width law", HEAD + f"""
  md   [label="metadata: spot radius r", {DATA}];
  t    [label="t = tan 2 theta = r / D\\nD = 2264 px", {FIXED}];
  law  [label="along_k = sqrt(alpha_k^2 + beta_k^2 (1 + t^2)^2)\\nacross_k = sqrt(gamma_k^2 + eps_k^2 t^2)\\n4 parameters per component", {NET}];
  w    [label="weights, centre: theta0\\n(no MLP: shape_head = None)", {NET}];
  pm   [label="prior mean of eta", {DIST}];
  q    [label="q(eta) as in B1 / svae_collapsed", {DIST}];
  p    [label="profile p", {OUTN}];
  md -> t -> law -> pm; w -> pm; pm -> q -> p;
}}""")

DIAGRAMS["exports"] = ("Exports to careless: the same checkpoint, two estimators", HEAD + f"""
  ck   [label="trained checkpoint\\n(epochs 39-59 averaged for B1, geometric)", {DATA}];
  rb   [label="i_rb: profile fit at export, no intensity prior\\nI = sum p (c - b) / sum p^2 per profile draw,\\naveraged; var = Schur complement + spread", {FIXED}];
  lc   [label="Laplace + collapsed: joint mode of (eta, I, b)\\n(damped Newton), q(eta) = N(mode, H^-1),\\n(I, b) | eta exact on a grid, 16 eta draws", {FIXED}];
  mtz1 [label="MTZ: I, SIGI\\n(a likelihood summary)", {OUTN}];
  mtz2 [label="MTZ: posterior mean, sd\\n(includes the Wilson prior)", {OUTN}];
  cl   [label="careless (config 4)\\napplies its own Wilson prior", {NET}];
  ck -> rb -> mtz1 -> cl; ck -> lc -> mtz2 -> cl;
}}""")

DIAGRAMS["downstream"] = ("Downstream chain and diagnostics", HEAD + f"""
  mtz  [label="unmerged MTZ\\n(one per arm)", {DATA}];
  cl   [label="careless poly, config 4\\nd_min 1.70, seed 1", {NET}];
  st   [label="careless.stats: CC1/2, CCanom,\\nCCpred, Rsplit, I/sigma, completeness", {OUTN}];
  ph   [label="phenix.refine x 2\\npinned R-free flags\\n(sim_7lvc_rfree.mtz)", {NET}];
  rf   [label="R-work / R-free by shell, FOM,\\nphase error, CC_work / CC_free, CC*", {OUTN}];
  pk   [label="anomalous peaks\\n(S, Mn sites)", {OUTN}];
  tr   [label="truth: ground_truth.mtz,\\ntruth_reflections.csv", {DATA}];
  mv   [label="merged F against F_true\\n(score_merged_vs_truth.py)", {OUTN}];
  mtz -> cl -> st; cl -> ph -> rf; ph -> pk; cl -> mv; tr -> mv;
}}""")

DIAGRAMS["joint"] = ("Planned joint model: pixels to scaled and merged intensities", HEAD + f"""
  px   [label="pixels of observation i\\nof reflection h", {DATA}];
  md   [label="metadata m_i\\n(lambda, shot, detector)", {DATA}];
  loc  [label="local latents: shape eta_i (Laplace q),\\nbackground b_i (Laplace)", {DIST}];
  li   [label="l_i(I) = int int p(c_i | I, b, eta) p(b) p(eta)\\non the observation's I grid\\n(importance weights p/q)", {FIXED}];
  sc   [label="scale g_i = exp(T(lambda) theta + u_shot + ...)\\nlow-dimensional, gauge fixed", {NET}];
  pj   [label="prior on J_h: Wilson Sigma(s) by EB,\\nbivariate for Friedel pairs, or flat", {DIST}];
  mj   [label="log p(pixels_h | theta) =\\nlog int p(J) prod_i l_i(g_i J) dJ\\n(grid quadrature)", {FIXED}];
  qj   [label="q(J_h): exact conditional posterior\\n-> merged F", {OUTN}];
  px -> loc -> li; px -> li; md -> sc; li -> mj; sc -> mj; pj -> mj; mj -> qj;
}}""")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Architectures", "",
             "Diagrams of every integrator, profile model, encoder and export used on the mos010 simulation, "
             "and of the planned joint model.",
             "They are drawn by `scripts/figures/fig_architecture_diagrams.py` (Graphviz, orthogonal edges); "
             "PDFs are in `docs/figures/architecture/`.",
             "Colours: blue data, orange learned, green distribution, purple closed form, red output.", ""]
    for name, (title, dot) in DIAGRAMS.items():
        src = OUT / f"{name}.dot"
        src.write_text(dot)
        for fmt in ("pdf", "svg"):
            subprocess.run(["dot", f"-T{fmt}", str(src), "-o", str(OUT / f"{name}.{fmt}")], check=True)
        lines += [f"## {title}", "", f"![{name}](figures/architecture/{name}.svg)", ""]
        print(OUT / f"{name}.pdf")
    DOC.write_text("\n".join(lines))
    print(DOC)


if __name__ == "__main__":
    main()

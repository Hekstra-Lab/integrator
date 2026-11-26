"""Strictly learned profile models whose latent prior depends on the reflection's geometry.

Faint reflections cannot fix their own shape, so a profile model must pool shape
across reflections. `learned_basis_profile` pools only through a global N(0, s^2)
prior on its latent, so every faint reflection decodes to one average shape,
wherever it sits on the detector (on mos010: compact round spots where the truth
is a radial streak). The streak models pool through geometry but also impose a
parametric streak form. The classes here keep the shape family strictly learned
and add only a geometry-conditioned prior:

    h ~ N(mu(m), s^2),   mu = MLP of the nine geometry features (profile_geometry),
    q(h | x, m) = N(mu(m) + delta(x), scale(x)^2),
    KL = 1/2 sum_j [ (delta_j^2 + scale_j^2) / s^2 - 1 - 2 log(scale_j / s) ]

The MLP's last layer starts at zero, so mu = 0 and the model at initialisation is
the parent exactly. The posterior is an offset from the prior mean, so a reflection
whose encoder output carries little shape information sits on its geometry's shape.

    learned_basis_geom_prior_profile    p = softmax(W h + b), the parent's learned basis
    gmm_geom_prior_profile              p = a mixture of K oriented Gaussians decoded from the latent,
                                        orientation free in the detector frame, starting isotropic
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.distributions.profile_geometry import N_GEOM_FEATURES, geometry_features
from integrator.model.distributions.profile_surrogates import (
    ProfileSurrogate,
    ProfileSurrogateOutput,
    _sample_and_decode,
)


class GeomPriorMLP(nn.Module):
    """mu(m): nine geometry features -> hidden (SiLU) -> d, last layer at zero."""

    def __init__(self, d: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_GEOM_FEATURES, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats: Tensor) -> Tensor:
        return self.net(feats)


class LearnedBasisGeomPriorSurrogate(ProfileSurrogate):
    """`learned_basis_profile` with the latent prior mean an MLP of the geometry features.

    Args (beyond the parent's):
        beam_center: detector beam centre in pixels, for the geometry features.
        lambda_min, lambda_max: the spectrum's ends, mapping wavelength to [-1, 1].
        geom_hidden: width of the prior-mean MLP.
    """

    def __init__(self, input_dim: int, latent_dim: int | None = None, output_dim: int = 625,
                 init_std: float = 0.5, prior_scale: float = 3.0, beam_center=None,
                 lambda_min: float | None = None, lambda_max: float | None = None,
                 geom_hidden: int = 64, **kwargs) -> None:
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim,
                         init_std=init_std, prior_scale=prior_scale, **kwargs)
        if beam_center is None:
            raise ValueError("learned_basis_geom_prior_profile needs beam_center")
        self.beam_center = [float(v) for v in beam_center]
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.prior_mean_head = GeomPriorMLP(self.d, geom_hidden)

    def prior_mean(self, metadata: dict, like: Tensor) -> Tensor:
        feats, _ = geometry_features(metadata, self.beam_center, self.lambda_min, self.lambda_max)
        return self.prior_mean_head(feats.to(like.dtype).to(like.device))

    def forward(self, x: Tensor, mc_samples: int = 1, metadata: dict | None = None, **kwargs) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("learned_basis_geom_prior_profile needs metadata")
        mu = self.prior_mean(metadata, x)
        delta = self.loc_head(x)
        loc = mu + delta
        scale = F.softplus(self.scale_head(x)) + 1e-6
        zp, mean_profile, mean_logits = _sample_and_decode(
            loc, scale, self.decoder.weight, self.decoder.bias, mc_samples,
            mask=kwargs.get("mask"), normalize=self._normalize)
        s = float(self.prior_scale)
        kl = 0.5 * ((delta**2 + scale**2) / s**2 - 1.0 - 2.0 * torch.log(scale / s)).sum(-1)
        return ProfileSurrogateOutput(zp=zp, mean_profile=mean_profile, loc=loc, scale=scale,
                                      prior_scale=self.prior_scale, kl=kl, mean_logits=mean_logits)


class GMMGeomPriorSurrogate(nn.Module):
    """A variational Gaussian mixture: shoebox and geometry -> latent z -> K oriented Gaussians.

        z ~ N(mu(m), s^2) prior,  q(z | x, m) = N(mu(m) + delta(x), scale(x)^2)   (as above)
        decoder MLP: z -> per component k: centre offset c_k (|c| <= max_shift, tanh),
            sds along its own axes sigma_k = sd_min + (sd_max - sd_min) sigmoid(.),
            angle theta_k (free, detector frame, from atan2 of two outputs), weight logit
        p = sum_k w_k G_k,  G_k = N(pixel centre; centre_k, R(theta_k) diag(sigma_k^2 + 1/12) R^T)
            normalised on the valid pixels before mixing, w = softmax(logits)

    Nothing ties the orientation to the beam-centre azimuth and nothing sets a streak
    form: elongation, orientation and their dependence on detector position are what
    the decoder and the geometry prior learn. The decoder's last layer starts at zero
    with biases giving K isotropic, centred components of sds `init_sds` and equal
    weights, so the model starts round everywhere.

    Args:
        input_dim: encoder width.  latent_dim: width of z.  output_dim: pixels.
        beam_center, lambda_min, lambda_max: for the geometry features.
        n_components: K.  geom_hidden: prior-mean MLP width.  hidden: decoder width.
        init_sds: starting isotropic sd of each component (pixels), len K.
        sd_min, sd_max: bounds on every sd (pixels; sd_max defaults to half the window).
        max_shift: bound on a component's centre offset (pixels).
        prior_scale: s.  init_std: starting posterior sd of z.
    """

    def __init__(self, input_dim: int, latent_dim: int | None = 12, output_dim: int = 625,
                 init_std: float = 0.5, prior_scale: float = 1.0, beam_center=None,
                 lambda_min: float | None = None, lambda_max: float | None = None,
                 n_components: int = 3, geom_hidden: int = 64, hidden: int = 64,
                 init_sds=(1.0, 2.5, 5.0), sd_min: float = 0.3, sd_max: float | None = None,
                 max_shift: float = 3.0, profile_shape=None, **kwargs) -> None:
        super().__init__()
        if beam_center is None:
            raise ValueError("gmm_geom_prior_profile needs beam_center")
        side = int(round(output_dim ** 0.5)) if profile_shape is None else int(profile_shape[-1])
        if side * side != output_dim:
            raise ValueError("gmm_geom_prior_profile is 2-D and square")
        self.d = int(latent_dim or 12)
        self.K = int(n_components)
        if len(init_sds) != self.K:
            init_sds = [1.0 * (2.5 ** k) for k in range(self.K)]
        self.prior_scale = float(prior_scale)
        self.beam_center = [float(v) for v in beam_center]
        self.lambda_min, self.lambda_max = lambda_min, lambda_max
        self.sd_min = float(sd_min)
        self.sd_max = float(sd_max) if sd_max is not None else side / 2.0
        self.max_shift = float(max_shift)
        self.loc_head = nn.Linear(input_dim, self.d)
        self.scale_head = nn.Linear(input_dim, self.d)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.constant_(self.scale_head.bias, float(torch.log(torch.expm1(torch.tensor(init_std)))))
        self.prior_mean_head = GeomPriorMLP(self.d, geom_hidden)
        # named mixture_decoder, not decoder: callbacks/figures.py records `qp.decoder` as a
        # linear basis (weight, bias), which this MLP is not
        self.mixture_decoder = nn.Sequential(nn.Linear(self.d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, 7 * self.K))
        last = self.mixture_decoder[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            b = torch.zeros(self.K, 7)
            for k, sd in enumerate(init_sds):
                frac = (min(max(sd, self.sd_min + 1e-3), self.sd_max - 1e-3) - self.sd_min) / (self.sd_max - self.sd_min)
                b[k, 2] = b[k, 3] = float(torch.logit(torch.tensor(frac)))
            b[:, 4] = 1.0  # angle components (cos-like, sin-like) = (1, 0): theta = 0
            last.bias.copy_(b.reshape(-1))
        ax = torch.arange(side, dtype=torch.float32) - (side - 1) / 2
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("_px", xx.reshape(-1), persistent=False)
        self.register_buffer("_py", yy.reshape(-1), persistent=False)

    def prior_mean(self, metadata: dict, like: Tensor) -> Tensor:
        feats, _ = geometry_features(metadata, self.beam_center, self.lambda_min, self.lambda_max)
        return self.prior_mean_head(feats.to(like.dtype).to(like.device))

    def components(self, z: Tensor):
        """z (..., d) -> centres (..., K, 2), sds (..., K, 2), angles (..., K), weights (..., K)."""
        out = self.mixture_decoder(z).reshape(*z.shape[:-1], self.K, 7)
        centre = self.max_shift * torch.tanh(out[..., 0:2] / self.max_shift)
        sds = self.sd_min + (self.sd_max - self.sd_min) * torch.sigmoid(out[..., 2:4])
        theta = torch.atan2(out[..., 5], out[..., 4])
        w = torch.softmax(out[..., 6], -1)
        return centre, sds, theta, w

    def decode(self, z: Tensor, fxy: Tensor, mask: Tensor | None) -> Tensor:
        """Profiles on the window for latents z (..., B, d); fxy (B, 2) the window-centred centroid."""
        centre, sds, theta, w = self.components(z)
        cx = fxy[:, 0, None] + centre[..., 0]
        cy = fxy[:, 1, None] + centre[..., 1]
        X = self._px.to(z) - cx[..., None]  # (..., B, K, P)
        Y = self._py.to(z) - cy[..., None]
        ct, st = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
        u, v = ct * X + st * Y, -st * X + ct * Y
        e = torch.exp(-0.5 * (u**2 / (sds[..., 0, None] ** 2 + 1 / 12) + v**2 / (sds[..., 1, None] ** 2 + 1 / 12)))
        if mask is not None:
            e = e * mask.reshape(mask.shape[0], 1, -1).to(e)
        e = e / e.sum(-1, keepdim=True).clamp_min(1e-30)
        p = (e * w[..., None]).sum(-2)
        return p / p.sum(-1, keepdim=True).clamp_min(1e-30)

    def forward(self, x: Tensor, mc_samples: int = 1, metadata: dict | None = None, **kwargs) -> ProfileSurrogateOutput:
        if metadata is None:
            raise ValueError("gmm_geom_prior_profile needs metadata")
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.reshape(mask.shape[0], -1)
        mu = self.prior_mean(metadata, x)
        delta = self.loc_head(x)
        loc = mu + delta
        scale = F.softplus(self.scale_head(x)) + 1e-6
        side = int(round(self._px.numel() ** 0.5))
        fxy = torch.stack([
            (metadata["xyzcal.px.0"].reshape(-1) - metadata["bbox.0"].reshape(-1)).to(x) - 0.5 - (side - 1) / 2,
            (metadata["xyzcal.px.1"].reshape(-1) - metadata["bbox.2"].reshape(-1)).to(x) - 0.5 - (side - 1) / 2,
        ], -1)
        z = loc[None] + scale[None] * torch.randn((mc_samples, *loc.shape), dtype=x.dtype, device=x.device)
        zp = self.decode(z, fxy, mask)
        mean_profile = self.decode(loc, fxy, mask)
        s = self.prior_scale
        kl = 0.5 * ((delta**2 + scale**2) / s**2 - 1.0 - 2.0 * torch.log(scale / s)).sum(-1)
        return ProfileSurrogateOutput(zp=zp, mean_profile=mean_profile, loc=loc, scale=scale,
                                      prior_scale=s, kl=kl, mean_logits=None)

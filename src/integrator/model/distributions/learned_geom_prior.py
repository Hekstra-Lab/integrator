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

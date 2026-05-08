import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum
from integrator.model.loss.wilson_loss import WilsonLoss


class AbsorptionCorrection(nn.Module):
    """Learnable absorption correction f_A(λ, r).

    Beer-Lambert: f_A = exp(-μ(λ) · L(r))
      μ(λ) = softplus(c₀ + c₁·λ³)         — physical λ³ dependence
      L(r) = softplus(Σ bₖ Tₖ(r̃))         — smooth path length vs radius

    f_A ∈ (0, 1] — always attenuates.
    """

    def __init__(
        self,
        beam_center: list[float],
        r_min: float = 0.0,
        r_max: float = 1500.0,
        path_degree: int = 4,
    ):
        super().__init__()
        r_mid = (r_min + r_max) / 2.0
        r_scale = (r_max - r_min) / 2.0

        self.register_buffer("beam_cx", torch.tensor(beam_center[0]))
        self.register_buffer("beam_cy", torch.tensor(beam_center[1]))
        self.register_buffer("r_mid", torch.tensor(r_mid))
        self.register_buffer("r_scale", torch.tensor(r_scale))
        self.path_degree = path_degree

        self.mu_coeffs = nn.Parameter(torch.tensor([-5.0, 0.0]))
        path_init = torch.zeros(path_degree + 1)
        path_init[0] = math.log(math.expm1(1.0))
        self.path_coeffs = nn.Parameter(path_init)

    def forward(
        self, wavelength: Tensor, x: Tensor, y: Tensor
    ) -> Tensor:
        """Returns f_A per reflection, in (0, 1]."""
        mu = F.softplus(self.mu_coeffs[0] + self.mu_coeffs[1] * wavelength.pow(3))

        r = torch.sqrt(
            (x - self.beam_cx).pow(2) + (y - self.beam_cy).pow(2)
        )
        rn = ((r - self.r_mid) / self.r_scale).clamp(-1.0, 1.0)
        phi_r = torch.stack(
            ChebyshevSpectrum._chebyshev(rn, self.path_degree), dim=-1
        )
        path_length = F.softplus(phi_r @ self.path_coeffs)

        return torch.exp(-mu * path_length)


class PolychromaticWilsonLoss(WilsonLoss):
    """Wilson loss for polychromatic (Laue) data with learned G(λ).

    Physical corrections (Ren & Moffat, J. Appl. Cryst. 28, 1995):
      polarization: f_P = 2 / (1 + cos²2θ − τ·cos2φ·sin²2θ)
      lorentz:      f_L = sin²2θ
      absorption:   f_A = exp(−μ(λ)·L(r)),  μ ∝ λ³ (Beer-Lambert)
    """

    def __init__(
        self,
        *,
        degree: int = 40,
        lambda_min: float = 0.95,
        lambda_max: float = 1.25,
        spectrum_init_from: str | None = None,
        freeze_prior: bool = False,
        # Physical corrections
        beam_center: list[float] | None = None,
        polarization: bool = False,
        polarization_fraction: float = 0.99,
        lorentz: bool = False,
        absorption_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.spectrum = ChebyshevSpectrum(
            degree=degree,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            init_from=spectrum_init_from,
        )

        # Warm-start B and concentration from checkpoint
        if spectrum_init_from is not None:
            saved = torch.load(
                spectrum_init_from, map_location="cpu", weights_only=False
            )
            if "raw_B" in saved and saved["raw_B"] is not None:
                with torch.no_grad():
                    self.raw_B.copy_(saved["raw_B"])
            if "log_alpha_per_group" in saved and self.learn_concentration:
                saved_alpha = saved["log_alpha_per_group"]
                if saved_alpha.shape == self.log_alpha_per_group.shape:
                    with torch.no_grad():
                        self.log_alpha_per_group.copy_(saved_alpha)

        if freeze_prior:
            self.spectrum.c.requires_grad_(False)
            self.raw_B.requires_grad_(False)
            if self.learn_concentration:
                self.log_alpha_per_group.requires_grad_(False)

        # Polarization correction (fixed geometric correction)
        self._apply_polarization = polarization
        self._apply_lorentz = lorentz
        if polarization:
            if beam_center is None:
                raise ValueError(
                    "polarization=True requires beam_center=[cx, cy]"
                )
            self.register_buffer("beam_cx", torch.tensor(beam_center[0]))
            self.register_buffer("beam_cy", torch.tensor(beam_center[1]))
            tau_pol = 2.0 * polarization_fraction - 1.0
            self.register_buffer("tau_pol", torch.tensor(tau_pol))

        # Absorption correction (learned)
        if absorption_cfg is not None:
            self.absorption = AbsorptionCorrection(**absorption_cfg)
        else:
            self.absorption = None

    def _polarization_factor(
        self, two_theta: Tensor, metadata: dict, device: torch.device
    ) -> Tensor:
        """f_P per reflection (Ren & Moffat eq. 11).

        Deterministic correction using known beam polarization fraction.
        φ measured from horizontal (synchrotron polarization plane).
        """
        x = metadata["xyzcal.px.0"].to(device)
        y = metadata["xyzcal.px.1"].to(device)
        phi = torch.atan2(y - self.beam_cy, x - self.beam_cx)

        cos2t = torch.cos(two_theta)
        sin2t = torch.sin(two_theta)
        return 2.0 / (
            1.0 + cos2t.pow(2)
            - self.tau_pol * torch.cos(2.0 * phi) * sin2t.pow(2)
        )

    @staticmethod
    def _lorentz_factor(two_theta: Tensor) -> Tensor:
        """f_L = sin²(2θ)  (Ren & Moffat eq. 10)."""
        return torch.sin(two_theta).pow(2).clamp(min=1e-8)

    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        if "wavelength" not in metadata:
            raise ValueError(
                "PolychromaticWilsonLoss requires metadata['wavelength']."
            )
        wavelength = metadata["wavelength"].to(device)
        G = torch.exp(self.spectrum.get_log_G(wavelength))
        B = self.get_B()
        tau = (1.0 / G) * torch.exp(2.0 * B * s_sq)

        if self._apply_polarization or self._apply_lorentz:
            d = metadata["d"].to(device)
            sin_theta = (wavelength / (2.0 * d.clamp(min=1e-6))).clamp(
                max=1.0
            )
            two_theta = 2.0 * torch.arcsin(sin_theta)

            if self._apply_polarization:
                P = self._polarization_factor(two_theta, metadata, device)
                tau = tau / P

            if self._apply_lorentz:
                L = self._lorentz_factor(two_theta)
                tau = tau / L

        if self.absorption is not None:
            x_px = metadata["xyzcal.px.0"].to(device)
            y_px = metadata["xyzcal.px.1"].to(device)
            f_A = self.absorption(wavelength, x_px, y_px)
            tau = tau / f_A

        return tau

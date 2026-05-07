import torch
from torch import Tensor

from integrator.model.loss.learned_spectrum import ChebyshevSpectrum
from integrator.model.loss.wilson_loss import WilsonLoss


class PolychromaticWilsonLoss(WilsonLoss):
    """Wilson loss for polychromatic (Laue) data with learned G(λ)."""

    def __init__(
        self,
        *,
        degree: int = 40,
        lambda_min: float = 0.95,
        lambda_max: float = 1.25,
        spectrum_init_from: str | None = None,
        freeze_prior: bool = False,
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
        return (1.0 / G) * torch.exp(2.0 * B * s_sq)

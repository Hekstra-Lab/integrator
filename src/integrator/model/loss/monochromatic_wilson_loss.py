import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.wilson_loss import WilsonLoss


class MonochromaticWilsonLoss(WilsonLoss):
    """Wilson loss for monochromatic data with scalar G."""

    def __init__(
        self,
        *,
        init_log_G: float = 0.0,
        lp_correction: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._apply_lp = lp_correction
        self.raw_G = nn.Parameter(torch.tensor(float(init_log_G)))

    def get_G(self) -> Tensor:
        return F.softplus(self.raw_G)

    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        G = self.get_G()
        B = self.get_B()
        tau = (1.0 / G) * torch.exp(2.0 * B * s_sq)

        if self._apply_lp:
            lp = metadata["lp"].to(device).clamp(min=1e-8)
            tau = tau * lp

        return tau

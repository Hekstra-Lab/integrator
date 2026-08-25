import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integrator.model.loss.count_likelihood import _inv_softplus
from integrator.model.loss.wilson_loss import WilsonLoss

_DEFAULT_INIT_G = 1.0


class MonochromaticWilsonLoss(WilsonLoss):
    """Wilson loss for monochromatic data with scalar G.

    Args:
        init_G: Initial overall scale G. When given, it also pins the scale:
            the one-time Wilson-plot fit from raw counts is skipped, so G starts
            exactly here. Leave as None to keep the fit (under `stabilize_scale`)
            or to fall back to a neutral G = 1.
        lp_correction: Multiply the Wilson rate by `metadata['lp']`.
    """

    def __init__(
        self,
        *,
        init_G: float | None = None,
        lp_correction: bool = False,
        n_scale_groups: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._apply_lp = lp_correction
        self.n_scale_groups = max(1, int(n_scale_groups))
        raw = self._raw_from_G(_DEFAULT_INIT_G if init_G is None else init_G)
        # One G per scale group. Two crystals differ in illuminated volume and
        # exposure, so a single G is mis-centred for both; a per-group G is
        # exactly solvable from dKL/dG = 0 (the mean B-corrected intensity of
        # that group) and carries no gauge freedom, because the intensities
        # themselves are still determined by each reflection's own pixels.
        self.raw_G = nn.Parameter(raw.repeat(self.n_scale_groups))
        if init_G is not None:
            # an explicit scale wins: never overwrite it with the count-based fit
            self.init_scale_from_counts = False

    def _raw_from_G(self, G: float) -> Tensor:
        """Invert the active G parameterization: raw_G such that get_G() == G."""
        g = torch.tensor(max(float(G), 1e-6))
        return torch.log(g) if self.stabilize_scale else _inv_softplus(g)

    def get_G(self) -> Tensor:
        """The scale, one entry per scale group."""
        if self.stabilize_scale:
            return torch.exp(self.raw_G)
        return F.softplus(self.raw_G)

    def diagnostics(self) -> dict[str, Tensor]:
        g = self.get_G().detach()
        stats = {"wilson_G": g.mean()}
        if self.n_scale_groups > 1:
            # the spread across groups is the quantity of interest: it says
            # how much the crystals differ, which one G would have hidden
            stats["wilson_G_min"] = g.min()
            stats["wilson_G_max"] = g.max()
        return {**stats, **super().diagnostics()}

    def _set_scale_from_fit(self, G0: float) -> None:
        with torch.no_grad():
            self.raw_G.copy_(self._raw_from_G(G0).to(self.raw_G.device))

    def _get_tau(
        self, metadata: dict, s_sq: Tensor, device: torch.device
    ) -> Tensor:
        G = self.get_G()
        if self.n_scale_groups > 1:
            labels = metadata.get("sweep_id")
            if labels is None:
                raise ValueError(
                    "n_scale_groups > 1 needs metadata['sweep_id'] to say "
                    "which crystal each reflection came from"
                )
            G = G[labels.to(device).long().clamp(0, self.n_scale_groups - 1)]
        else:
            G = G.squeeze()
        B = self.get_B()
        tau = (1.0 / G) * torch.exp(2.0 * B * s_sq)

        if self._apply_lp:
            lp = metadata["lp"].to(device).clamp(min=1e-8)
            tau = tau * lp

        return tau

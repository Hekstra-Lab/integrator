import torch


class DistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        intensity_distribution,
        background_distribution,
        spot_profile_model,
        bg_indicator=None,
        eps=1e-12,
        beta=1.0,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.intensity_distribution = intensity_distribution
        self.background_distribution = background_distribution
        self.spot_profile_model = spot_profile_model
        self.bg_indicator = bg_indicator if bg_indicator is not None else None

    def forward(
        self,
        representation,
        dxyz,
    ):
        bg_profile = (
            self.bg_indicator(representation) if self.bg_indicator is not None else None
        )
        spot_profile, L = self.spot_profile_model(representation, dxyz)
        q_bg = self.background_distribution(representation)
        q_I = self.intensity_distribution(representation)

        return q_bg, q_I, spot_profile, L, bg_profile

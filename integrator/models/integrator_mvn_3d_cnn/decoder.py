import torch


class Decoder(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        q_bg,
        q_I,
        profile,
        bg_profile,
        mc_samples=100,
        use_bg_profile=False,
    ):
        # Sample from variational distributions
        z = q_I.rsample([mc_samples])
        bg = q_bg.rsample([mc_samples])

        if use_bg_profile:
            rate = (
                z.permute(1, 0, 2) * profile.unsqueeze(1)
                + bg.permute(1, 0, 2) * bg_profile
            )
        else:
            rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(1, 0, 2)

        return rate, z, bg

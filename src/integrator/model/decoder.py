import torch


class Decoder(torch.nn.Module):
    def __init__(
        self,
        dirichlet=False,
    ):
        super().__init__()
        self.dirichlet = dirichlet

    def forward(
        self,
        q_I,
        q_bg,
        profile,
        mc_samples=100,
    ):
        # Sample from variational distributions
        z = q_I.rsample([mc_samples]).unsqueeze(-1)
        bg = q_bg.rsample([mc_samples]).unsqueeze(-1)

        if self.dirichlet:
            rate = z.permute(1, 0, 2) * profile.permute(1, 0, 2) + bg.permute(1, 0, 2)
        else:
            rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(1, 0, 2)

        return rate

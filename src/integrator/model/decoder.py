import torch


# class Decoder(torch.nn.Module):
# def __init__(
# self,
# ):
# super().__init__()

# def forward(
# self,
# q_I,
# q_bg,
# profile,
# mc_samples=100,
# ):
# # Sample from variational distributions
# z = q_I.rsample([mc_samples]).unsqueeze(-1)
# bg = q_bg.rsample([mc_samples]).unsqueeze(-1)

# rate = z.permute(1, 0, 2) * profile.unsqueeze(1) + bg.permute(1, 0, 2)

# return rate


class Decoder(torch.nn.Module):
    def __init__(
        self,
        mc_samples=100,
    ):
        super().__init__()

    def forward(
        self,
        q_I,
        q_bg,
        q_p,
        mc_samples=mc_samples,
    ):
        # Sample from variational distributions
        z = q_I.rsample([mc_samples]).unsqueeze(-1)
        bg = q_bg.rsample([mc_samples]).unsqueeze(-1)
        qp = q_p.rsample([mc_samples])

        rate = z.permute(1, 0, 2) * qp.permute(1, 0, 2) + bg.permute(1, 0, 2)

        return rate

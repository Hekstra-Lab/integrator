from pylab import *
import torch


class IntegratorV3(torch.nn.Module):
    """
    Attributes:
        encoder: encodes shoeboxes
        distribution_builder: builds variational distributions and profile
        likelihood: MLE cost function
        counts_std:
    """

    def __init__(
        self,
        encoder,
        distribution_builder,
        likelihood,
    ):
        super().__init__()
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.likelihood = likelihood
        self.counts_std = None

    def set_counts_std(self, value):
        self.counts_std = torch.nn.Parameter(value, requires_grad=False)

    def get_intensity_sigma(self, shoebox):
        # lists to store I and SigI
        I, SigI = [], []

        # pixel coordinates
        xyz = shoebox[..., 0:3]

        # pixel distances to centroid
        dxyz = shoebox[..., 3:6]

        # observed photon counts
        counts = torch.clamp(shoebox[..., -1], min=0)

        for batch in zip(
            torch.split(xyz, batch_size, dim=0),
            torch.split(dxyz, batch_size, dim=0),
            torch.split(counts, batch_size, dim=0),
        ):
            # device where parameters are located
            device = next(self.parameters()).device

            i, s = self.get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
            I.append(i.detach().cpu().numpy())
            SigI.append(s.detach().cpu().numpy())
        I, SigI = np.concatenate(I), np.concatenate(SigI)
        return I, SigI

    def get_per_spot_normalization(self, counts):
        return torch.clamp(counts[:, :, -1], min=0).sum(-1)

    def get_intensity_sigma_batch(
        self,
        shoebox,
        mask,
        mc_samples=100,
    ):
        # photon counts
        counts = torch.clamp(shoebox[..., -1], min=0)

        # distances to centroid
        dxyz = shoebox[..., 3:6]

        # encode shoebox
        representation = self.encoder(shoebox, mask)

        # build q_bg, q_I, and profile
        q_bg, q_I, profile = self.distribution_builder(representation, dxyz)

        I, SigI = q_I.mean, q_I.stddev

        ll, kl_term = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            mc_samples=mc_samples,
            mask=mask,
        )
        nll = -ll.mean()

        return I, SigI, profile, counts, (nll + kl_term)

    def forward(
        self,
        shoebox,
        mask,
        mc_samples=100,
    ):
        # todo
        # Do not clamp counts
        counts = torch.clamp(shoebox[..., -1], min=0)

        # distances to centroid
        dxyz = shoebox[..., 3:6]

        # encode shoebox
        representation = self.encoder(shoebox, mask)

        # build q_I, q_bg, and profile
        q_bg, q_I, profile = self.distribution_builder(representation, dxyz)

        ll, kl_term = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            mc_samples=mc_samples,
            mask=mask,
        )
        nll = -ll.mean()

        return nll + kl_term

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm

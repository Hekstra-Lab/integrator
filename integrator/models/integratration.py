from pylab import *
import torch
from integrator.layers import Standardize


class IntegratorV3(torch.nn.Module):
    """
    Integration module

    Attributes:
        encoder (torch.nn.Module): Encodes shoeboxes.
        distribution_builder (torch.nn.Module): Builds variational distributions and profile.
        likelihood (torch.nn.Module): MLE cost function.
        counts_std (torch.nn.Parameter): Standard deviation of counts. Not trainable.
    """

    def __init__(
        self,
        standardize,
        encoder,
        distribution_builder,
        likelihood,
    ):
        super().__init__()
        self.standardize = standardize
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.likelihood = likelihood
        self.counts_std = None

    def set_counts_std(self, value):
        """
        Set the standard deviation of counts.

        Args:
            value (torch.Tensor): Value to set as the standard deviation of counts.
        """
        self.counts_std = torch.nn.Parameter(value, requires_grad=False)

    def get_intensity_sigma(self, shoebox, dead_pixel_mask):
        """
        Get the intensity and sigma values for the shoebox.

        Args:
            shoebox (torch.Tensor): Shoebox tensor.

        Returns:
            tuple: Intensity and sigma values.
        """
        # lists to store I and SigI
        I, SigI = [], []

        # observed photon counts
        counts = torch.clamp(shoebox[..., -1], min=0)

        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))

        # pixel coordinates
        xyz = shoebox_[..., 0:3]

        # pixel distances to centroid
        dxyz = shoebox_[..., 3:6]

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
        """
        Get the per-spot normalization.

        Args:
            counts (torch.Tensor): Counts tensor.

        Returns:
            torch.Tensor: Per-spot normalization.
        """
        return torch.clamp(counts[:, :, -1], min=0).sum(-1)

    def get_intensity_sigma_batch(
        self,
        shoebox,
        dead_pixel_mask,
        mc_samples=100,
    ):
        """
        Get the intensity, sigma, profile, counts, and loss for a batch of shoeboxes.

        Args:
            shoebox (torch.Tensor): Shoebox tensor.
            mask (torch.Tensor): Mask tensor.
            mc_samples (int): Number of Monte Carlo samples. Defaults to 100.

        Returns:
            tuple: Intensity, sigma, profile, counts, and loss.
        """
        # photon counts
        counts = torch.clamp(shoebox[..., -1], min=0)

        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
        # distances to centroid
        dxyz = shoebox_[..., 3:6]

        # encode shoebox
        representation = self.encoder(shoebox_, dead_pixel_mask)

        # build q_bg, q_I, and profile
        q_bg, q_I, profile = self.distribution_builder(representation, dxyz)

        I, SigI = q_I.mean, q_I.stddev

        ll, kl_term = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            mc_samples=mc_samples,
            mask=dead_pixel_mask.squeeze(-1),
        )
        nll = -ll.mean()

        return I, SigI, profile, counts, (nll + kl_term)

    def forward(
        self,
        shoebox,
        padding_mask,
        dead_pixel_mask,
        mc_samples=100,
    ):
        """
        Forward pass of the integrator.

        Args:
            shoebox (torch.Tensor): Shoebox tensor
            padding_mask (torch.Tensor): Mask of padded entries
            dead_pixel_mask (torch.Tensor): Mask of dead pixels and padded entries
            mc_samples (int): Number of Monte Carlo samples. Defaults to 100.

        Returns:
            torch.Tensor: Negative log-likelihood loss
        """

        # Do not clamp counts
        counts = shoebox[..., -1]

        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))

        # distances to centroid
        dxyz = shoebox_[..., 3:6]

        # encode shoebox
        representation = self.encoder(shoebox_, dead_pixel_mask)

        # build q_I, q_bg, and profile
        q_bg, q_I, profile = self.distribution_builder(representation, dxyz)

        ll, kl_term = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            mc_samples=mc_samples,
            mask=dead_pixel_mask.squeeze(-1),
        )
        nll = -ll.mean()

        return nll + kl_term

    def grad_norm(self):
        """
        Calculate the gradient norm of the model parameters.

        Returns:
            torch.Tensor: Gradient norm.
        """
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm

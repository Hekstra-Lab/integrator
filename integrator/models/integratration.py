from pylab import *
import torch
from integrator.layers import Standardize


class Integrator(torch.nn.Module):
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

    def check_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm().item()}")
            else:
                print(f"{name} has no gradients")

    def forward(
        self,
        shoebox,
        dead_pixel_mask,
        is_flat,
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

        # get counts
        counts = torch.clamp(shoebox[..., -1], min=0)
        # dxyz = shoebox[..., 3:6]

        # standardize data
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
        dxyz = shoebox[..., 3:6]
        # dxy = shoebox_[..., 3:5]
        # dxy = shoebox[..., 3:5]

        # distances to centroid

        # encode shoebox
        representation = self.encoder(shoebox_, dead_pixel_mask.unsqueeze(-1))

        # build q_I, q_bg, and profile
        q_bg, q_I, profile, L = self.distribution_builder(
            representation, dxyz, dead_pixel_mask, is_flat
        )
        # q_I, profile = self.distribution_builder(representation, dxyz, dead_pixel_mask)

        # calculate ll and kl
        ll, kl_term, rate_ = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            L,
            mc_samples=mc_samples,
            mask=dead_pixel_mask.squeeze(-1),
        )

        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(
            -1
        )  # mean across mc_samples
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        if self.training == True:
            # return nll + kl_term
            return (nll + kl_term, rate_, q_I, profile, q_bg, counts, L)

        else:
            return (nll + kl_term, rate_, q_I, profile, q_bg, counts, L)

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

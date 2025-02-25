import torch
from integrator.model.loss import BaseLoss
from typing import Optional, Dict, Any, Union, Tuple


class Loss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        p_bg: Optional[torch.distributions.Distribution] = None,
        p_I: Optional[torch.distributions.Distribution] = None,
        p_p: Optional[torch.distributions.Distribution] = None,
        # Surrogate/Prior pairings
        bg_pairing: str = "gamma_gamma",
        I_pairing: str = "gamma_gamma",
        p_pairing: Optional[str] = None,
        # Scale parameters
        p_p_scale: float = 0.001,
        p_bg_scale: float = 0.0001,
        p_I_scale: float = 0.0001,
        recon_scale: float = 0.00,
        simpson_scale: Optional[float] = None,
        tv_loss_scale: Optional[float] = None,
    ):
        super().__init__()

        # Don't specify device in __init__, let PyTorch handle it
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("recon_scale", torch.tensor(recon_scale))
        self.register_buffer("beta", torch.tensor(beta))

        # Scale parameters
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Prior distributions
        # Move tensors to appropriate device in forward pass
        self.p_bg = p_bg
        self.p_I = p_I
        self.p_p = p_p

        self.bg_pairing = bg_pairing
        self.I_pairing = I_pairing
        self.p_pairing = p_pairing

        # Dirichlet prior parameters will be moved to correct device in forward
        if p_pairing == "dirichlet_dirichlet" and p_p is None:
            self.register_buffer(
                "dirichlet_concentration", torch.ones(3 * 21 * 21) * 0.1
            )

        self.simpson_scale = simpson_scale
        self.tv_loss_scale = tv_loss_scale

    def to(self, device):
        """
        Override to() to ensure all distribution parameters are moved to the correct device.
        """
        super().to(device)

        # Move prior distributions to device
        if self.p_bg is not None:
            self._move_distribution_to_device(self.p_bg, device)

        if self.p_I is not None:
            self._move_distribution_to_device(self.p_I, device)

        if self.p_p is not None:
            self._move_distribution_to_device(self.p_p, device)

        return self

    def _move_distribution_to_device(self, dist, device):
        """Helper method to move distribution parameters to the specified device"""
        # Check which type of distribution and move accordingly
        if isinstance(dist, torch.distributions.gamma.Gamma):
            dist.concentration = dist.concentration.to(device)
            dist.rate = dist.rate.to(device)
        elif isinstance(dist, torch.distributions.log_normal.LogNormal):
            dist.loc = dist.loc.to(device)
            dist.scale = dist.scale.to(device)
        elif isinstance(dist, torch.distributions.dirichlet.Dirichlet):
            dist.concentration = dist.concentration.to(device)
        elif isinstance(dist, torch.distributions.beta.Beta):
            dist.concentration1 = dist.concentration1.to(device)
            dist.concentration0 = dist.concentration0.to(device)
        elif isinstance(dist, torch.distributions.laplace.Laplace):
            dist.loc = dist.loc.to(device)
            dist.scale = dist.scale.to(device)

    def inverse_simpson_regularization(
        self, p: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Args:
            p: Tensor of shape (batch, depth, height, width)
        Returns:
            inv_simpson_reg: Scalar regularization term to minimize.
        """
        batch_size = p.shape[0]
        p_flat = p.view(batch_size, -1)  # Shape: (batch, num_components)
        simpson = torch.sum(p_flat**2, dim=1)  # Shape: (batch,)
        inv_simpson = 1.0 / (simpson + eps)  # Add eps to avoid division by zero
        return inv_simpson  # Return per-batch values, don't reduce yet

    def total_variation_3d(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Tensor of shape (batch, depth, height, width)
        Returns:
            tv_loss: Tensor of shape (batch,) with per-batch TV loss
        """
        batch_size = volume.shape[0]

        # Initialize per-batch TV loss
        batch_tv_loss = torch.zeros(batch_size, device=volume.device)

        for b in range(batch_size):
            # Get single example volume
            single_vol = volume[b : b + 1]

            # Calculate TV components for this example
            diff_depth = torch.abs(single_vol[:, 1:, :, :] - single_vol[:, :-1, :, :])
            diff_height = torch.abs(single_vol[:, :, 1:, :] - single_vol[:, :, :-1, :])
            diff_width = torch.abs(single_vol[:, :, :, 1:] - single_vol[:, :, :, :-1])

            # Sum for this example and store
            batch_tv_loss[b] = diff_depth.sum() + diff_height.sum() + diff_width.sum()

        return batch_tv_loss

    def _kl_sampling(self, q_dist, p_dist, num_samples=100):
        """
        Calculate KL divergence using sampling when analytic form is not available.

        Args:
            q_dist: The variational (posterior) distribution
            p_dist: The prior distribution
            num_samples: Number of samples for Monte Carlo estimation

        Returns:
            KL divergence estimate, preserving batch dimension
        """
        # Monte Carlo approximation: E_q[log q(x) - log p(x)]
        samples = q_dist.rsample((num_samples,))
        log_q = q_dist.log_prob(samples)
        log_p = p_dist.log_prob(samples)

        # Average across sample dimension only, preserve batch dimension
        return (log_q - log_p).mean(dim=0)

    def compute_kl(self, q_dist, p_dist, pair_type):
        """
        Compute KL divergence based on the surrogate/prior pair type.

        Args:
            q_dist: Surrogate (posterior) distribution
            p_dist: Prior distribution
            pair_type: String indicating the distribution pair type

        Returns:
            KL divergence tensor, preserving batch dimension
        """
        # Cases where we can use PyTorch's built-in KL divergence
        if pair_type in ["gamma_gamma", "dirichlet_dirichlet", "beta_beta"]:
            kl = torch.distributions.kl.kl_divergence(q_dist, p_dist)
            return kl

        # Normalized Beta / Dirichlet pairing (sampling-based approach)
        elif pair_type == "normalized_beta_dirichlet":
            return self._kl_sampling(q_dist, p_dist)

        # Cases where we need to use sampling-based KL divergence
        elif pair_type in ["lognormal_gamma", "beta_laplace"]:
            return self._kl_sampling(q_dist, p_dist)

        else:
            raise ValueError(f"Unsupported KL divergence pair type: {pair_type}")

    def _ensure_batch_dim(self, tensor, batch_size, device):
        """
        Ensure tensor has batch dimension, broadcasting if needed.

        Args:
            tensor: Input tensor to check
            batch_size: Expected batch size
            device: Device for the tensor

        Returns:
            Tensor with batch dimension of size batch_size
        """
        if tensor.dim() == 0:  # Scalar tensor
            return tensor.expand(batch_size).to(device)

        if tensor.dim() == 1 and tensor.size(0) == batch_size:
            return tensor  # Already correct shape

        # If tensor has complex shape, reduce all but batch dimension
        if tensor.dim() > 1 and tensor.size(0) == batch_size:
            non_batch_dims = list(range(1, tensor.dim()))
            if non_batch_dims:
                return tensor.mean(dim=non_batch_dims)

        # Default case: something unexpected, return broadcasted tensor
        return tensor.expand(batch_size).to(device)

    def forward(self, rate, counts, q_p, q_I, q_bg, dead_pixel_mask):
        """
        Forward pass that preserves batch dimension until final reduction.
        """
        # Ensure all inputs are on the same device
        device = rate.device
        batch_size = rate.shape[0]

        # Initialize Dirichlet prior if needed
        if self.p_pairing == "dirichlet_dirichlet":
            if self.p_p is None:
                self.p_p = torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )
            else:
                # even if it was already initialized
                if self.p_p.concentration.device != device:
                    self.p_p.concentration = self.p_p.concentration.to(device)

        # Ensure other components are on the correct device
        counts = counts.to(device)
        dead_pixel_mask = dead_pixel_mask.to(device)

        # Calculate log likelihood
        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        # Calculate all KL terms, keeping batch dimension intact
        kl_terms = torch.zeros(batch_size, device=device)

        # Only calculate profile KL if we have both the surrogate and prior distributions
        if self.p_p is not None and self.p_pairing is not None:
            kl_p = self.compute_kl(q_p, self.p_p, self.p_pairing)
            # Ensure kl_p has batch dimension
            kl_p = self._ensure_batch_dim(kl_p, batch_size, device)
            kl_terms += kl_p * self.p_p_scale

        # Calculate background KL divergence
        kl_bg = self.compute_kl(q_bg, self.p_bg, self.bg_pairing)
        kl_bg = self._ensure_batch_dim(kl_bg, batch_size, device)
        kl_terms += kl_bg * self.p_bg_scale

        # Calculate intensity KL divergence
        kl_I = self.compute_kl(q_I, self.p_I, self.I_pairing)
        kl_I = self._ensure_batch_dim(kl_I, batch_size, device)
        kl_terms += kl_I * self.p_I_scale

        # Calculate reconstruction loss (per batch)
        recon_loss_batch = torch.abs(rate.mean(1) - counts) / (counts + self.eps)
        recon_loss_batch = (
            recon_loss_batch.mean(dim=1) * self.recon_scale
        )  # Keep batch dimension

        # Initialize additional loss terms (per batch)
        profile_simpson_batch = torch.zeros(batch_size, device=device)
        tv_loss_batch = torch.zeros(batch_size, device=device)

        if self.simpson_scale is not None and self.p_pairing == "dirichlet_dirichlet":
            # Profile simpson regularization (per batch)
            profile_simpson_batch = (
                self.inverse_simpson_regularization(q_p.mean.reshape(-1, 3, 21, 21))
                * self.simpson_scale
            )

        if self.tv_loss_scale is not None:
            # TV loss (per batch)
            tv_loss_batch = (
                self.total_variation_3d(q_p.mean.reshape(-1, 3, 21, 21))
                * self.tv_loss_scale
            )

        # Calculate negative log likelihood (per batch)
        ll_mean = torch.mean(ll, dim=1) * dead_pixel_mask.squeeze(-1)
        # Sum over pixels but keep batch dimension
        neg_ll_batch = -ll_mean.sum(dim=1)

        # Combine all loss terms (per batch)
        batch_loss = (
            neg_ll_batch
            + kl_terms
            + recon_loss_batch
            + tv_loss_batch
            + profile_simpson_batch
        )

        # Now reduce to scalar for final loss
        total_loss = batch_loss.mean()

        # Match your original return format, but use batch-preserved values where possible
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            recon_loss_batch.mean(),
            kl_bg.mean(),
            kl_I.mean(),
            kl_p.mean() if self.p_pairing == "dirichlet_dirichlet" else 0.0,
        )

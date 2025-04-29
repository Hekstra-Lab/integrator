import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Dirichlet, Gamma, LogNormal
from integrator.model.encoders import MLPImageEncoder, MLPMetadataEncoder


def create_center_focused_dirichlet_prior(
    shape=(3, 21, 21),
    base_alpha=0.1,  # outer region
    center_alpha=100.0,  # high alpha at the center => center gets more mass
    decay_factor=1,
    peak_percentage=0.1,
):
    channels, height, width = shape
    alpha_3d = np.ones(shape) * base_alpha

    # center indices
    center_c = channels // 2
    center_h = height // 2
    center_w = width // 2

    # loop over voxels
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                # Normalized distance from center
                dist_c = abs(c - center_c) / (channels / 2)
                dist_h = abs(h - center_h) / (height / 2)
                dist_w = abs(w - center_w) / (width / 2)
                distance = np.sqrt(dist_c**2 + dist_h**2 + dist_w**2) / np.sqrt(3)

                if distance < peak_percentage * 5:
                    alpha_value = (
                        center_alpha
                        - (center_alpha - base_alpha)
                        * (distance / (peak_percentage * 5)) ** decay_factor
                    )
                    alpha_3d[c, h, w] = alpha_value

    alpha_vector = torch.tensor(alpha_3d.flatten(), dtype=torch.float32)
    return alpha_vector


class Loss2(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        # Profile prior
        p_p_name=None,  # Type: "dirichlet", "beta", or None
        p_p_params=None,  # Parameters for the distribution
        p_p_scale=0.0001,
        # Background prior
        p_bg_name="gamma",
        p_bg_params={"concentration": 1.0, "rate": 1.0},
        p_bg_scale=0.0001,
        # Intensity prior
        use_center_focused_prior=True,
        prior_shape=(3, 21, 21),
        prior_base_alpha=0.1,
        prior_center_alpha=50.0,
        prior_decay_factor=0.4,
        prior_peak_percentage=0.026,
        p_I_name="gamma",
        p_I_params={"concentration": 1.0, "rate": 1.0},
        p_I_scale=0.0001,
        prior_tensor=None,
        use_robust=False,
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_I_rate", torch.tensor(p_I_params["rate"]))

        # Store distribution names and params
        self.p_p_name = p_p_name
        self.p_p_params = p_p_params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params
        self.p_I_name = p_I_name
        self.p_I_params = p_I_params
        if prior_tensor is not None:
            self.concentration = torch.load(prior_tensor, weights_only=False)
        else:
            self.concentration = torch.ones(1323) * p_p_params["concentration"]

        self._register_distribution_params(p_bg_name, p_bg_params, prefix="p_bg_")
        self._register_distribution_params(p_I_name, p_I_params, prefix="p_I_")

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]
        self.use_robust = use_robust

        # Handle profile prior (p_p) - special handling for Dirichlet
        # Create center-focused Dirichlet prior
        alpha_vector = create_center_focused_dirichlet_prior(
            shape=prior_shape,
            base_alpha=prior_base_alpha,
            center_alpha=prior_center_alpha,
            decay_factor=prior_decay_factor,
            peak_percentage=prior_peak_percentage,
        )

        self.register_buffer("dirichlet_concentration", alpha_vector)

        # Store shape for profile reshaping
        self.prior_shape = prior_shape

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def _register_distribution_params(self, name, params, prefix):
        """Register distribution parameters as buffers with appropriate prefixes"""
        if name is None or params is None:
            return
        if name == "gamma":
            self.register_buffer(
                f"{prefix}concentration", torch.tensor(params["concentration"])
            )
            self.register_buffer(f"{prefix}rate", torch.tensor(params["rate"]))
        elif name == "log_normal":
            self.register_buffer(f"{prefix}loc", torch.tensor(params["loc"]))
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))
        elif name == "exponential":
            self.register_buffer(f"{prefix}rate", torch.tensor(params["rate"]))
        elif name == "half_normal":
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))

        elif name == "normal":
            self.register_buffer(f"{prefix}loc", torch.tensor(params["loc"]))
            self.register_buffer(f"{prefix}scale", torch.tensor(params["scale"]))

    def get_prior(self, name, params_prefix, device, default_return=None):
        """Create a distribution on the specified device"""
        if name is None:
            return default_return
        if name == "gamma":
            concentration = getattr(self, f"{params_prefix}concentration").to(device)
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.gamma.Gamma(
                concentration=concentration, rate=rate
            )
        elif name == "log_normal":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.log_normal.LogNormal(loc=loc, scale=scale)

        elif name == "exponential":
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.exponential.Exponential(rate=rate)
        elif name == "half_normal":
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.half_normal.HalfNormal(scale=scale)

        elif name == "beta":
            concentration1 = getattr(self, f"{params_prefix}concentration1").to(device)
            concentration0 = getattr(self, f"{params_prefix}concentration0").to(device)
            return torch.distributions.beta.Beta(
                concentration1=concentration1, concentration0=concentration0
            )
        elif name == "laplace":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.laplace.Laplace(loc=loc, scale=scale)
        elif name == "dirichlet":
            # For Dirichlet, use the dirichlet_concentration buffer
            if hasattr(self, "dirichlet_concentration"):
                return torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

        # Default case: return None or provided default
        return default_return

    def forward(
        self,
        rate,
        counts,
        q_p,
        q_I,
        q_bg,
        masks,
    ):
        # get device and batch size
        device = rate.device
        batch_size = rate.shape[0]
        self.current_batch_size = batch_size

        counts = counts.to(device)
        masks = masks.to(device)

        p_p = torch.distributions.dirichlet.Dirichlet(self.concentration.to(device))
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)
        p_I = self.get_prior(self.p_I_name, "p_I_", device)

        # calculate kl terms
        kl_terms = torch.zeros(batch_size, device=device)

        kl_I = self.compute_kl(q_I, p_I)
        kl_terms += kl_I * self.p_I_scale

        # calculate background and intensity kl divergence
        kl_bg = self.compute_kl(q_bg, p_bg)
        kl_bg = kl_bg.sum(-1)
        kl_terms += kl_bg * self.p_bg_scale

        kl_p = self.compute_kl(q_p, p_p)
        kl_terms += kl_p * self.p_p_scale

        if self.use_robust:
            counts_unsqueezed = counts.unsqueeze(1)
            rates = rate + self.eps

            dispersion = torch.tensor(1.0)

            # Negative binomial log probability formula
            # P(X=k) = Gamma(k+r)/(Gamma(r)*Gamma(k+1)) * (r/(r+μ))^r * (μ/(r+μ))^k
            # where r is dispersion, μ is rate, k is counts

            # log P(X=k) = log(Gamma(k+r)) - log(Gamma(r)) - log(Gamma(k+1)) + r*log(r/(r+μ)) + k*log(μ/(r+μ))
            term1 = torch.lgamma(counts_unsqueezed + dispersion)
            term2 = torch.lgamma(dispersion)
            term3 = torch.lgamma(counts_unsqueezed + 1)
            term4 = dispersion * torch.log(dispersion / (dispersion + rates))
            term5 = counts_unsqueezed * torch.log(rates / (dispersion + rates))

            log_prob = term1 - term2 - term3 + term4 + term5
            log_prob = torch.clamp(
                log_prob, min=-1000.0
            )  # Prevent extreme negative values

        else:
            ll = torch.distributions.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            # calculate negative log likelihood
            neg_ll_batch = (-ll_mean).sum(1)

            # combine all loss terms
            batch_loss = neg_ll_batch + kl_terms

        # Calculate mean over batch dimension and apply masks
        ll_mean = torch.mean(log_prob, dim=1) * masks.squeeze(-1)

        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum(1)

        # combine all loss terms
        batch_loss = neg_ll_batch + kl_terms

        # final scalar loss
        total_loss = batch_loss.mean()

        # return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean() * self.p_bg_scale,
            kl_I.mean() * self.p_I_scale,
            kl_p.mean() * self.p_p_scale,
        )

    # def forward(
    # self,
    # rate,
    # counts,
    # q_p,
    # q_I,
    # q_bg,
    # masks,
    # ):
    # # get device and batch size
    # device = rate.device
    # batch_size = rate.shape[0]
    # self.current_batch_size = batch_size

    # counts = counts.to(device)
    # masks = masks.to(device)

    # p_p = torch.distributions.dirichlet.Dirichlet(self.concentration.to(device))

    # # p_bg = torch.distributions.half_normal.HalfNormal(
    # # scale=torch.tensor(1.0, device=device)
    # # )

    # p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)
    # p_I = self.get_prior(self.p_I_name, "p_I_", device)

    # # calculate kl terms
    # kl_terms = torch.zeros(batch_size, device=device)

    # kl_I = self.compute_kl(q_I, p_I)
    # kl_terms += kl_I * self.p_I_scale

    # # calculate background and intensity kl divergence
    # kl_bg = self.compute_kl(q_bg, p_bg)
    # kl_bg = kl_bg.sum(-1)
    # kl_terms += kl_bg * self.p_bg_scale

    # kl_p = self.compute_kl(q_p, p_p)
    # kl_terms += kl_p * self.p_p_scale

    # ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
    # ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)

    # # calculate negative log likelihood
    # neg_ll_batch = (-ll_mean).sum(1)

    # # combine all loss terms
    # batch_loss = neg_ll_batch + kl_terms

    # # final scalar loss
    # total_loss = batch_loss.mean()

    # # return all components for monitoring
    # return (
    # total_loss,
    # neg_ll_batch.mean(),
    # kl_terms.mean(),
    # kl_bg.mean() * self.p_bg_scale,
    # kl_I.mean() * self.p_I_scale,
    # kl_p.mean() * self.p_p_scale,
    # )

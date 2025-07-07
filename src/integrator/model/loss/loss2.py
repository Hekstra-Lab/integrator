import numpy as np
import torch


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
        pprf_name=None,
        pprf_params=None,
        pprf_weight=0.0001,
        pbg_name="gamma",
        pbg_params={"concentration": 1.0, "rate": 1.0},
        pbg_weight=0.0001,
        prior_base_alpha=0.1,
        prior_center_alpha=50.0,
        prior_decay_factor=0.4,
        prior_peak_percentage=0.026,
        pi_name="gamma",
        pi_params={"concentration": 1.0, "rate": 1.0},
        pi_weight=0.001,
        prior_tensor=None,
        use_robust=False,
        shape=(3, 21, 21),
    ):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("pbg_weight", torch.tensor(pbg_weight))
        self.register_buffer("pprf_weight", torch.tensor(pprf_weight))
        # self.register_buffer("pi_scale", pi_scale)
        self.pi_weight = pi_weight

        # Store distribution names and params
        self.pprf_name = pprf_name
        self.pprf_params = pprf_params
        self.pbg_name = pbg_name
        self.pbg_params = pbg_params
        self.pi_name = pi_name
        self.pi_params = pi_params
        if prior_tensor is not None:
            self.concentration = torch.load(prior_tensor, weights_only=False)
            self.concentration[
                self.concentration > torch.quantile(self.concentration, 0.99)
            ] *= 40
            self.concentration /= self.concentration.max()
        else:
            self.concentration = (
                torch.ones(shape[0] * shape[1] * shape[2])
                * pprf_params["concentration"]
            )

        self._register_distribution_params(pbg_name, pbg_params, prefix="pbg_")
        self._register_distribution_params(pi_name, pi_params, prefix="pi_")

        # Number of elements in the profile
        self.profile_size = shape[0] * shape[1] * shape[2]

        # Number of elements in the profile
        self.profile_size = shape[0] * shape[1] * shape[2]
        self.use_robust = use_robust

        # Handle profile prior (pprf) - special handling for Dirichlet
        # Create center-focused Dirichlet prior
        alpha_vector = create_center_focused_dirichlet_prior(
            shape=shape,
            base_alpha=prior_base_alpha,
            center_alpha=prior_center_alpha,
            decay_factor=prior_decay_factor,
            peak_percentage=prior_peak_percentage,
        )

        self.register_buffer("dirichlet_concentration", alpha_vector)

        # Store shape for profile reshaping
        self.prior_shape = shape

    def compute_kl(self, q_dist, pdist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, pdist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = pdist.log_prob(samples)
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
        elif name == "half_cauchy":
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
        elif name == "half_cauchy":
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.half_cauchy.HalfCauchy(scale=scale)
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
        return default_return

    def forward(
        self,
        rate,
        counts,
        q_p,
        q_i,
        q_bg,
        masks,
    ):
        # get device and batch size
        device = rate.device
        batch_size = rate.shape[0]
        self.current_batch_size = batch_size

        counts = counts.to(device)
        masks = masks.to(device)

        pprf = torch.distributions.dirichlet.Dirichlet(self.concentration.to(device))
        pbg = self.get_prior(self.pbg_name, "pbg_", device)
        pi = self.get_prior(self.pi_name, "pi_", device)

        # calculate kl terms
        kl_terms = torch.zeros(batch_size, device=device)

        kl_i = self.compute_kl(q_i, pi)
        kl_terms += kl_i * self.pi_weight

        # calculate background and intensity kl divergence
        kl_bg = self.compute_kl(q_bg, pbg)
        kl_terms += kl_bg * self.pbg_weight

        kl_p = self.compute_kl(q_p, pprf)
        kl_terms += kl_p * self.pprf_weight

        log_prob = torch.distributions.Poisson(rate + self.eps).log_prob(
            counts.unsqueeze(1)
        )

        ll_mean = torch.mean(log_prob, dim=1) * masks.squeeze(-1)

        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum(1)
        neg_ll_batch = neg_ll_batch

        # combine all loss terms
        batch_loss = neg_ll_batch + kl_terms

        # final scalar loss
        total_loss = batch_loss.mean()

        # return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            (kl_bg * self.pbg_weight).mean(),
            kl_i.mean() * self.pi_weight,
            (kl_p * self.pprf_weight).mean(),
        )

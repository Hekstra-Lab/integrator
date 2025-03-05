import torch.nn as nn
import torch


class tempBernoulliLoss(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        p_bg: torch.distributions.Distribution | None = None,
        p_I: torch.distributions.Distribution | None = None,
        p_p: torch.distributions.Distribution | None = None,
        p_z_prior=0.95,
        p_p_scale=1.0,
        p_bg_scale=1.0,
        p_I_scale=0.001,
        p_z_scale=50.0,
        entropy_scale=0.01,  # New parameter for entropy regularization
        intensity_reg_scale=0.005,  # New parameter for intensity regularization
    ):
        super().__init__()
        # Save prior probability for signal existence
        self.register_buffer("p_z_prior", torch.tensor(p_z_prior))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))
        self.register_buffer("p_z_scale", torch.tensor(p_z_scale))
        self.register_buffer("entropy_scale", torch.tensor(entropy_scale))
        self.register_buffer("intensity_reg_scale", torch.tensor(intensity_reg_scale))

        # Prior distributions for intensity and background
        self.p_bg = p_bg
        self.p_I = p_I
        self.p_p = p_p

        # For Dirichlet prior if profile is not provided
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21) * 0.05)

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def compute_profile_entropy(self, profile):
        """
        Compute entropy of profile distribution.
        For probability distributions, higher entropy = more spread out.
        """
        # If profile is a distribution with mean attribute, get the mean
        if hasattr(profile, "mean"):
            profile_values = profile.mean() if callable(profile.mean) else profile.mean
        # If profile is a distribution with probs attribute, get the probs
        elif hasattr(profile, "probs"):
            profile_values = (
                profile.probs() if callable(profile.probs) else profile.probs
            )
        # Else, assume it's already the profile values tensor
        else:
            profile_values = profile

        # Handle different tensor shapes/dimensions
        if profile_values.dim() > 2:
            # Flatten all but the batch dimension
            flat_profile = profile_values.view(profile_values.shape[0], -1)
        else:
            flat_profile = profile_values

        # Normalize if not already normalized (sum to 1)
        sums = flat_profile.sum(dim=-1, keepdim=True)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
            flat_profile = flat_profile / (sums + self.eps)

        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(flat_profile * torch.log(flat_profile + self.eps), dim=-1)
        return entropy

    def compute_profile_spread(self, profile):
        """
        Calculate a measure of how spread out the profile is.
        Higher values mean more spread out profiles.
        Returns negative concentration (sum of squared values).
        """
        # If profile is a distribution with mean attribute, get the mean
        if hasattr(profile, "mean"):
            profile_values = profile.mean() if callable(profile.mean) else profile.mean
        # If profile is a distribution with probs attribute, get the probs
        elif hasattr(profile, "probs"):
            profile_values = (
                profile.probs() if callable(profile.probs) else profile.probs
            )
        # Else, assume it's already the profile values tensor
        else:
            profile_values = profile

        # Handle different tensor shapes/dimensions
        if profile_values.dim() > 2:
            # Flatten all but the batch dimension
            flat_profile = profile_values.view(profile_values.shape[0], -1)
        else:
            flat_profile = profile_values

        # Normalize if not already normalized (sum to 1)
        sums = flat_profile.sum(dim=-1, keepdim=True)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
            flat_profile = flat_profile / (sums + self.eps)

        # Calculate negative concentration: -sum(p²)
        # More concentrated profiles have higher sum of squares
        negative_concentration = -torch.sum(flat_profile**2, dim=-1)
        return negative_concentration

    def forward(self, rate_off, rate_on, z_perm, counts, q_p, q_z, q_I, q_bg, masks):
        # def forward(self, rate, counts, q_p, q_z, q_I, q_bg, masks):
        if hasattr(q_p, "arg_constraints"):
            device = rate.device
            batch_size = rate.shape[0]

            counts = counts.to(device)
            masks = masks.to(device)

            # If no prior for the profile is provided, use a Dirichlet with fixed concentration.
            if self.p_p is None:
                self.p_p = torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

            # ===== Likelihood =====
            # Observed counts are drawn from a Poisson with the predicted rate.
            ll = torch.distributions.poisson.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )
            # Average log likelihood over Monte Carlo samples and pixels, then mask.
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            # ===== KL Divergences =====
            # 1. KL for signal existence (using the Bernoulli closed-form)
            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # 2. KL for signal intensity; weighted by the expected signal probability.
            kl_I_signal = self.compute_kl(q_I, self.p_I)
            weighted_kl_I_signal = q_z.probs * kl_I_signal * self.p_I_scale

            # 3. KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # 4. KL for spatial profile
            kl_p = self.compute_kl(q_p, self.p_p)
            # Sum over spatial dimension if using Beta distributions
            if kl_p.dim() > 1:
                kl_p = kl_p.sum(dim=1)
            weighted_kl_p = q_z.probs * kl_p * self.p_p_scale

            # ===== Entropy Regularization =====
            # Calculate profile entropy - encourage high entropy when signal probability is low
            profile_entropy = self.compute_profile_entropy(q_p)
            # Negative entropy loss (to maximize entropy) weighted by inverse signal probability
            entropy_loss = -self.entropy_scale * (1.0 - q_z.probs) * profile_entropy

            # ===== Intensity-to-Spread Regularization =====
            # Penalize high intensity for spread-out profiles with low signal prob
            intensity_mean = q_I.mean
            profile_spread = self.compute_profile_spread(q_p)
            intensity_reg_loss = (
                self.intensity_reg_scale
                * intensity_mean
                * profile_spread
                * (1.0 - q_z.probs)
            )

            # Combine all regularization terms
            regularization_loss = entropy_loss + intensity_reg_loss

            # Combine all KL terms
            kl_terms = kl_z + weighted_kl_I_signal + kl_bg + weighted_kl_p

            # ===== Total ELBO Loss with Regularization =====
            batch_loss = neg_ll_batch + kl_terms + regularization_loss
            total_loss = batch_loss.mean()

            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I_signal.mean(),
                kl_bg.mean(),
                weighted_kl_p.mean(),
                entropy_loss.mean()
                + intensity_reg_loss.mean(),  # Combined regularization
            )
        else:
            device = counts.device
            batch_size = counts.shape[0]

            # Expand counts to match [batch_size, 1, num_pixels] so it can broadcast with mc_samples
            counts_expanded = counts.unsqueeze(1)  # [b,1,num_pixels]
            # Similarly for masks:
            masks_expanded = masks.unsqueeze(1)  # [b,1,num_pixels]

            # ===== Mixture Log Probability =====
            # 1) log p(x|off)
            log_poisson_off = torch.distributions.Poisson(rate_off + self.eps).log_prob(
                counts_expanded
            )
            # 2) log p(x|on)
            log_poisson_on = torch.distributions.Poisson(rate_on + self.eps).log_prob(
                counts_expanded
            )

            # We have z in shape [b, mc, 1], broadcast it across pixels
            # shape => [b, mc, num_pixels]
            z_prob = z_perm  # rename for clarity

            # mixture = (1-z)*exp(log_poisson_off) + z*exp(log_poisson_on)
            # in log-space: logaddexp( off + log(1-z), on + log(z) )
            log_mixture = torch.logaddexp(
                log_poisson_off + torch.log(1 - z_prob + self.eps),
                log_poisson_on + torch.log(z_prob + self.eps),
            )
            # shape => [b, mc, num_pixels]

            # Now sum or average over pixels, mask, then average over mc_samples:
            # Apply masks:
            log_mixture_masked = log_mixture * masks_expanded
            # mean over mc dimension => dim=1
            # sum over pixel dimension => dim=2
            ll_per_batch = log_mixture_masked.mean(dim=1).sum(dim=1)  # [b]

            neg_ll_batch = -ll_per_batch

            kl_terms = torch.zeros(batch_size, device=device)

            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # Calculate KL terms

            # KL for intensity
            kl_I = self.compute_kl(q_I, self.p_I)
            weighted_kl_I = q_z.probs * kl_I * self.p_I_scale

            # KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # Calculate negative log likelihood

            kl_terms = kl_z + kl_bg + weighted_kl_I

            # Calculate entropy and profile spread regularization
            # For MVN profiles (non-Dirichlet case)
            profile_entropy = self.compute_profile_entropy(q_p)
            entropy_loss = -self.entropy_scale * (1.0 - q_z.probs) * profile_entropy

            # Profile spread regularization
            intensity_mean = q_I.mean
            profile_spread = self.compute_profile_spread(q_p)
            intensity_reg_loss = (
                self.intensity_reg_scale
                * intensity_mean
                * profile_spread
                * (1.0 - q_z.probs)
            )

            # Combine all regularization terms
            regularization_loss = entropy_loss + intensity_reg_loss

            # Combine all KL terms
            kl_terms = kl_z + kl_bg + weighted_kl_I

            # Total loss with regularization
            batch_loss = neg_ll_batch + kl_terms + regularization_loss
            total_loss = batch_loss.mean()

            # Return all components for monitoring
            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I.mean(),
                kl_bg.mean(),
                torch.tensor(0.0, device=device),
                regularization_loss.mean(),  # Add regularization term to returned values
            )


class BernoulliLoss(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        p_bg: torch.distributions.Distribution | None = None,
        p_I: torch.distributions.Distribution | None = None,
        p_p: torch.distributions.Distribution | None = None,
        p_z_prior=0.95,
        p_p_scale=1.0,
        p_bg_scale=1.0,
        p_I_scale=0.001,
        p_z_scale=50.0,
    ):
        super().__init__()
        # Prior probability for signal existence
        self.register_buffer("p_z_prior", torch.tensor(p_z_prior))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))
        self.register_buffer("p_z_scale", torch.tensor(p_z_scale))

        # Prior distributions for intensity and background
        self.p_bg = p_bg
        self.p_I = p_I
        self.p_p = p_p

        # For Dirichlet prior if profile is not provided
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21) * 0.05)

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def forward(self, rate, counts, q_p, q_I, q_bg, masks, q_z=None):
        if hasattr(q_p, "arg_constraints"):
            device = rate.device
            batch_size = rate.shape[0]

            counts = counts.to(device)
            masks = masks.to(device)

            # If no prior for the profile is provided, use a Dirichlet with fixed concentration.
            if self.p_p is None:
                self.p_p = torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

            # ===== Likelihood =====
            # Observed counts are drawn from a Poisson with the predicted rate.
            ll = torch.distributions.poisson.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )
            # Average log likelihood over Monte Carlo samples and pixels, then mask.
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            # ===== KL Divergences =====
            # 1. KL for signal existence (using the Bernoulli closed-form)
            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # 2. KL for signal intensity; weighted by the expected signal probability.
            kl_I_signal = self.compute_kl(q_I, self.p_I)
            weighted_kl_I_signal = q_z.probs * kl_I_signal * self.p_I_scale

            # 3. KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # 4. KL for spatial profile
            kl_p = self.compute_kl(q_p, self.p_p)
            # Sum over spatial dimension if using Beta distributions
            if kl_p.dim() > 1:
                kl_p = kl_p.sum(dim=1)
            weighted_kl_p = q_z.probs * kl_p * self.p_p_scale

            kl_terms = kl_z + weighted_kl_I_signal + kl_bg + weighted_kl_p

            # ===== Total ELBO Loss =====
            batch_loss = neg_ll_batch + kl_terms
            total_loss = batch_loss.mean()

            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I_signal.mean(),
                kl_bg.mean(),
                kl_p.mean(),
            )
        else:
            device = rate.device
            batch_size = rate.shape[0]

            # Calculate log likelihood
            ll = torch.distributions.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )

            kl_terms = torch.zeros(batch_size, device=device)

            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # Calculate KL terms

            # KL for intensity
            kl_I = self.compute_kl(q_I, self.p_I)
            weighted_kl_I = q_z.probs * kl_I * self.p_I_scale

            # KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # Calculate negative log likelihood
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            kl_terms = kl_z + kl_bg + weighted_kl_I

            # Combine all loss terms
            batch_loss = neg_ll_batch + kl_terms
            total_loss = batch_loss.mean()

            # Return all components for monitoring
            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I.mean(),
                kl_bg.mean(),
                torch.tensor(0.0, device=device),
            )


class tempBernoulliLoss(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        p_bg: torch.distributions.Distribution | None = None,
        p_I: torch.distributions.Distribution | None = None,
        p_p: torch.distributions.Distribution | None = None,
        p_z_prior=0.95,
        p_p_scale=1.0,
        p_bg_scale=1.0,
        p_I_scale=0.001,
        p_z_scale=50.0,
        entropy_scale=0.01,  # New parameter for entropy regularization
        intensity_reg_scale=0.005,  # New parameter for intensity regularization
    ):
        super().__init__()
        # Save prior probability for signal existence
        self.register_buffer("p_z_prior", torch.tensor(p_z_prior))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))
        self.register_buffer("p_z_scale", torch.tensor(p_z_scale))
        self.register_buffer("entropy_scale", torch.tensor(entropy_scale))
        self.register_buffer("intensity_reg_scale", torch.tensor(intensity_reg_scale))

        # Prior distributions for intensity and background
        self.p_bg = p_bg
        self.p_I = p_I
        self.p_p = p_p

        # For Dirichlet prior if profile is not provided
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21) * 0.05)

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def compute_profile_entropy(self, profile):
        """
        Compute entropy of profile distribution.
        For probability distributions, higher entropy = more spread out.
        """
        # If profile is a distribution with mean attribute, get the mean
        if hasattr(profile, "mean"):
            profile_values = profile.mean() if callable(profile.mean) else profile.mean
        # If profile is a distribution with probs attribute, get the probs
        elif hasattr(profile, "probs"):
            profile_values = (
                profile.probs() if callable(profile.probs) else profile.probs
            )
        # Else, assume it's already the profile values tensor
        else:
            profile_values = profile

        # Handle different tensor shapes/dimensions
        if profile_values.dim() > 2:
            # Flatten all but the batch dimension
            flat_profile = profile_values.view(profile_values.shape[0], -1)
        else:
            flat_profile = profile_values

        # Normalize if not already normalized (sum to 1)
        sums = flat_profile.sum(dim=-1, keepdim=True)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
            flat_profile = flat_profile / (sums + self.eps)

        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(flat_profile * torch.log(flat_profile + self.eps), dim=-1)
        return entropy

    def compute_profile_spread(self, profile):
        """
        Calculate a measure of how spread out the profile is.
        Higher values mean more spread out profiles.
        Returns negative concentration (sum of squared values).
        """
        # If profile is a distribution with mean attribute, get the mean
        if hasattr(profile, "mean"):
            profile_values = profile.mean() if callable(profile.mean) else profile.mean
        # If profile is a distribution with probs attribute, get the probs
        elif hasattr(profile, "probs"):
            profile_values = (
                profile.probs() if callable(profile.probs) else profile.probs
            )
        # Else, assume it's already the profile values tensor
        else:
            profile_values = profile

        # Handle different tensor shapes/dimensions
        if profile_values.dim() > 2:
            # Flatten all but the batch dimension
            flat_profile = profile_values.view(profile_values.shape[0], -1)
        else:
            flat_profile = profile_values

        # Normalize if not already normalized (sum to 1)
        sums = flat_profile.sum(dim=-1, keepdim=True)
        if not torch.allclose(sums, torch.ones_like(sums), rtol=1e-3):
            flat_profile = flat_profile / (sums + self.eps)

        # Calculate negative concentration: -sum(p²)
        # More concentrated profiles have higher sum of squares
        negative_concentration = -torch.sum(flat_profile**2, dim=-1)
        return negative_concentration

    def forward(self, rate, counts, q_p, q_z, q_I, q_bg, masks):
        if hasattr(q_p, "arg_constraints"):
            device = rate.device
            batch_size = rate.shape[0]

            counts = counts.to(device)
            masks = masks.to(device)

            # If no prior for the profile is provided, use a Dirichlet with fixed concentration.
            if self.p_p is None:
                self.p_p = torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

            # ===== Likelihood =====
            # Observed counts are drawn from a Poisson with the predicted rate.
            ll = torch.distributions.poisson.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )
            # Average log likelihood over Monte Carlo samples and pixels, then mask.
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            # ===== KL Divergences =====
            # 1. KL for signal existence (using the Bernoulli closed-form)
            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # 2. KL for signal intensity; weighted by the expected signal probability.
            kl_I_signal = self.compute_kl(q_I, self.p_I)
            weighted_kl_I_signal = q_z.probs * kl_I_signal * self.p_I_scale

            # 3. KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # 4. KL for spatial profile
            kl_p = self.compute_kl(q_p, self.p_p)
            # Sum over spatial dimension if using Beta distributions
            if kl_p.dim() > 1:
                kl_p = kl_p.sum(dim=1)
            weighted_kl_p = q_z.probs * kl_p * self.p_p_scale

            # ===== Entropy Regularization =====
            # Calculate profile entropy - encourage high entropy when signal probability is low
            profile_entropy = self.compute_profile_entropy(q_p)
            # Negative entropy loss (to maximize entropy) weighted by inverse signal probability
            entropy_loss = -self.entropy_scale * (1.0 - q_z.probs) * profile_entropy

            # ===== Intensity-to-Spread Regularization =====
            # Penalize high intensity for spread-out profiles with low signal prob
            intensity_mean = q_I.mean
            profile_spread = self.compute_profile_spread(q_p)
            intensity_reg_loss = (
                self.intensity_reg_scale
                * intensity_mean
                * profile_spread
                * (1.0 - q_z.probs)
            )

            # Combine all regularization terms
            regularization_loss = entropy_loss + intensity_reg_loss

            # Combine all KL terms
            kl_terms = kl_z + weighted_kl_I_signal + kl_bg + weighted_kl_p

            # ===== Total ELBO Loss with Regularization =====
            batch_loss = neg_ll_batch + kl_terms + regularization_loss
            total_loss = batch_loss.mean()

            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I_signal.mean(),
                kl_bg.mean(),
                weighted_kl_p.mean(),
                entropy_loss.mean()
                + intensity_reg_loss.mean(),  # Combined regularization
            )
        else:
            device = rate.device
            batch_size = rate.shape[0]

            # Calculate log likelihood
            ll = torch.distributions.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )

            kl_terms = torch.zeros(batch_size, device=device)

            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # Calculate KL terms
            kl_I = self.compute_kl(q_I, self.p_I)
            weighted_kl_I = q_z.probs * kl_I * self.p_I_scale
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # Calculate negative log likelihood
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            # Calculate entropy and profile spread regularization
            # For MVN profiles (non-Dirichlet case)
            profile_entropy = self.compute_profile_entropy(q_p)
            entropy_loss = -self.entropy_scale * (1.0 - q_z.probs) * profile_entropy

            # Profile spread regularization
            intensity_mean = q_I.mean
            profile_spread = self.compute_profile_spread(q_p)
            intensity_reg_loss = (
                self.intensity_reg_scale
                * intensity_mean
                * profile_spread
                * (1.0 - q_z.probs)
            )

            # Combine all regularization terms
            regularization_loss = entropy_loss + intensity_reg_loss

            # Combine all KL terms
            kl_terms = kl_z + kl_bg + weighted_kl_I

            # Total loss with regularization
            batch_loss = neg_ll_batch + kl_terms + regularization_loss
            total_loss = batch_loss.mean()

            # Return all components for monitoring
            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I.mean(),
                kl_bg.mean(),
                torch.tensor(0.0, device=device),
                regularization_loss.mean(),  # Add regularization term to returned values
            )


class tempBernoulliLoss(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        p_bg: torch.distributions.Distribution | None = None,
        p_I: torch.distributions.Distribution | None = None,
        p_p: torch.distributions.Distribution | None = None,
        p_z_prior=0.95,
        p_p_scale=1.0,
        p_bg_scale=1.0,
        p_I_scale=0.001,
        p_z_scale=50.0,
    ):
        super().__init__()
        # Prior probability for signal existence
        self.register_buffer("p_z_prior", torch.tensor(p_z_prior))
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("p_I_scale", torch.tensor(p_I_scale))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))
        self.register_buffer("p_z_scale", torch.tensor(p_z_scale))

        # Prior distributions for intensity and background
        self.p_bg = p_bg
        self.p_I = p_I
        self.p_p = p_p

        # For Dirichlet prior if profile is not provided
        self.register_buffer("dirichlet_concentration", torch.ones(3 * 21 * 21) * 0.05)

    def compute_kl(self, q_dist, p_dist):
        """Compute KL divergence between distributions, with fallback sampling if needed."""
        try:
            return torch.distributions.kl.kl_divergence(q_dist, p_dist)
        except NotImplementedError:
            samples = q_dist.rsample([100])
            log_q = q_dist.log_prob(samples)
            log_p = p_dist.log_prob(samples)
            return (log_q - log_p).mean(dim=0)

    def forward(self, rate, counts, q_p, q_z, q_I, q_bg, masks):
        if hasattr(q_p, "arg_constraints"):
            device = rate.device
            batch_size = rate.shape[0]

            counts = counts.to(device)
            masks = masks.to(device)

            # If no prior for the profile is provided, use a Dirichlet with fixed concentration.
            if self.p_p is None:
                self.p_p = torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

            # ===== Likelihood =====
            # Observed counts are drawn from a Poisson with the predicted rate.
            ll = torch.distributions.poisson.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )
            # Average log likelihood over Monte Carlo samples and pixels, then mask.
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            # ===== KL Divergences =====
            # 1. KL for signal existence (using the Bernoulli closed-form)
            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # 2. KL for signal intensity; weighted by the expected signal probability.
            kl_I_signal = self.compute_kl(q_I, self.p_I)
            weighted_kl_I_signal = q_z.probs * kl_I_signal * self.p_I_scale

            # 3. KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # 4. KL for spatial profile
            kl_p = self.compute_kl(q_p, self.p_p)
            # Sum over spatial dimension if using Beta distributions
            if kl_p.dim() > 1:
                kl_p = kl_p.sum(dim=1)
            weighted_kl_p = q_z.probs * kl_p * self.p_p_scale

            kl_terms = kl_z + weighted_kl_I_signal + kl_bg + weighted_kl_p

            # ===== Total ELBO Loss =====
            batch_loss = neg_ll_batch + kl_terms
            total_loss = batch_loss.mean()

            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I_signal.mean(),
                kl_bg.mean(),
                kl_p.mean(),
            )
        else:
            device = rate.device
            batch_size = rate.shape[0]

            # Calculate log likelihood
            ll = torch.distributions.Poisson(rate + self.eps).log_prob(
                counts.unsqueeze(1)
            )

            kl_terms = torch.zeros(batch_size, device=device)

            prior_z = torch.ones_like(q_z.probs) * self.p_z_prior
            kl_z = self.p_z_scale * (
                q_z.probs
                * (torch.log(q_z.probs + self.eps) - torch.log(prior_z + self.eps))
                + (1 - q_z.probs)
                * (
                    torch.log(1 - q_z.probs + self.eps)
                    - torch.log(1 - prior_z + self.eps)
                )
            )

            # Calculate KL terms

            # KL for intensity
            kl_I = self.compute_kl(q_I, self.p_I)
            weighted_kl_I = q_z.probs * kl_I * self.p_I_scale

            # KL for background
            kl_bg = self.compute_kl(q_bg, self.p_bg) * self.p_bg_scale

            # Calculate negative log likelihood
            ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
            neg_ll_batch = -ll_mean.sum(dim=1)

            kl_terms = kl_z + kl_bg + weighted_kl_I

            # Combine all loss terms
            batch_loss = neg_ll_batch + kl_terms
            total_loss = batch_loss.mean()

            # Return all components for monitoring
            return (
                total_loss,
                neg_ll_batch.mean(),
                kl_terms.mean(),
                kl_z.mean(),
                weighted_kl_I.mean(),
                kl_bg.mean(),
                torch.tensor(0.0, device=device),
            )

import torch.nn as nn
import torch.nn.functional as F
from integrator.layers import Linear
import torch
from torch.distributions import RelaxedBernoulli, Gamma


class RelaxedBernoulliDistribution(nn.Module):
    def __init__(self, dmodel, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # Signal existence probability network (outputs logits)

        self.signal_net = nn.Sequential(
            Linear(dmodel, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Initialize with negative bias
            # nn.Linear(64, 1),  # Output logits for signal existence
        )

    def forward(self, representation):
        # 1. Predict the signal existence logits and create a relaxed Bernoulli distribution.
        signal_logit = self.signal_net(representation)
        # Use RelaxedBernoulli to get a differentiable approximation to the binary indicator.
        q_z = RelaxedBernoulli(temperature=self.temperature, logits=signal_logit)

        return q_z


class ConfidenceIntegrator(nn.Module):
    def __init__(self, dmodel, temperature=0.1):
        super().__init__()
        self.temperature = temperature  # Temperature for the relaxed Bernoulli

        # Signal existence probability network (outputs logits)
        self.signal_net = nn.Sequential(
            Linear(dmodel, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Initialize with negative bias
            # nn.Linear(64, 1),  # Output logits for signal existence
        )

        # nn.init.constant_(self.signal_net[-1].bias, 1.0)
        # Intensity distribution parameters for the signal (when z=1)
        self.intensity_net = nn.Sequential(
            Linear(dmodel, 64),
            nn.ReLU(),
            Linear(64, 2),  # Outputs for alpha and beta of the Gamma distribution
        )

        # Background distribution parameters (always exists)
        self.background_net = nn.Sequential(
            Linear(dmodel, 64),
            nn.ReLU(),
            Linear(64, 2),  # Outputs for alpha and beta of the Gamma distribution
        )

        # Small constant for numerical stability
        self.eps = 1e-6

    def forward(self, representation):
        batch_size = representation.size(0)

        # 1. Predict the signal existence logits and create a relaxed Bernoulli distribution.
        signal_logit = self.signal_net(representation)
        # Use RelaxedBernoulli to get a differentiable approximation to the binary indicator.
        q_z = RelaxedBernoulli(temperature=self.temperature, logits=signal_logit)
        # Sample a continuous value between 0 and 1
        z_sample = q_z.rsample()

        # 2. Predict intensity parameters for the signal
        int_params = self.intensity_net(representation)
        int_alpha = F.softplus(int_params[:, 0]) + self.eps
        int_beta = F.softplus(int_params[:, 1]) + self.eps
        qI = Gamma(int_alpha, int_beta)
        # Use rsample for reparameterization
        I_sample = qI.rsample().unsqueeze(-1)

        # Effective signal intensity (gated by z)
        effective_signal_intensity = z_sample * I_sample

        # 3. Predict background parameters (always exists)
        bg_params = self.background_net(representation)
        bg_alpha = F.softplus(bg_params[:, 0]) + self.eps
        bg_beta = F.softplus(bg_params[:, 1]) + self.eps
        q_bg = Gamma(bg_alpha, bg_beta)

        # In your generative model, you would use the effective_signal_intensity (multiplied by a spatial profile if needed)
        # added to the background to form the Poisson rate for your observed counts.

        return q_z, qI, q_bg, effective_signal_intensity

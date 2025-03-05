import torch
import torch.nn as nn
from integrator.model.decoders import BaseDecoder


# class BernoulliDecoder(nn.Module):
# def __init__(self, mc_samples=100):
# super().__init__()
# self.mc_samples = mc_samples

# def forward(self, q_z, q_I_signal, q_bg, q_p):
# batch_size = q_z.probs.size(0)

# # ===== Step 1: Sample from all distributions =====
# # Use rsample for the relaxed Bernoulli and reparameterizable Gamma & Dirichlet.
# z_samples = q_z.rsample([self.mc_samples])  # [mc_samples, batch_size, 1]
# I_signal_samples = q_I_signal.rsample([self.mc_samples]).unsqueeze(
# -1
# )  # [mc_samples, batch_size, 1]
# bg_samples = q_bg.rsample([self.mc_samples]).unsqueeze(
# -1
# )  # [mc_samples, batch_size, 1]
# p_samples = q_p.rsample(
# [self.mc_samples]
# )  # [mc_samples, batch_size, num_components]

# # ===== Step 2: Compute effective signal intensity =====
# # This is the "spike-and-slab": when z is near 0, the effective intensity is near zero.
# effective_signal_intensity = (
# z_samples * I_signal_samples
# )  # [mc_samples, batch_size, 1]

# # ===== Step 3: Permute to batch-first ordering =====
# effective_signal_intensity = effective_signal_intensity.permute(
# 1, 0, 2
# )  # [batch_size, mc_samples, 1]
# bg_perm = bg_samples.permute(1, 0, 2)  # [batch_size, mc_samples, 1]
# p_perm = p_samples.permute(1, 0, 2)  # [batch_size, mc_samples, num_components]
# z_perm = z_samples.permute(1, 0, 2)  # [batch_size, mc_samples, 1]

# # ===== Step 4: Calculate rate =====
# # Here the rate is a combination of the effective signal (modulated by profile) and background.
# rate = (
# effective_signal_intensity * p_perm + bg_perm
# )  # [batch_size, mc_samples, num_components]

# return rate, z_perm


class tempBernoulliDecoder(nn.Module):
    def __init__(self, mc_samples=100):
        super().__init__()
        self.mc_samples = mc_samples

    def forward(self, q_z, q_I, q_bg, q_p):
        if hasattr(q_p, "arg_constraints"):
            batch_size = q_z.probs.size(0)

            # ===== Step 1: Sample from all distributions =====
            I_signal_samples = q_I.rsample([self.mc_samples]).unsqueeze(-1)
            bg_samples = q_bg.rsample([self.mc_samples]).unsqueeze(-1)
            z_samples = q_z.rsample([self.mc_samples])
            p_samples = q_p.rsample([self.mc_samples])

            # ===== Step 2: Permute to batch-first ordering =====
            z_perm = z_samples.permute(1, 0, 2)  # [batch_size, mc, 1]
            I_perm = I_signal_samples.permute(1, 0, 2)
            bg_perm = bg_samples.permute(1, 0, 2)
            p_perm = p_samples.permute(1, 0, 2)  # e.g. [batch_size, mc, #pixels]

            # ===== Step 3: Instead of blending with z, define OFF and ON rates =====
            # OFF: bg only
            rate_off = bg_perm

            # ON: bg + I * p
            rate_on = bg_perm + I_perm * p_perm

            # We'll return both, plus z for later usage in KL etc.
            return rate_off, rate_on, z_perm

        else:
            # If you're using a simpler or "deterministic" p,
            # do the analogous logic:
            I_samples = q_I.rsample([self.mc_samples]).unsqueeze(-1)
            bg = q_bg.rsample([self.mc_samples]).unsqueeze(-1)
            z_samples = q_z.rsample([self.mc_samples])

            I_perm = I_samples.permute(1, 0, 2)
            bg_perm = bg.permute(1, 0, 2)
            z_perm = z_samples.permute(1, 0, 2)

            # For profile p, no sampling needed or p is a tensor
            profile_expanded = q_p.unsqueeze(1).expand(-1, self.mc_samples, -1)

            # OFF
            rate_off = bg_perm
            # ON
            rate_on = bg_perm + I_perm * profile_expanded

            return rate_off, rate_on, z_perm


class BernoulliDecoder(nn.Module):
    def __init__(self, mc_samples=100):
        super().__init__()
        self.mc_samples = mc_samples

    def forward(self, q_z, q_I, q_bg, q_p):
        if hasattr(q_p, "arg_constraints"):
            batch_size = q_z.probs.size(0)

            # ===== Step 1: Sample from all distributions =====
            I_signal_samples = q_I.rsample([self.mc_samples]).unsqueeze(
                -1
            )  # [mc_samples, batch_size, 1]
            bg_samples = q_bg.rsample([self.mc_samples]).unsqueeze(
                -1
            )  # [mc_samples, batch_size, 1]
            z_samples = q_z.rsample([self.mc_samples])  # [mc_samples, batch_size, 1]

            p_samples = q_p.rsample(
                [self.mc_samples]
            )  # [mc_samples, batch_size, num_components]

            # ===== Step 2: Permute to batch-first ordering =====
            z_perm = z_samples.permute(1, 0, 2)  # [batch_size, mc_samples, 1]
            I_perm = I_signal_samples.permute(1, 0, 2)  # [batch_size, mc_samples, 1]
            bg_perm = bg_samples.permute(1, 0, 2)  # [batch_size, mc_samples, 1]
            p_perm = p_samples.permute(
                1, 0, 2
            )  # [batch_size, mc_samples, num_components]

            # ===== Step 3: Calculate rate =====
            # Make it explicit that Z modulates both I and P: λ = Z * (I * P) + bg
            # This doesn't change the mathematics, but clarifies the model structure
            signal_component = z_perm * I_perm * p_perm  # Z * I * P
            rate = signal_component + bg_perm

            return rate, z_perm
        else:
            # Sample from variational distributions
            I_samples = q_I.rsample([self.mc_samples]).unsqueeze(-1)

            bg = q_bg.rsample([self.mc_samples]).unsqueeze(-1)

            z_samples = q_z.rsample([self.mc_samples])

            # permute to batch-first ordering
            I_perm = I_samples.permute(1, 0, 2)
            bg_perm = bg.permute(1, 0, 2)
            z_perm = z_samples.permute(1, 0, 2)

            # Use deterministic profile (no sampling needed)
            # Expand profile to match MC samples dimension
            profile_expanded = q_p.unsqueeze(1).expand(-1, self.mc_samples, -1)

            signal_component = z_perm * I_perm * profile_expanded

            rate = signal_component + bg_perm
            return rate, z_perm

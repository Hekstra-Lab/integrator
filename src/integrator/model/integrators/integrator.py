import torch
import numpy as np
import math
import torch.nn as nn
from integrator.model.integrators import BaseIntegrator
from integrator.layers import Linear
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch
from integrator.model.distribution import BaseDistribution
from integrator.model.encoders import CNNResNet2
from integrator.layers import Linear, Constraint
from torch.distributions import Dirichlet, Gamma, LogNormal
from integrator.model.encoders import MLPImageEncoder, MLPMetadataEncoder


# %%
# NOTE: This is temporary code to write a new Dirichlet Integrator


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


def create_center_spike_dirichlet_prior(
    shape=(3, 21, 21),
    outer_alpha: float = 5.0,  # large α → near‐uniform outside
    center_alpha: float = 0.05,  # small α → very spiky at exact center
    peak_radius: float = 0.1,  # fraction of max‐distance defining the “influence” zone
    decay: float = 1.0,  # power‐law exponent for the fall-off
):
    """
    Returns a torch vector of length prod(shape) to be used as a Dirichlet α‐vector,
    with α=center_alpha at the center voxel, rising to α=outer_alpha at distance>=peak_radius.
    """
    # 1) create a grid of normalized distances in [0,1]
    C, H, W = shape
    # center indices
    cc, ch, cw = (np.array(shape) - 1) / 2.0
    # coordinates
    zs = np.arange(C)[:, None, None]
    ys = np.arange(H)[None, :, None]
    xs = np.arange(W)[None, None, :]
    # normalized distances along each axis
    dz = (zs - cc) / (C / 2)
    dy = (ys - ch) / (H / 2)
    dx = (xs - cw) / (W / 2)
    # euclidean distance normalized to [0,1]
    dist = np.sqrt(dx**2 + dy**2 + dz**2) / np.sqrt(3)

    # 2) build α‐map: start at outer_alpha everywhere
    alpha_3d = np.full(shape, outer_alpha, dtype=np.float32)

    # 3) inside the “peak” radius, interpolate down to center_alpha
    mask = dist <= peak_radius
    scaled = dist[mask] / peak_radius  # in [0,1]
    # α(dist) = center_alpha + (outer_alpha - center_alpha) * scaled**decay
    alpha_3d[mask] = center_alpha + (outer_alpha - center_alpha) * (scaled**decay)

    # 4) return as a flat torch tensor
    return torch.tensor(alpha_3d.flatten(), dtype=torch.float32)


class UnetLoss(torch.nn.Module):
    def __init__(
        self,
        beta=1.0,
        eps=1e-5,
        # Profile prior
        p_p_name=None,  # Type: "dirichlet", "beta", or None
        p_p_params=None,  # Parameters for the distribution
        p_p_scale=0.008,
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

        self._register_distribution_params(p_bg_name, p_bg_params, prefix="p_bg_")
        self._register_distribution_params(p_I_name, p_I_params, prefix="p_I_")

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]

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
        elif name == "half_normal":
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.half_normal.HalfNormal(scale=scale)

        elif name == "log_normal":
            loc = getattr(self, f"{params_prefix}loc").to(device)
            scale = getattr(self, f"{params_prefix}scale").to(device)
            return torch.distributions.log_normal.LogNormal(loc=loc, scale=scale)
        elif name == "exponential":
            rate = getattr(self, f"{params_prefix}rate").to(device)
            return torch.distributions.exponential.Exponential(rate=rate)

        elif name == "dirichlet":
            # For Dirichlet, use the dirichlet_concentration buffer
            if hasattr(self, "dirichlet_concentration"):
                return torch.distributions.dirichlet.Dirichlet(
                    self.dirichlet_concentration.to(device)
                )

        # Default case: return None or provided default
        return default_return

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

    def mc_kl(self, q, p, num_samples=10):
        # Sample from q
        samples = q.rsample((num_samples,))
        log_q = q.log_prob(samples)
        log_p = p.log_prob(samples)
        kl_estimate = (log_q - log_p).mean(dim=0)
        return kl_estimate.sum(dim=-1)

    # Then in forward you can do something like:

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

        # p_p = self.get_prior(self.p_p_name, "p_p_", device)

        p_p = torch.distributions.dirichlet.Dirichlet(
            # self.dirichlet_concentration.to(device)
            torch.ones(1323, device=device)
            * 1e-6
        )

        p_bg = torch.distributions.half_normal.HalfNormal(
            scale=torch.tensor(2.0, device=device)
        )
        # p_I = torch.distributions.log_normal.LogNormal(
        # loc=torch.tensor(2.0, device=device),
        # scale=torch.tensor(1.5, device=device),
        # )

        p_I = torch.distributions.gamma.Gamma(
            concentration=torch.tensor(self.p_I_concentration, device=device),
            rate=torch.tensor(self.p_I_rate, device=device),
        )

        # calculate kl terms
        kl_terms = torch.zeros(batch_size, device=device)

        kl_I = torch.distributions.kl.kl_divergence(q_I, p_I)
        # kl_I = self.mc_kl(q_I, p_I, num_samples=100)
        kl_terms += kl_I * self.p_I_scale

        # calculate background and intensity kl divergence
        kl_bg = torch.distributions.kl.kl_divergence(q_bg, p_bg)
        kl_bg = kl_bg.sum(-1)
        kl_terms += kl_bg

        kl_p = torch.distributions.kl.kl_divergence(q_p, p_p)
        kl_terms += kl_p * self.p_p_scale

        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)

        # calculate negative log likelihood
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


class DirichletProfile(torch.nn.Module):
    """
    Dirichlet profile model
    """

    def __init__(self, dmodel=None, num_components=3 * 21 * 21):
        super().__init__()
        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, num_components)
        self.dmodel = dmodel
        self.eps = 1e-6

    def forward(self, alphas):
        if self.dmodel is not None:
            alphas = self.alpha_layer(alphas)
        alphas = F.softplus(alphas) + self.eps
        q_p = torch.distributions.dirichlet.Dirichlet(alphas)

        return q_p


# %%
# dirichlet version
class Integrator(BaseIntegrator):
    def __init__(
        self,
        encoder,
        metadata_encoder,
        loss,
        qbg,
        qp,
        qI,
        # decoder,
        mc_samples=100,
        learning_rate=1e-3,
        image_encoder=None,
        max_iterations=4,
        profile_threshold=0.001,
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "image_encoder",
                "mlp_encoder",
                "profile_model",
                "unet",
                "signal_preprocessor",
            ]
        )
        self.learning_rate = learning_rate
        self.mc_samples = mc_samples

        # Model components
        self.encoder = encoder
        self.qp = DirichletProfile(dmodel=64)
        self.qI = qI
        self.qbg = qbg
        # self.decoder = decoder
        self.automatic_optimization = True
        self.loss_fn = UnetLoss()
        self.max_iterations = max_iterations

    def calculate_intensities(self, counts, qbg, qp, masks):
        with torch.no_grad():
            counts = counts * masks
            # zbg = qbg.rsample([self.mc_samples, 1323]).squeeze(-1).permute(2, 0, 1)
            zbg = qbg.rsample([self.mc_samples]).permute(1, 0, 2)
            zp = qp
            vi = zbg + 1e-6

            # kabsch sum
            for i in range(4):
                num = (counts.unsqueeze(1) - zbg) * zp * masks.unsqueeze(1) / vi
                denom = zp.pow(2) / vi
                I = num.sum(-1) / denom.sum(-1)  # [batch_size, mc_samples]
                vi = (I.unsqueeze(-1) * zp) + zbg
                vi = vi.mean(-1, keepdim=True)
            kabsch_sum_mean = I.mean(-1)
            kabsch_sum_var = I.var(-1)

            # profile masking
            zp = zp * masks.unsqueeze(1)  # profiles
            thresholds = torch.quantile(
                zp, 0.99, dim=-1, keepdim=True
            )  # threshold values
            profile_mask = zp > thresholds
            N_used = profile_mask.sum(-1).float()  # number of pixels per mask
            masked_counts = counts.unsqueeze(1) * profile_mask
            profile_masking_I = (masked_counts - zbg * profile_mask).sum(-1)
            profile_masking_mean = profile_masking_I.mean(-1)
            centered_thresh = profile_masking_I - profile_masking_mean.unsqueeze(-1)
            profile_masking_var = (centered_thresh**2).sum(-1) / (
                N_used.mean(-1) + 1e-6
            )

            intensities = {
                "profile_masking_mean": profile_masking_mean,
                "profile_masking_var": profile_masking_var,
                "kabsch_sum_mean": kabsch_sum_mean,
                "kabsch_sum_var": kabsch_sum_var,
            }

            return intensities

    def forward(self, counts, shoebox, metadata, masks, reference):
        # Unpack batch
        counts = torch.clamp(counts, min=0) * masks
        batch_size = shoebox.shape[0]

        # shoebox = torch.cat([shoebox[:, :, -1], metadata], dim=-1)
        rep = self.encoder(shoebox, masks)

        qbg = self.qbg(rep)
        qp = self.qp(rep)
        qI = self.qI(rep)

        zbg = qbg.rsample([self.mc_samples]).permute(1, 0, 2)
        zp = qp.rsample([self.mc_samples]).permute(1, 0, 2)
        zI = qI.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)

        intensity_mean = qI.mean  # [batch_size]
        intensity_var = qI.variance  # [batch_size]

        rate = zI * zp + zbg

        return {
            "rates": rate,
            "counts": counts,
            "masks": masks,
            "qbg": qbg,
            "qp": qp,
            "qp_mean": qp.mean,
            "qI": qI,
            "intensity_mean": intensity_mean,
            "intensity_mean": intensity_mean,
            "intensity_var": intensity_var,
            "dials_I_sum_value": reference[:, 6],
            "dials_I_sum_var": reference[:, 7],
            "dials_I_prf_value": reference[:, 8],
            "dials_I_prf_var": reference[:, 9],
            "refl_ids": reference[:, -1],
            "profile": qp.mean,
            "zp": zp,
            "qp_factor": torch.tensor(0.0),
            "qp_diag": torch.tensor(0.0),
            "x_c": reference[:, 0],
            "y_c": reference[:, 1],
            "z_c": reference[:, 2],
            "x_c_mm": reference[:, 3],
            "y_c_mm": reference[:, 4],
            "z_c_mm": reference[:, 5],
            "dials_bg_mean": reference[:, 10],
            "dials_bg_sum_value": reference[:, 11],
            "dials_bg_sum_var": reference[:, 12],
            "d": reference[:, 13],
        }

    def training_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch

        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        # Calculate loss
        (loss, neg_ll, kl, kl_bg, kl_I, kl_p) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        # Log metrics
        self.log("train: loss", loss.mean())
        self.log("train: nll", neg_ll.mean())
        self.log("train: kl", kl.mean())
        self.log("train: kl_bg", kl_bg.mean())
        self.log("train: kl_I", kl_I.mean())
        self.log("qI mean mean", outputs["qI"].mean.mean())
        self.log("qI mean min", outputs["qI"].mean.min())
        self.log("qI mean max", outputs["qI"].mean.max())
        self.log("qbg mean mean", outputs["qbg"].mean.mean())
        self.log("qbg mean min", outputs["qbg"].mean.min())
        self.log("qbg mean max", outputs["qbg"].mean.max())
        self.log("qbg variance mean", outputs["qbg"].variance.mean())

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        shoebox, dials, masks, metadata, counts = batch
        # Get model outputs
        outputs = self(shoebox, dials, masks, metadata, counts)

        (
            loss,
            neg_ll,
            kl,
            kl_bg,
            kl_I,
            kl_p,
        ) = self.loss_fn(
            rate=outputs["rates"],
            counts=outputs["counts"],
            q_p=outputs["qp"],
            q_I=outputs["qI"],
            q_bg=outputs["qbg"],
            masks=outputs["masks"],
        )

        # Log metrics
        self.log("val: loss", loss.mean())
        self.log("val: nll", neg_ll.mean())
        self.log("val: kl", kl.mean())
        self.log("val: kl_bg", kl_bg.mean())
        self.log("val: kl_I", kl_I.mean())
        self.log("val: kl_p", kl_p.mean())

        return outputs

    def predict_step(self, batch, batch_idx):
        shoebox, dials, masks, metadata, counts = batch
        outputs = self(shoebox, dials, masks, metadata, counts)
        intensities = self.calculate_intensities(
            counts=outputs["counts"],
            qbg=outputs["qbg"],
            qp=outputs["profile"],
            masks=outputs["masks"],
        )

        return {
            "intensity_mean": outputs["intensity_mean"],  # qI.mean
            "intensity_var": outputs["intensity_var"],  # qI.variance
            "refl_ids": outputs["refl_ids"],
            "dials_I_sum_var": outputs["dials_I_sum_var"],
            "dials_I_prf_value": outputs["dials_I_prf_value"],
            "dials_I_prf_var": outputs["dials_I_prf_var"],
            "qbg": outputs["qbg"].mean,
            "qbg_scale": outputs["qbg"].scale,  # halfnormal param
            "counts": outputs["counts"],
            "profile": outputs["profile"],
            "profile_masking_mean": intensities["profile_masking_mean"],
            "profile_masking_var": intensities["profile_masking_var"],
            "kabsch_sum_mean": intensities["kabsch_sum_mean"],
            "kabsch_sum_var": intensities["kabsch_sum_var"],
            "x_c": outputs["x_c"],
            "y_c": outputs["y_c"],
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

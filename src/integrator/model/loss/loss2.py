import numpy as np
import torch
from torch import Tensor
from torch.distributions import (
    Dirichlet,
    Distribution,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    LogNormal,
)

from integrator.model.loss import BaseLoss


def create_center_focused_dirichlet_prior(
    shape: tuple[int, ...] = (3, 21, 21),
    base_alpha: float = 0.1,  # outer region
    center_alpha: float = 100.0,  # high alpha at the center => center gets more mass
    decay_factor: float = 1.0,
    peak_percentage: float = 0.1,
) -> Tensor:
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


# TODO Remove mutable data structures
class Loss2(BaseLoss):
    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-5,
        pprf_name: str | None = None,
        pprf_params: dict | None = None,
        pprf_weight: float = 0.0001,
        pbg_name: str = "gamma",
        pbg_params: dict = {"concentration": 1.0, "rate": 1.0},
        pbg_weight: float = 0.0001,
        prior_base_alpha: float = 0.1,
        prior_center_alpha: float = 50.0,
        prior_decay_factor: float = 0.4,
        prior_peak_percentage: float = 0.026,
        pi_name: str = "gamma",
        pi_params: dict = {"concentration": 1.0, "rate": 1.0},
        pi_weight: float = 0.001,
        prior_tensor: None = None,
        use_robust: bool = False,
        shape: tuple[int, ...] = (3, 21, 21),
        quantile: float = 0.99,
        pprf_conc_factor: int = 40,
        mc_samples_kl: int = 100,
    ):
        super().__init__(
            mc_samples=mc_samples_kl,
        )

        self.pbg_weight: Tensor
        self.pprf_weight: Tensor
        self.eps: Tensor

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

        if prior_tensor is not None and pprf_params is None:
            self.concentration = torch.load(prior_tensor, weights_only=False)
            self.concentration[
                self.concentration > torch.quantile(self.concentration, quantile)
            ] *= pprf_conc_factor
            self.concentration /= self.concentration.max()
        elif pprf_params is not None:
            self.concentration = (
                torch.ones(shape[0] * shape[1] * shape[2]) * pprf_params["concentration"]
            )

        self._register_distribution_params(pbg_name, pbg_params, prefix="pbg_")
        self._register_distribution_params(pi_name, pi_params, prefix="pi_")

        self.profile_size = shape[0] * shape[1] * shape[2]

        # Number of elements in the profile
        self.use_robust = use_robust

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

    def _register_distribution_params(self, name, params, prefix):
        """Register distribution parameters as buffers with appropriate prefixes"""
        if name is None or params is None:
            return
        if name == "gamma":
            self.register_buffer(f"{prefix}concentration", torch.tensor(params["concentration"]))
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

    def get_prior(
        self,
        name: str,
        params_prefix: str,
        device: torch.device,
    ) -> Distribution:
        """Create a distribution on the specified device"""

        params = {}
        if name == "gamma":
            params = {
                "concentration": getattr(self, f"{params_prefix}concentration").to(device),
                "rate": getattr(self, f"{params_prefix}rate").to(device),
            }
        if name == "log_normal":
            params = {
                "loc": getattr(self, f"{params_prefix}loc").to(device),
                "scale": getattr(self, f"{params_prefix}scale").to(device),
            }
        if name == "half_normal":
            params = {"scale": getattr(self, f"{params_prefix}scale").to(device)}
        if name == "half_cauchy":
            params = {"scale": getattr(self, f"{params_prefix}scale").to(device)}
        if name == "exponential":
            params = {"rate": getattr(self, f"{params_prefix}rate").to(device)}
        if name == "dirichlet":
            params = {"concentration": self.dirichlet_concentration.to(device)}

        distribution_map = {
            "gamma": Gamma,
            "log_normal": LogNormal,
            "exponential": Exponential,
            "dirichlet": Dirichlet,
            "half_normal": HalfNormal,
            "half_cauchy": HalfCauchy,
        }

        return distribution_map[name](**params)

    def forward(
        self,
        rate: Tensor,
        counts: Tensor,
        q_p: Distribution,
        q_i: Distribution,
        q_bg: Distribution,
        masks: Tensor,
    ) -> dict:
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

        log_prob = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))

        ll_mean = torch.mean(log_prob, dim=1) * masks.squeeze(-1)

        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum(1)
        # neg_ll_batch = neg_ll_batch

        # combine all loss terms
        batch_loss = neg_ll_batch + kl_terms

        # final scalar loss
        total_loss = batch_loss.mean()

        # return all components for monitoring
        return {
            "total_loss": total_loss,
            "neg_ll_mean": neg_ll_batch.mean(),
            "kl_mean": kl_terms.mean(),
            "kl_bg_mean": (kl_bg * self.pbg_weight).mean(),
            "kl_i_mean": kl_i.mean() * self.pi_weight,
            "kl_p": (kl_p * self.pprf_weight).mean(),
        }


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    from integrator.model.distributions import (
        DirichletDistribution,
        FoldedNormalDistribution,
    )
    from integrator.model.encoders import IntensityEncoder2D, ShoeboxEncoder2D

    # hyperparameters
    mc_samples = 100

    # encoders
    sbox_encoder_2d = ShoeboxEncoder2D()
    intensity_encoder_2d = IntensityEncoder2D()

    # distributions
    qbg_ = FoldedNormalDistribution(dmodel=64)
    qi_ = FoldedNormalDistribution(dmodel=64)
    qprf_ = DirichletDistribution(dmodel=64, input_shape=(21, 21))

    # generate a random batch
    batch = F.softplus(torch.randn(10, 21 * 21))
    masks = torch.randint(2, (10, 21 * 21))

    # encode batch
    shoebox_rep = sbox_encoder_2d(batch.reshape(batch.shape[0], 1, 21, 21), masks)
    intensity_rep = intensity_encoder_2d(batch.reshape(batch.shape[0], 1, 21, 21), masks)

    # get distributinos
    qbg = qbg_(intensity_rep)
    qi = qi_(intensity_rep)
    qprf = qprf_(shoebox_rep)

    # get samples
    zbg = qbg.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)
    zprf = qprf.rsample([mc_samples]).permute(1, 0, 2)
    zi = qi.rsample([mc_samples]).unsqueeze(-1).permute(1, 0, 2)

    rate = zi * zprf + zbg  # [B,S,Pix]

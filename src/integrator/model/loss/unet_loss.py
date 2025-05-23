import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


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


# %%
class UnetLoss(torch.nn.Module):
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
        prior_center_alpha=5.0,
        prior_decay_factor=0.2,
        prior_peak_percentage=0.05,
        p_I_name="gamma",
        p_I_params={"concentration": 1.0, "rate": 1.0},
        p_I_scale=0.001,
    ):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))
        self.register_buffer("p_bg_scale", torch.tensor(p_bg_scale))
        self.register_buffer("p_p_scale", torch.tensor(p_p_scale))

        # Store distribution names and params
        self.p_bg_name = p_bg_name
        self.p_bg_params = p_bg_params
        self.p_p_name = p_p_name
        self.p_p_params = p_p_params

        # Register parameters for bg distribution
        self._register_distribution_params(p_bg_name, p_bg_params, prefix="p_bg_")

        # Number of elements in the profile
        self.profile_size = prior_shape[0] * prior_shape[1] * prior_shape[2]

        # Handle profile prior (p_p) - special handling for Dirichlet
        if p_p_name == "dirichlet":
            # Check if concentration is provided
            if p_p_params and "concentration" in p_p_params:
                # If concentration is provided, create uniform Dirichlet with that concentration
                alpha_vector = (
                    torch.ones(self.profile_size) * p_p_params["concentration"]
                )
                self.register_buffer("dirichlet_concentration", alpha_vector)
            elif use_center_focused_prior:
                # Create center-focused Dirichlet prior
                alpha_vector = create_center_focused_dirichlet_prior(
                    shape=prior_shape,
                    base_alpha=prior_base_alpha,
                    center_alpha=prior_center_alpha,
                    decay_factor=prior_decay_factor,
                    peak_percentage=prior_peak_percentage,
                )
                self.register_buffer("dirichlet_concentration", alpha_vector)
            else:
                # Default uniform Dirichlet with concentration=1.0
                alpha_vector = torch.ones(self.profile_size)
                self.register_buffer("dirichlet_concentration", alpha_vector)
        elif p_p_name is not None:
            # Register parameters for other distribution types
            self._register_distribution_params(p_p_name, p_p_params, prefix="p_p_")

        # Store shape for profile reshaping
        self.prior_shape = prior_shape

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

        elif name == "beta":
            self.register_buffer(
                f"{prefix}concentration1", torch.tensor(params["concentration1"])
            )
            self.register_buffer(
                f"{prefix}concentration0", torch.tensor(params["concentration0"])
            )
        elif name == "laplace":
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
                concentration = self.dirichlet_concentration.to(device)
                # Get batch size from the q distribution (you'll need to pass it in)
                if hasattr(self, "current_batch_size") and self.current_batch_size > 1:
                    # Create a batch of identical priors
                    # This still constrains all samples to the same prior shape
                    # but allows proper batch-wise KL calculation
                    concentration = concentration.unsqueeze(0).expand(
                        self.current_batch_size, -1
                    )
                return torch.distributions.dirichlet.Dirichlet(concentration)

        return default_return

    def forward(self, rate, counts, q_p, q_bg, masks):
        # Get device and batch size
        device = rate.device
        batch_size = rate.shape[0]
        self.current_batch_size = batch_size

        # Ensure inputs are on the correct device
        counts = counts.to(device)
        masks = masks.to(device)

        # Create distributions on the correct device
        p_bg = self.get_prior(self.p_bg_name, "p_bg_", device)
        p_p = self.get_prior(self.p_p_name, "p_p_", device)

        # Calculate KL terms
        kl_terms = torch.zeros(batch_size, device=device)
        kl_p = torch.tensor(0.0, device=device)  # Default value

        # Only calculate profile KL if we have both distributions
        kl_p = torch.distributions.kl.kl_divergence(q_p, p_p)
        kl_terms += kl_p * self.p_p_scale

        # Calculate background and intensity KL divergence
        kl_bg = torch.distributions.kl.kl_divergence(q_bg, p_bg)
        kl_bg = kl_bg.sum(-1)
        kl_terms += kl_bg * self.p_bg_scale

        ll = torch.distributions.Poisson(rate + self.eps).log_prob(counts.unsqueeze(1))
        ll_mean = torch.mean(ll, dim=1) * masks.squeeze(-1)
        # ll_mean = (
        # (
        # torch.distributions.Poisson(rate).log_prob(counts.unsqueeze(1))
        # * masks.unsqueeze(1)
        # ).mean(1)
        # ).sum(1) / masks.sum(1)

        # Calculate negative log likelihood
        neg_ll_batch = (-ll_mean).sum(1)

        # Combine all loss terms
        batch_loss = neg_ll_batch + kl_terms

        # Final scalar loss
        total_loss = batch_loss.mean()

        # Return all components for monitoring
        return (
            total_loss,
            neg_ll_batch.mean(),
            kl_terms.mean(),
            kl_bg.mean() * self.p_bg_scale,
            kl_p.mean() * self.p_p_scale,
        )


# def create_center_spike_dirichlet_prior(
# shape=(3, 21, 21),
# outer_alpha: float = 5.0,  # large α → near‐uniform outside
# center_alpha: float = 0.05,  # small α → very spiky at exact center
# peak_radius: float = 0.1,  # fraction of max‐distance defining the “influence” zone
# decay: float = 1.0,  # power‐law exponent for the fall-off
# ):
# """
# Returns a torch vector of length prod(shape) to be used as a Dirichlet α‐vector,
# with α=center_alpha at the center voxel, rising to α=outer_alpha at distance>=peak_radius.
# """
# # 1) create a grid of normalized distances in [0,1]
# C, H, W = shape
# # center indices
# cc, ch, cw = (np.array(shape) - 1) / 2.0
# # coordinates
# zs = np.arange(C)[:, None, None]
# ys = np.arange(H)[None, :, None]
# xs = np.arange(W)[None, None, :]
# # normalized distances along each axis
# dz = (zs - cc) / (C / 2)
# dy = (ys - ch) / (H / 2)
# dx = (xs - cw) / (W / 2)
# # euclidean distance normalized to [0,1]
# dist = np.sqrt(dx**2 + dy**2 + dz**2) / np.sqrt(3)

# # 2) build α‐map: start at outer_alpha everywhere
# alpha_3d = np.full(shape, outer_alpha, dtype=np.float32)

# # 3) inside the “peak” radius, interpolate down to center_alpha
# mask = dist <= peak_radius
# scaled = dist[mask] / peak_radius  # in [0,1]
# # α(dist) = center_alpha + (outer_alpha - center_alpha) * scaled**decay
# alpha_3d[mask] = center_alpha + (outer_alpha - center_alpha) * (scaled**decay)

# # 4) return as a flat torch tensor
# return torch.tensor(alpha_3d.flatten(), dtype=torch.float32)


# if __name__ == "__main__":
# plt.imshow(
# torch.distributions.dirichlet.Dirichlet(create_center_focused_dirichlet_prior())
# .rsample()
# .reshape(3, 21, 21)[1]
# )

# plt.imshow(
# torch.distributions.Dirichlet((torch.ones(1323) * (1 / 1323))).mode.reshape(
# 3, 21, 21
# )[1]
# )

# def create_centered_dirichlet_prior(
# shape=(3,21,21),
# K: float = 50.0,       # total concentration
# sigma_frac: float = 0.25 # controls how quickly weights fall off from center
# ):
# """
# Returns a torch tensor of length prod(shape) to use as Dirichlet α-vector,
# such that the implied prior mean peaks at the center of the volume and
# the total concentration is K.
# """
# C, H, W = shape
# # --- 1) compute a Gaussian-shaped weight map w ---
# # grid of coordinates centered at zero
# zs = np.linspace(-1,1,C)[:,None,None]
# ys = np.linspace(-1,1,H)[None,:,None]
# xs = np.linspace(-1,1,W)[None,None,:]
# # squared distance from center
# r2 = zs**2 + ys**2 + xs**2
# # Gaussian profile
# w = np.exp(-r2 / (2*(sigma_frac**2)))
# # normalize so sum(w)=1
# w = w / w.sum()

# # --- 2) build α = K * w, then flatten to a vector ---
# alpha = (K * w).astype(np.float32)
# return torch.from_numpy(alpha.flatten())

# def box_cutoff_dirichlet(
# shape=(3,21,21),
# K: float = 50.0,
# half_widths=(3.0, 6.0, 6.0),  # (L_z, L_y, L_x) in voxels
# epsilon: float = 1e-4        # small floor off‐box
# ):
# d,h,w = shape
# Lz, Ly, Lx = half_widths

# # --- your same grid ---
# zc = torch.arange(d).float() - (d-1)/2
# yc = torch.arange(h).float() - (h-1)/2
# xc = torch.arange(w).float() - (w-1)/2

# Z = zc.view(d,1,1).expand(d,h,w)
# Y = yc.view(1,h,1).expand(d,h,w)
# X = xc.view(1,1,w).expand(d,h,w)

# pos = torch.stack([X,Y,Z], dim=-1).view(-1,3)  # (N,3)
# x, y, z = pos[:,0], pos[:,1], pos[:,2]

# # --- mask inside the box ---
# mask = ((z.abs() <= Lz) &
# (y.abs() <= Ly) &
# (x.abs() <= Lx)).float()  # (N,)

# # --- floor off‐box, then normalize ---
# w_tilde = mask + epsilon * (1 - mask)
# w = w_tilde / w_tilde.sum()

# alpha = K * w
# return alpha

# # %%
# d = 3
# h = 21
# w = 21


# # Create centered coordinate grid
# z_coords = torch.arange(d).float() - (d - 1) / 2
# y_coords = torch.arange(h).float() - (h - 1) / 2
# x_coords = torch.arange(w).float() - (w - 1) / 2

# z_coords = z_coords.view(d, 1, 1).expand(d, h, w)
# y_coords = y_coords.view(1, h, 1).expand(d, h, w)
# x_coords = x_coords.view(1, 1, w).expand(d, h, w)
# pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
# pixel_positions = pixel_positions.view(-1, 3)

# def shift_origin(
# pixel_positions: torch.Tensor,
# shape: tuple[int,int,int],
# new_origin_idx_zyx: tuple[float,float,float]
# ) -> torch.Tensor:
# """
# Given:
# - pixel_positions: (d*h*w, 3) tensor of (x,y,z) coords zeroed at the CENTER voxel
# - shape = (d, h, w)
# - new_origin_idx_zyx = (i_z0, i_y0, i_x0) in voxel‐index units
# Returns:
# - shifted_positions: same shape, but now zeroed at new_origin_idx_zyx
# """
# d, h, w = shape
# i_z0, i_y0, i_x0 = new_origin_idx_zyx

# # 1) compute Δ = (i_x0 - (w-1)/2,  i_y0 - (h-1)/2,  i_z0 - (d-1)/2)
# delta = torch.tensor([
# i_x0 - (w - 1) / 2,
# i_y0 - (h - 1) / 2,
# i_z0 - (d - 1) / 2,
# ], dtype=pixel_positions.dtype, device=pixel_positions.device)

# # 2) subtract it from every (x,y,z)
# shifted = pixel_positions - delta  # now (0,0,0) lives at new_origin_idx_zyx
# return shifted

# # shifted = shift_origin(pixel_positions, (d,h,w), (0.0,0.,0))
# shifted = pixel_positions - torch.ones_like(pixel_positions) * torch.tensor([0.0,0.0, 0.0])

# # %%

# plt.imshow(torch.zeros(21,21),cmap='gray_r')
# plt.grid(True, linestyle='-', linewidth=1.0,color='black')
# plt.xticks([])
# plt.yticks([])
# plt.show()


# # %%
# cov = torch.distributions.Wishart(
# df=10,
# scale_tril=torch.eye(3),
# ).sample()

# torch.distributions.LowRankMultivariateNormal(
# loc=torch.zeros(3),
# cov_factor=torch.eye(3),
# cov_diag=torch.ones(3),
# )

# log_prob = torch.distributions.MultivariateNormal(
# loc= torch.tensor([0.0, 0.0, 0.0]),
# covariance_matrix= torch.eye(3)*8,
# )
# # %%

# log_prob = torch.distributions.LowRankMultivariateNormal(
# loc=torch.zeros(3),
# cov_factor=torch.eye(3)*torch.tensor([0.2, 0.20, 0.05]),
# cov_diag=torch.ones(3)*3,
# ).log_prob(pixel_positions)

# log_probs_ = log_prob - log_prob.max(dim=-1,keepdim=True)[0]
# profile = torch.exp(log_probs_).reshape(d,h,w)


# plt.imshow(profile[1])
# # plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# # plt.title("Profile")
# plt.savefig(
# "/Users/luis/Downloads/profile_mvn7.png",
# dpi=300,
# bbox_inches="tight",
# transparent=True,
# )

# plt.show()

# # %%

# plt.imshow((shifted - torch.zeros_like(pixel_positions)).pow(2).sum(-1).sqrt().reshape(d,h,w)[1])

# plt.show()

# # %%

# torch.(1,1,0)

# plt.imshow((pixel_positions  - torch.zeros_like(pixel_positions)).pow(2).sum(-1).sqrt().reshape(d,h,w)[1])

# plt.show()


# plt.imshow(pixel_positions[:,3].reshape(3, 21, 21)[1].abs())
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()

# # %%
# alpha = create_center_spike_dirichlet_prior(
# outer_alpha=1.0,
# center_alpha=1e-7,
# decay=20.0,
# peak_radius=0.2,
# ).reshape(
# 3, 21, 21
# )

# plt.imshow(
# create_center_spike_dirichlet_prior(
# outer_alpha=1.0,
# center_alpha=1e-7,
# decay=20.0,
# peak_radius=0.2,
# ).reshape(
# 3, 21, 21
# )[1]
# )

# # %%

# def gaussian_centered_via_grid(
# shape=(3,21,21),
# K: float = 50.0,
# sigma: float = 3.0  # in voxels
# ):
# d,h,w = shape
# zc = torch.arange(d).float() - (d-1)/2
# yc = torch.arange(h).float() - (h-1)/2
# xc = torch.arange(w).float() - (w-1)/2

# Z = zc.view(d,1,1).expand(d,h,w)
# Y = yc.view(1,h,1).expand(d,h,w)
# X = xc.view(1,1,w).expand(d,h,w)

# # pixel positions
# pos = torch.stack([X,Y,Z], dim=-1).view(-1,3)  # (N,3)

# # squared distance from center
# r2 = (pos**2).sum(dim=1)  # (N,)k

# # gaussian
# w_tilde = torch.exp(-r2 / (2*sigma*sigma))  # (N,)
# w = w_tilde / w_tilde.sum()                # normalize

# alpha = K * w                              # (N,)
# return alpha  # ready for torch.distributions.Dirichlet(alpha)

# def gaussian_sharpened_dirichlet(
# shape=(3,21,21),
# K: float = 50.0,
# sigma: float = 3.0,
# p: float = 5.0     # exponent for sharpening
# ):
# # 1) same grid‐based Gaussian weights
# d,h,w = shape
# zs = torch.arange(d).float() - (d-1)/2
# ys = torch.arange(h).float() - (h-1)/2
# xs = torch.arange(w).float() - (w-1)/2
# Z = zs.view(d,1,1).expand(d,h,w)
# Y = ys.view(1,h,1).expand(d,h,w)
# X = xs.view(1,1,w).expand(d,h,w)
# pos = torch.stack([X,Y,Z], dim=-1).view(-1,3)
# r2 = (pos**2).sum(dim=1)
# w_unnorm = torch.exp(-r2/(2*sigma**2))
# # 2) sharpen
# w_sharp = w_unnorm ** p
# # 3) normalize
# w = w_sharp / w_sharp.sum()
# # 4) alphas
# return K * w


# # %%
# # alpha = create_centered_dirichlet_prior(K=1,sigma_frac=0.12)
# # alpha = box_cutoff_dirichlet(half_widths=(3.0, 3.0, 5.0), K=1.0)
# # alpha = gaussian_centered_via_grid(sigma=1.0,k=100)

# decay = 1
# peak_percentage = 0.01
# alpha = create_center_focused_dirichlet_prior(base_alpha=.01,center_alpha=20,decay_factor=decay,peak_percentage=peak_percentage)

# plt.imshow(
# torch.distributions.dirichlet.Dirichlet(alpha)
# .sample()
# .reshape(3, 21, 21)[1]
# )
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.savefig(
# f"/Users/luis/Downloads/dirichlet_alpha_{peak_percentage}_mean.png",
# dpi=300,
# bbox_inches="tight",
# transparent=True,
# )
# plt.show()

# # %%
# alph = 1.0

# dirichlet = torch.distributions.dirichlet.Dirichlet(torch.ones(21*21)*alph)
# sample = dirichlet.rsample().reshape(21, 21)

# plt.imshow(
# # sample
# dirichlet.mean.reshape(21, 21),
# )
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# # plt.title(f'alpha: {alph}\nmean: {dirichlet.mean[0] :.2e}\n    var: {dirichlet.variance[0] :.2e}')
# plt.title(f'α: {alph}\n𝔼(Dir(α)')
# #transparent background
# plt.savefig(
# f"/Users/luis/Downloads/dirichlet_alpha_{alph}_mean.png",
# dpi=300,
# bbox_inches="tight",
# transparent=True,
# # bbox_inches="tight",
# )
# plt.show()

# # %%
# torch.distributions.dirichlet.Dirichlet(torch.ones(21*21)*1).variance

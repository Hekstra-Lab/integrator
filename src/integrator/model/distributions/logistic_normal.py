import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProfilePosterior:
    """Variational posterior q(h|x) = N(mu_h, diag(sigma_h^2)).
    Profiles are recovered as prf = softmax(W @ h + b).
    """

    def __init__(
        self,
        mu_h: Tensor,
        std_h: Tensor,
        W: Tensor,
        b: Tensor,
        sigma_prior: float,
    ) -> None:
        self.mu_h = mu_h  # (B, d)
        self.std_h = std_h  # (B, d)
        self.W = W  # (K, d)
        self.b = b  # (K,)
        self.sigma_prior = sigma_prior  # scalar

        # Dirichlet compatibility: integrators store qp.concentration
        self.concentration: Tensor | None = None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Reparameterized profile sample.

        Parameters
        ----------
        sample_shape : torch.Size
            Leading sample dimensions (e.g. [S] for S MC samples).

        Returns
        -------
        Tensor, shape (*sample_shape, B, K)
            Probability vectors on the simplex.
        """
        eps = torch.randn(
            *sample_shape, *self.mu_h.shape, device=self.mu_h.device
        )  # (*sample_shape, B, d)
        h = self.mu_h + self.std_h * eps  # (*sample_shape, B, d)
        logits = h @ self.W.T + self.b  # (*sample_shape, B, K)
        return F.softmax(logits, dim=-1)

    def rsample_h(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Reparameterized sample of the latent h (not the profile)."""
        eps = torch.randn(
            *sample_shape, *self.mu_h.shape, device=self.mu_h.device
        )
        return self.mu_h + self.std_h * eps

    def h_to_profile(self, h: Tensor) -> Tensor:
        """Deterministic map from latent h to profile vector."""
        logits = h @ self.W.T + self.b
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mean(self) -> Tensor:
        """Profile evaluated at the posterior mean h.

        This is a point estimate, not the true mean of the profile
        distribution (which has no closed form under the logistic-normal).

        Returns: (B, K)
        """
        logits = self.mu_h @ self.W.T + self.b
        return F.softmax(logits, dim=-1)

    @property
    def mean_profile(self) -> Tensor:
        """Alias for .mean — profile at posterior mean h."""
        return self.mean

    @property
    def mean_h(self) -> Tensor:
        """Posterior mean of h.  Shape: (B, d)"""
        return self.mu_h

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    def kl_divergence(self, group_labels: Tensor | None = None) -> Tensor:
        """Closed-form KL(q(h) || p(h)).

        q(h) = N(mu_h, diag(sigma_h^2))
        p(h) = N(0,    sigma_p^2 * I)

        Parameters
        ----------
        group_labels : Tensor | None
            Ignored in the base class (global prior).  Subclasses like
            ``PerBinProfilePosterior`` use this to select per-bin priors.

        Returns: (B,) — KL per batch element.
        """
        sigma_p_sq = self.sigma_prior**2
        sigma_q_sq = self.std_h**2

        kl = 0.5 * (
            sigma_q_sq / sigma_p_sq
            + self.mu_h**2 / sigma_p_sq
            - 1.0
            - torch.log(sigma_q_sq / sigma_p_sq)
        ).sum(dim=-1)

        return kl  # (B,)


# %%
class LogisticNormalSurrogate(nn.Module):
    def __init__(self, input_dim: int, basis_path: str) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)
        self.W: Tensor
        self.b: Tensor
        self.register_buffer("W", basis["W"])  # (K, d)
        self.register_buffer("b", basis["b"])  # (K,)
        self.d: int = int(basis["d"])
        self.sigma_prior: float = float(basis.get("sigma_prior", 3.0))

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)

        # Initialise std_head so initial std ≈ exp(-1) ≈ 0.37
        # softplus(-0.81) ≈ 0.37
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, -0.81)

    def forward(self, x: Tensor) -> ProfilePosterior:
        """Map encoder output to a ProfilePosterior.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
            Output of the profile encoder.

        Returns
        -------
        ProfilePosterior
        """
        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        return ProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            W=self.W,
            b=self.b,
            sigma_prior=self.sigma_prior,
        )


# Learned linear decoder surrogate


class LinearProfileSurrogate(nn.Module):
    """Profile surrogate with a learned linear decoder.

        prf = softmax(W @ h + b)
        q(h | x) = N(mu_h(x), diag(sigma_h(x)²))

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output.
    latent_dim : int
        Dimension of the latent h. Default 8.
    output_dim : int
        Number of profile pixels (H * W). Default 441.
    sigma_prior : float
        Prior std for h ~ N(0, sigma_prior² I). Default 3.0.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        output_dim: int = 441,
        sigma_prior: float = 3.0,
    ) -> None:
        super().__init__()

        self.d: int = latent_dim
        self.sigma_prior: float = float(sigma_prior)

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

        # Initialise std_head so initial std ≈ exp(-1) ≈ 0.37
        # softplus(-0.81) ≈ 0.37
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, -0.81)

    def forward(self, x: Tensor) -> ProfilePosterior:
        mu_h = self.mu_head(x)  # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)
        return ProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            W=self.decoder.weight,  # (output_dim, d) — trainable
            b=self.decoder.bias,  # (output_dim,)   — trainable
            sigma_prior=self.sigma_prior,
        )


# Gaussian profile
def _h_to_physical_params(
    h: Tensor,
    center_base: float = 10.0,
    center_scale: float = 1.5,
    log_sigma_base: float = 0.7,
    width_scale: float = 0.4,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Map latent h (..., 5) → physical Gaussian parameters.

    Returns (cx, cy, sigma1, sigma2, theta), all shape (...,).
    """
    cx = center_base + h[..., 0] * center_scale
    cy = center_base + h[..., 1] * center_scale
    sigma1 = (log_sigma_base + h[..., 2] * width_scale).exp()
    sigma2 = (log_sigma_base + h[..., 3] * width_scale).exp()
    theta = torch.pi * torch.sigmoid(h[..., 4])
    return cx, cy, sigma1, sigma2, theta


def _physical_params_to_profile(
    cx: Tensor,
    cy: Tensor,
    sigma1: Tensor,
    sigma2: Tensor,
    theta: Tensor,
    H: int = 21,
    W: int = 21,
) -> Tensor:
    """Normalized 2-D rotated Gaussian profile.

    Parameters are (...,) shaped; returns (..., H*W).
    """
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=cx.device),
        torch.arange(W, dtype=torch.float32, device=cx.device),
        indexing="ij",
    )
    # expand grid for broadcasting
    for _ in range(cx.dim()):
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)

    cx = cx[..., None, None]
    cy = cy[..., None, None]
    s1 = sigma1[..., None, None]
    s2 = sigma2[..., None, None]
    th = theta[..., None, None]

    cos_t, sin_t = th.cos(), th.sin()
    dx = xx - cx
    dy = yy - cy
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    profile = torch.exp(-0.5 * (x_rot**2 / s1**2 + y_rot**2 / s2**2))
    profile = profile / profile.sum(dim=(-2, -1), keepdim=True).clamp(
        min=1e-10
    )
    return profile.reshape(*cx.shape[:-2], H * W)


def _h_to_profile_physical(
    h: Tensor,
    center_base: float = 10.0,
    center_scale: float = 1.5,
    log_sigma_base: float = 0.7,
    width_scale: float = 0.4,
    H: int = 21,
    W: int = 21,
) -> Tensor:
    """Full pipeline: h (..., 5) → normalized profile (..., H*W)."""
    cx, cy, s1, s2, th = _h_to_physical_params(
        h, center_base, center_scale, log_sigma_base, width_scale
    )
    return _physical_params_to_profile(cx, cy, s1, s2, th, H, W)


class PhysicalGaussianProfilePosterior(ProfilePosterior):
    """Profile posterior using a physical 2-D Gaussian parameterization.

    h ∈ R^5 encodes (cx, cy, log σ₁, log σ₂, θ_raw).
    The profile is a normalized rotated Gaussian on the H×W grid.

    Inherits from ProfilePosterior so loss.py's isinstance check works.
    kl_divergence() and rsample_h() are unchanged (pure h-space ops).
    rsample() and mean are overridden to use the physical mapping.
    """

    def __init__(
        self,
        mu_h: Tensor,
        std_h: Tensor,
        transform_config: dict,
        sigma_prior: float = 1.0,
    ) -> None:
        # Bypass ProfilePosterior.__init__ (which requires W, b).
        self.mu_h = mu_h
        self.std_h = std_h
        self.sigma_prior = sigma_prior
        self._transform_config = transform_config
        self.concentration: Tensor | None = None  # Dirichlet compat shim

    def rsample(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Reparameterized profile samples. Returns (*sample_shape, B, H*W)."""
        h = self.rsample_h(sample_shape)  # (*sample_shape, B, 5)
        return _h_to_profile_physical(h, **self._transform_config)

    def h_to_profile(self, h: Tensor) -> Tensor:
        """Deterministic map h → profile."""
        return _h_to_profile_physical(h, **self._transform_config)

    @property
    def mean(self) -> Tensor:
        """Profile at the posterior mean h. Returns (B, H*W)."""
        return _h_to_profile_physical(self.mu_h, **self._transform_config)

    @property
    def mean_profile(self) -> Tensor:
        return self.mean


class PhysicalGaussianProfileSurrogate(nn.Module):
    """Profile surrogate using a physical 2-D Gaussian parameterization.

    Reads scalar hyper-parameters from ``profile_basis.pt`` saved by
    simulate_shoeboxes_mvn.py (basis_type='physical_gaussian').  Unlike the
    Hermite surrogate there is no W/b matrix — the file just holds d,
    sigma_prior, and the transform scalars.

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output.
    basis_path : str
        Path to profile_basis.pt with basis_type == 'physical_gaussian'.
    """

    def __init__(self, input_dim: int, basis_path: str) -> None:
        super().__init__()

        config = torch.load(basis_path, weights_only=False)
        if config.get("basis_type") != "physical_gaussian":
            raise ValueError(
                f"Expected basis_type='physical_gaussian', got '{config.get('basis_type')}'. "
                "Use LogisticNormalSurrogate for Hermite bases."
            )

        self.d: int = int(config["d"])  # 5
        self.sigma_prior: float = float(config.get("sigma_prior", 1.0))
        self._transform_config: dict = {
            "center_base": float(config.get("center_base", 10.0)),
            "center_scale": float(config.get("center_scale", 1.5)),
            "log_sigma_base": float(config.get("log_sigma_base", 0.7)),
            "width_scale": float(config.get("width_scale", 0.4)),
        }

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)

        # Initialise std_head so initial std ≈ exp(-1) ≈ 0.37
        # softplus(-0.81) ≈ 0.37
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, -0.81)

    def forward(self, x: Tensor) -> PhysicalGaussianProfilePosterior:
        mu_h = self.mu_head(x)  # (B, 5)
        std_h = F.softplus(self.std_head(x))  # (B, 5)
        return PhysicalGaussianProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            transform_config=self._transform_config,
            sigma_prior=self.sigma_prior,
        )


# ------------------------------------------------------------------
# Per-bin latent profile prior
# ------------------------------------------------------------------


class PerBinProfilePosterior(ProfilePosterior):
    """Profile posterior with per-bin Gaussian prior in latent space.

    Instead of a single global prior N(0, σ²I), each resolution/azimuthal
    bin k has its own prior N(μ_k, diag(σ_k²)).  The per-bin parameters
    are estimated from data (PCA or Hermite projection of bg-subtracted
    profiles) and stored as buffers in the surrogate module.

    KL is still closed-form Gaussian, computed in the low-dimensional
    latent space (d ~ 8-14), giving ~15-30 nats instead of the 200-500
    nats typical of per-bin Dirichlet on the full simplex.
    """

    def __init__(
        self,
        mu_h: Tensor,
        std_h: Tensor,
        W: Tensor,
        b: Tensor,
        sigma_prior: float,
        mu_prior: Tensor,
        std_prior: Tensor,
    ) -> None:
        super().__init__(mu_h, std_h, W, b, sigma_prior)
        self.mu_prior = mu_prior    # (n_bins, d)
        self.std_prior = std_prior  # (n_bins, d)

    def kl_divergence(self, group_labels: Tensor | None = None) -> Tensor:
        """KL(q(h) || p_k(h)) with per-bin prior.

        If group_labels is provided, uses per-bin prior N(μ_k, diag(σ_k²)).
        Otherwise falls back to the global prior N(0, σ²I).

        Parameters
        ----------
        group_labels : Tensor | None
            Bin index per reflection, shape (B,).  Long tensor.

        Returns
        -------
        Tensor, shape (B,)
        """
        if group_labels is None:
            return super().kl_divergence()

        mu_p = self.mu_prior[group_labels]    # (B, d)
        std_p = self.std_prior[group_labels]  # (B, d)

        var_q = self.std_h ** 2
        var_p = std_p ** 2

        kl = 0.5 * (
            var_q / var_p
            + (self.mu_h - mu_p) ** 2 / var_p
            - 1.0
            - torch.log(var_q / var_p)
        ).sum(dim=-1)

        return kl  # (B,)


class PerBinLogisticNormalSurrogate(nn.Module):
    """Profile surrogate with fixed basis (Hermite or PCA) and per-bin priors.

    Loads a ``profile_basis_per_bin.pt`` file containing:
        - W (K, d): basis matrix (Hermite functions or PCA components)
        - b (K,): bias (log of reference profile or mean of log-profiles)
        - mu_per_group (n_bins, d): per-bin prior mean in latent space
        - std_per_group (n_bins, d): per-bin prior std in latent space
        - sigma_prior (float): global fallback prior std
        - d (int): latent dimensionality

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output.
    basis_path : str
        Path to profile_basis_per_bin.pt.
    """

    def __init__(self, input_dim: int, basis_path: str) -> None:
        super().__init__()

        basis = torch.load(basis_path, weights_only=False)

        self.register_buffer("W", basis["W"])                    # (K, d)
        self.register_buffer("b", basis["b"])                    # (K,)
        self.register_buffer("mu_per_group", basis["mu_per_group"])    # (n_bins, d)
        self.register_buffer("std_per_group", basis["std_per_group"])  # (n_bins, d)

        self.d: int = int(basis["d"])
        self.sigma_prior: float = float(basis.get("sigma_prior", 3.0))

        self.mu_head = nn.Linear(input_dim, self.d)
        self.std_head = nn.Linear(input_dim, self.d)

        # Initialise std_head so initial std ≈ exp(-1) ≈ 0.37
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, -0.81)

    def forward(
        self, x: Tensor, group_labels: Tensor | None = None,
    ) -> PerBinProfilePosterior:
        mu_h = self.mu_head(x)              # (B, d)
        std_h = F.softplus(self.std_head(x))  # (B, d)

        return PerBinProfilePosterior(
            mu_h=mu_h,
            std_h=std_h,
            W=self.W,
            b=self.b,
            sigma_prior=self.sigma_prior,
            mu_prior=self.mu_per_group,
            std_prior=self.std_per_group,
        )

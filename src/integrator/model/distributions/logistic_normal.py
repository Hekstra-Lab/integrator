import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProfilePosterior:
    """Variational posterior q(h|x) = N(mu_h, diag(sigma_h^2)).
    Profiles are recovered as prf = softmax(W @ h + b)
    """

    def __init__(
        self,
        mu_h: Tensor,
        logvar_h: Tensor,
        W: Tensor,
        b: Tensor,
        sigma_prior: float,
    ) -> None:
        self.mu_h = mu_h  # (B, d)
        self.logvar_h = logvar_h  # (B, d)
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
        std_h = torch.exp(0.5 * self.logvar_h)  # (B, d)
        eps = torch.randn(
            *sample_shape, *self.mu_h.shape, device=self.mu_h.device
        )  # (*sample_shape, B, d)
        h = self.mu_h + std_h * eps  # (*sample_shape, B, d)
        logits = h @ self.W.T + self.b  # (*sample_shape, B, K)
        return F.softmax(logits, dim=-1)

    def rsample_h(self, sample_shape: torch.Size = torch.Size([])) -> Tensor:
        """Reparameterized sample of the latent h (not the profile)."""
        std_h = torch.exp(0.5 * self.logvar_h)
        eps = torch.randn(
            *sample_shape, *self.mu_h.shape, device=self.mu_h.device
        )
        return self.mu_h + std_h * eps

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

    def kl_divergence(self) -> Tensor:
        """Closed-form KL(q(h) || p(h)).

        q(h) = N(mu_h, diag(sigma_h^2))
        p(h) = N(0,    sigma_p^2 * I)

        Returns: (B,) — KL per batch element.
        """
        sigma_p_sq = self.sigma_prior**2
        sigma_q_sq = self.logvar_h.exp()

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
        self.logvar_head = nn.Linear(input_dim, self.d)

        # Initialise logvar_head so initial sigma^2 ≈ exp(-2) ≈ 0.14
        # (variance smaller than the prior sigma_p^2 = 9 → informative start)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

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
        logvar_h = self.logvar_head(x).clamp(-10.0, 10.0)  # (B, d)

        return ProfilePosterior(
            mu_h=mu_h,
            logvar_h=logvar_h,
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
        self.logvar_head = nn.Linear(input_dim, self.d)
        self.decoder = nn.Linear(self.d, output_dim)

        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x: Tensor) -> ProfilePosterior:
        mu_h = self.mu_head(x)  # (B, d)
        logvar_h = self.logvar_head(x).clamp(-10.0, 10.0)  # (B, d)
        return ProfilePosterior(
            mu_h=mu_h,
            logvar_h=logvar_h,
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
        logvar_h: Tensor,
        transform_config: dict,
        sigma_prior: float = 1.0,
    ) -> None:
        # Bypass ProfilePosterior.__init__ (which requires W, b).
        self.mu_h = mu_h
        self.logvar_h = logvar_h
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
        self.logvar_head = nn.Linear(input_dim, self.d)

        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def forward(self, x: Tensor) -> PhysicalGaussianProfilePosterior:
        mu_h = self.mu_head(x)  # (B, 5)
        logvar_h = self.logvar_head(x).clamp(-10.0, 10.0)  # (B, 5)
        return PhysicalGaussianProfilePosterior(
            mu_h=mu_h,
            logvar_h=logvar_h,
            transform_config=self._transform_config,
            sigma_prior=self.sigma_prior,
        )

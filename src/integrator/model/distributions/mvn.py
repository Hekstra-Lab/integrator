import torch
from torch.distributions import (
    ComposeTransform,
    MultivariateNormal,
    Transform,
    constraints,
)
from torch.distributions.transforms import AffineTransform, SoftplusTransform
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix

from integrator.layers import Linear


class FillTriL(Transform):
    """
    Transform for converting a real-valued vector into a lower triangular matrix
    """

    def __init__(self):
        super().__init__()

    @property
    def domain(self):
        return constraints.real_vector

    @property
    def codomain(self):
        return constraints.lower_triangular

    @property
    def bijective(self):
        return True

    def _call(self, x):
        """
        Converts real-valued vector to lower triangular matrix.

        Args:
            x (torch.Tensor): input real-valued vector
        Returns:
            torch.Tensor: Lower triangular matrix
        """

        return vec_to_tril_matrix(x)

    def _inverse(self, y):
        return tril_matrix_to_vec(y)

    def log_abs_det_jacobian(self, x, y):
        batch_shape = x.shape[:-1]
        return torch.zeros(batch_shape, dtype=x.dtype, device=x.device)


class DiagTransform(Transform):
    """
    Applies transformation to the diagonal of a square matrix
    """

    def __init__(self, diag_transform):
        super().__init__()
        self.diag_transform = diag_transform

    @property
    def domain(self):
        return self.diag_transform.domain

    @property
    def codomain(self):
        return self.diag_transform.codomain

    @property
    def bijective(self):
        return self.diag_transform.bijective

    def _call(self, x):
        """
        Args:
            x (torch.Tensor): Input matrix
        Returns
            torch.Tensor: Transformed matrix
        """
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        transformed_diagonal = self.diag_transform(diagonal)
        result = x.diagonal_scatter(transformed_diagonal, dim1=-2, dim2=-1)

        return result

    def _inverse(self, y):
        diagonal = y.diagonal(dim1=-2, dim2=-1)
        result = y.diagonal_scatter(self.diag_transform.inv(diagonal), dim1=-2, dim2=-1)
        return result

    def log_abs_det_jacobian(self, x, y):
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        return self.diag_transform.log_abs_det_jacobian(diagonal, y)


class FillScaleTriL(ComposeTransform):
    """
    A `ComposeTransform` that reshapes a real-valued vector into a lower triangular matrix.
    The diagonal of the matrix is transformed with `diag_transform`.
    """

    def __init__(self, diag_transform=None):
        if diag_transform is None:
            diag_transform = torch.distributions.ComposeTransform(
                ([SoftplusTransform(), AffineTransform(1e-5, 1.0)],)
            )
        super().__init__([FillTriL(), DiagTransform(diag_transform=diag_transform)])
        self.diag_transform = diag_transform

    @property
    def bijective(self):
        return True

    def log_abs_det_jacobian(self, x, y):
        x = FillTriL()._call(x)
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        return self.diag_transform.log_abs_det_jacobian(diagonal, diagonal)

    @staticmethod
    def params_size(event_size):
        """
        Returns the number of parameters required to create an n-by-n lower triangular matrix, which is given by n*(n+1)//2

        Args:
            event_size (int): size of event
        Returns:
            int: Number of parameters needed

        """
        return event_size * (event_size + 1) // 2


class MVNDistribution(torch.nn.Module):
    def __init__(self, dmodel, image_shape):
        super().__init__()
        self.dmodel = dmodel
        self.eps = 1e-6

        # Use different transformation for more flexible scale learning
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())

        # Create scale and mean prediction layers
        self.scale_layer = Linear(self.dmodel, 6)

        # Initialize scale_layer to output an isotropic Gaussian by default
        with torch.no_grad():
            # Invert ELU+1: bias = value - 1 for value > 0 (inverse of elu(x) + 1)
            init_scale = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
            torch.nn.init.zeros_(
                self.scale_layer.weight
            )  # prevents representation from influencing scale at init

        self.mean_layer = Linear(self.dmodel, 3)

        # Calculate image dimensions
        d, h, w = image_shape
        self.image_shape = image_shape

        # Create centered coordinate grid
        z_coords = torch.arange(d).float() - (d - 1) / 2
        y_coords = torch.arange(h).float() - (h - 1) / 2
        x_coords = torch.arange(w).float() - (w - 1) / 2

        z_coords = z_coords.view(d, 1, 1).expand(d, h, w)
        y_coords = y_coords.view(1, h, 1).expand(d, h, w)
        x_coords = x_coords.view(1, 1, w).expand(d, h, w)

        # Stack coordinates
        pixel_positions = torch.stack([x_coords, y_coords, z_coords], dim=-1)
        pixel_positions = pixel_positions.view(-1, 3)

        # Register buffer
        self.register_buffer("pixel_positions", pixel_positions)

        # Create a default scale parameter for initialization guidance
        self.register_buffer(
            "scale_init", torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]).view(1, 1, 6)
        )

    def forward(self, representation):
        batch_size = representation.size(0)

        # Predict mean offsets
        means = self.mean_layer(representation).view(batch_size, 1, 3)

        # Predict scale parameter}
        scales_raw = self.scale_layer(representation).view(batch_size, 1, 6)
        scales = torch.nn.functional.softplus(scales_raw) + self.eps

        # Transform scales
        L = self.L_transform(scales).to(torch.float32)

        # Create MVN distribution
        mvn = MultivariateNormal(means, scale_tril=L)

        # Compute log probabilities
        pixel_positions = self.pixel_positions.unsqueeze(0).expand(batch_size, -1, -1)
        log_probs = mvn.log_prob(pixel_positions)

        # Convert to probabilities
        # Subtract max for numerical stability (prevents overflow)
        log_probs_stable = log_probs - log_probs.max(dim=1, keepdim=True)[0]
        profile = torch.exp(log_probs_stable)

        # Normalize to sum to 1
        profile = profile / (profile.sum(dim=1, keepdim=True) + 1e-10)

        return profile


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    profile_model = MVNProfile(dmodel=64, image_shape=(3, 21, 21))

    # Create a batch of representations (assuming 10 sample with 64-dimensional representation)
    representation = torch.randn(10, 64)

    profile = profile_model(representation)

    # The output should have shape [1, 3*21*21] = [1, 1323]
    # -
    expanded = profile.unsqueeze_(1).expand(-1, 100, -1)

    background = torch.distributions.HalfNormal(1.0)
    intensity = torch.distributions.Normal(200, 5)
    bg_sample = background.sample([1323]).reshape(3, 21, 21)[1]
    i_sample = intensity.sample()

    # profile
    plt.imshow(profile[0].detach().reshape(3, 21, 21)[1])
    plt.show()

    # background
    plt.imshow(bg_sample)
    plt.show()

    # shoebox
    observation = (i_sample * profile[0].detach().reshape(3, 21, 21))[1] + bg_sample

    plt.imshow(observation)
    plt.show()
    max_I_prf = ((i_sample * profile[0].detach().reshape(3, 21, 21))[1]).max()
    min_I_prf = ((i_sample * profile[0].detach().reshape(3, 21, 21))[1]).min()

    vmax = torch.maximum(max_I_prf, bg_sample.max())
    vmin = torch.minimum(min_I_prf, bg_sample.min())

    figs, axes = plt.subplots(1, 3)

    axes[0].imshow(bg_sample, cmap="gray_r", vmax=vmax, vmin=vmin)
    axes[0].tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    axes[0].set_title("bg")
    axes[1].imshow(profile[0].detach().reshape(3, 21, 21)[1], cmap="gray_r")
    axes[1].tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    axes[1].set_title("I*prf")
    axes[2].imshow(observation, cmap="gray_r", vmax=vmax, vmin=vmin)
    axes[2].tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    axes[2].set_title("I*prf + bg")

    plt.savefig("/Users/luis/Downloads/data_generating_process.pdf")
    plt.show()

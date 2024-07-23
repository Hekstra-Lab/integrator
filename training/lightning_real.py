import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader
from integrator.layers import Standardize
import torch
import polars as pl
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP
from rs_distributions import distributions as rsd
from torch.utils.data import DataLoader
from rs_distributions.transforms import FillScaleTriL
from torch.distributions.transforms import ExpTransform
from tqdm import tqdm
from integrator.models.encoder import MeanPool
from torch.distributions.transforms import ExpTransform
from torch.distributions.transforms import (
    Transform,
    ComposeTransform,
    SoftplusTransform,
)
from torch.distributions.utils import vec_to_tril_matrix, tril_matrix_to_vec


# %%
class Standardize(torch.nn.Module):
    def __init__(
        self, center=True, feature_dim=7, max_counts=float("inf"), epsilon=1e-6
    ):
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("pixel_count", torch.tensor(0.0))  # Counter for pixels
        self.register_buffer("image_count", torch.tensor(0.0))  # Counter for images

        # Mask to exclude certain features from mean subtraction (0 for exclusion, 1 for inclusion)
        self.mean_mask = torch.ones((1, 1, feature_dim))
        self.mean_mask[
            ..., 3:6
        ] = 0  # Exclude 4th, 5th, and 6th features (0-based index)

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / self.pixel_count.clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, im, mask=None):
        if mask is None:
            k = len(
                im
            )  # Assuming 'k' should be number of elements in 'im' if no mask is provided
        else:
            k = mask.sum()  # count num of pixels in batch
        self.pixel_count += k
        self.image_count += len(im)

        if mask is None:
            diff = im - self.mean
        else:
            diff = (im - self.mean) * mask.unsqueeze(-1)

        new_mean = self.mean + torch.sum(diff, dim=(0, 1)) / self.pixel_count
        if mask is None:
            self.m2 += torch.sum((im - new_mean) * diff, dim=(0, 1))
        else:
            self.m2 += torch.sum(
                (im - new_mean) * mask.unsqueeze(-1) * diff, dim=(0, 1)
            )
        self.mean = new_mean

    def standardize(self, im, mask=None):
        if self.center:
            if mask is None:
                return (im - self.mean * self.mean_mask) / self.std
            else:
                return (
                    (im - self.mean * self.mean_mask) * mask.unsqueeze(-1)
                ) / self.std
        return im / self.std

    def forward(self, im, mask=None, training=True):
        if self.image_count >= self.max_counts:
            training = False

        if training:
            self.update(im, mask)

        return self.standardize(im, mask)


# %%
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

    def __init__(self, diag_transform=SoftplusTransform()):
        super().__init__([FillTriL(), DiagTransform(diag_transform=diag_transform)])
        self.diag_transform = diag_transform

    @property
    def bijective(self):
        return True

    def log_abs_det_jacobian(self, x, y):
        x = FillTriL()._call(x)
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        return self.diag_transform.log_abs_det_jacobian(diagonal, diagonal)


# %%
class MLP(torch.nn.Module):
    """
    If d_in \neq width, you must specify it .
    """

    # If d_in \neq width, you must specify it
    def __init__(self, width, depth, dropout=None, d_in=None, output_dims=None):
        """
        Multi-layer perceptron (MLP) module

        Args:
            width (int): Width of the hidden layers
            depth (int): Number of residual layers
            dropout (float, optional): Dropout probability. Defaults to None.
            d_in (int, optional): Input dimension. If not equal to width, it must be specified. Defaults to None.
            output_dims (int, optional): Output dimension. If specified, an additional linear layer is added at the end. Defaults to None.
        """
        super().__init__()
        layers = []
        if d_in is not None:
            layers.append(Linear(d_in, width))
        layers.extend([ResidualLayer(width, dropout=dropout) for i in range(depth)])
        if output_dims is not None:
            layers.append(Linear(width, output_dims))
        self.main = torch.nn.Sequential(*layers)

    def forward(self, data, **kwargs):
        out = self.main(data)
        return out


# %%
class DistributionBuilder(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        intensity_dist,
        background_dist,
        eps=1e-12,
        beta=1.0,
        output_dim=10,
        batch_size=10,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.output_dim = output_dim
        self.linear1 = Linear(dmodel, 4)
        self.linear_L = Linear(dmodel, 6)
        self.intensity_dist = intensity_dist
        self.background_dist = background_dist
        self.L_transform = FillScaleTriL(diag_transform=ExpTransform())
        self.batch_size = batch_size

    def constraint(self, x):
        return torch.nn.functional.softplus(x, beta=self.beta) + self.eps

    def intensity_distribution(self, params):
        loc = params[..., 0]
        scale = params[..., 1]
        scale = self.constraint(scale)
        q_I = self.intensity_dist(loc, scale)
        return q_I

    def init_weights(self):
        L_weights = self.linear_L.weight.data
        L_bias = self.linear_L.bias.data

        scale = 0.01
        torch.nn.init.uniform_(L_weights, -scale, scale)
        torch.nn.init.uniform_(L_bias, 1 - scale, 1 + scale)

        L_weights.zero_()
        L_bias.zero_()
        L_bias[0] = 1.0
        L_bias[2] = 1.0
        L_bias[5] = 1.0

    def background(self, params):
        mu = params[..., 2]
        # mu = self.constraint(mu)
        sigma = params[..., 3]
        sigma = self.constraint(sigma)
        q_bg = self.background_dist(mu, sigma)
        return q_bg

    def MVNProfile3D(self, L_params, dxyz, mask):
        factory_kwargs = {
            "device": dxyz.device,
            "dtype": dxyz.dtype,
        }
        batch_size = L_params.size(0)
        mu = torch.zeros((batch_size, 1, 3), requires_grad=False, **factory_kwargs)
        # mu = params[...,4:7]
        # L = params[..., 7:]
        # L = self.Linear_L([..., 4:]
        L = self.L_transform(L_params)
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, scale_tril=L
        )
        log_probs = mvn.log_prob(dxyz)
        profile = torch.exp(log_probs)

        return profile, L

    def forward(self, representation, dxy, mask):
        params = self.linear1(representation)
        L_params = self.linear_L(representation)
        profile, L = self.MVNProfile3D(L_params, dxy, mask)
        # profile = self.GaussianProfile(params, dxy, mask)

        # variational background distribution
        q_bg = self.background(params)

        # variational intensity distribution
        q_I = self.intensity_distribution(params)

        return q_bg, q_I, profile, L
        # return q_I, profile


# %%
class Encoder(torch.nn.Module):
    def __init__(self, depth, dmodel, feature_dim, dropout=None):
        super().__init__()
        self.dropout = None
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=self.dropout, output_dims=dmodel
        )
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask=None):
        out = self.mlp_1(shoebox_data)
        pooled_out = self.mean_pool(out, mask)
        return pooled_out


# Embed shoeboxes
class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.register_buffer(
            "dim",
            torch.tensor(dim),
        )

    def forward(self, data, mask=None):
        data = data * mask
        out = torch.sum(data, dim=1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = torch.sum(mask, dim=-2, keepdim=True)
        out = out / denom

        return out


# %%
class PoissonLikelihoodV2(torch.nn.Module):
    """
    Attributes:
        beta:
        p_I_scale: scale DKL(q_I||p_I)
        p_bg_scale: scale DKL(q_I||p_I)
        prior_I: prior distribution for intensity
        prior_bg: prior distribution for background
    """

    def __init__(
        self,
        beta=1.0,
        eps=1e-8,
        prior_I=None,
        prior_bg=None,
        prior_profile=None,
        p_I_scale=0.01,  # influence of DKL(LogNorm||LogNorm) term
        p_bg_scale=0.01,
        p_profile_scale=0.01,
    ):
        super().__init__()
        self.eps = torch.nn.Parameter(data=torch.tensor(eps), requires_grad=False)
        self.beta = torch.nn.Parameter(data=torch.tensor(beta), requires_grad=False)
        self.p_I_scale = torch.nn.Parameter(
            data=torch.tensor(p_I_scale), requires_grad=False
        )
        self.p_bg_scale = torch.nn.Parameter(
            data=torch.tensor(p_bg_scale), requires_grad=False
        )
        self.prior_profile_scale = torch.nn.Parameter(
            data=torch.tensor(p_profile_scale), requires_grad=False
        )
        self.prior_I = prior_I
        self.prior_bg = prior_bg
        self.prior_profile = prior_profile

    def forward(
        self,
        counts,
        q_bg,
        q_I,
        profile,
        L,
        eps=1e-8,
        mc_samples=10,
        mask=None,
    ):
        """
        Args:
            counts: observed photon counts
            q_bg: variational background distribution
            q_I: variational intensity distribution
            profile: MVN profile model
            mc_samples: number of monte carlo samples
            vi: use KL-term
            mask: mask for padded entries

        Returns: log-likelihood and KL(q|p)
        """

        # Sample from variational distributions
        z = q_I.rsample([mc_samples])
        bg = q_bg.rsample([mc_samples])

        # Set KL term
        kl_term = 0

        # Calculate the rate
        # rate = z.permute(1,0,2) * (profile) + bg.permute(1,0,2)
        rate = z.permute(1, 0, 2) * (profile.unsqueeze(1)) + bg.permute(1, 0, 2)

        # ll = torch.distributions.Poisson(rate).log_prob(counts)
        ll = torch.distributions.Poisson(rate).log_prob(counts.unsqueeze(1))

        # ll = ll * mask if mask is not None else ll

        # Calculate KL-divergence only if the corresponding priors and distributions are available
        if q_I is not None and self.prior_I is not None:
            kl_I = q_I.log_prob(z) - self.prior_I.log_prob(z)
            kl_term += kl_I.mean() * self.p_I_scale

        if q_bg is not None and self.prior_bg is not None:
            kl_bg = q_bg.log_prob(bg) - self.prior_bg.log_prob(bg)
            kl_bg = kl_bg * mask if mask is not None else kl_bg
            kl_term += kl_bg.mean() * self.p_bg_scale

        if self.prior_profile is not None:
            profile_dist = torch.distributions.MultivariateNormal(
                torch.zeros_like(L), scale_tril=L
            )
            kl_profile = torch.distributions.kl_divergence(
                profile_dist, self.prior_profile
            )
            kl_term += kl_profile.mean() * self.prior_profile_scale

        return ll, kl_term, rate


# %%
# Integrator
class Integrator(torch.nn.Module):
    """
    Integration module

    Attributes:
        encoder (torch.nn.Module): Encodes shoeboxes.
        distribution_builder (torch.nn.Module): Builds variational distributions and profile.
        likelihood (torch.nn.Module): MLE cost function.
        counts_std (torch.nn.Parameter): Standard deviation of counts. Not trainable.
    """

    def __init__(
        self,
        standardize,
        encoder,
        distribution_builder,
        likelihood,
    ):
        super().__init__()
        self.standardize = standardize
        self.encoder = encoder
        self.distribution_builder = distribution_builder
        self.likelihood = likelihood
        self.counts_std = None

    def check_gradients(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient norm: {param.grad.norm().item()}")
            else:
                print(f"{name} has no gradients")

    def forward(
        self,
        shoebox,
        dead_pixel_mask,
        mc_samples=100,
    ):
        """
        Forward pass of the integrator.

        Args:
            shoebox (torch.Tensor): Shoebox tensor
            padding_mask (torch.Tensor): Mask of padded entries
            dead_pixel_mask (torch.Tensor): Mask of dead pixels and padded entries
            mc_samples (int): Number of Monte Carlo samples. Defaults to 100.

        Returns:
            torch.Tensor: Negative log-likelihood loss
        """

        # get counts
        counts = torch.clamp(shoebox[..., -1], min=0)
        # dxyz = shoebox[..., 3:6]

        # standardize data
        shoebox_ = self.standardize(shoebox, dead_pixel_mask.squeeze(-1))
        dxyz = shoebox_[..., 3:6]

        # distances to centroid

        # encode shoebox
        representation = self.encoder(shoebox_, dead_pixel_mask)

        # build q_I, q_bg, and profile
        q_bg, q_I, profile, L = self.distribution_builder(
            representation, dxyz, dead_pixel_mask
        )
        # q_I, profile = self.distribution_builder(representation, dxyz, dead_pixel_mask)

        # calculate ll and kl
        ll, kl_term, rate_ = self.likelihood(
            counts,
            q_bg,
            q_I,
            profile,
            L,
            mc_samples=mc_samples,
            mask=dead_pixel_mask.squeeze(-1),
        )

        ll_mean = torch.mean(ll, dim=1, keepdims=True) * dead_pixel_mask.squeeze(
            -1
        )  # mean across mc_samples
        nll = -(torch.sum(ll_mean) / torch.sum(dead_pixel_mask))

        if self.training == True:
            # return nll + kl_term
            return (nll + kl_term, rate_, q_I, profile, q_bg, counts, L)

        else:
            return (nll + kl_term, rate_, q_I, profile, q_bg, counts, L)

    def grad_norm(self):
        """
        Calculate the gradient norm of the model parameters.

        Returns:
            torch.Tensor: Gradient norm.
        """
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm


# %%
# Hyperparameters
depth = 10
dmodel = 64
feature_dim = 7
dropout = 0.5
beta = 1.0
mc_samples = 100
max_size = 1024
eps = 1e-12
batch_size = 2
learning_rate = 0.001
epochs = 10

# Load training data
loaded_data = torch.load("shoebox_data.pt")
loaded_data

# store training data into tensors
dataset = loaded_data["dataset"]
dataset = dataset.to(torch.float32)
true_intensities = loaded_data["true_intensities"]
true_poisson_rate = loaded_data["poisson_rate"]
true_rate = loaded_data[Rij]
true_covariances = loaded_data["Cov"]
weighted_true_intensity = loaded_data["weighted_intensies"]
masks_ = torch.ones(dataset.shape[0], dataset.shape[1], 1)


class SimulatedData(torch.utils.data.Dataset):
    def __init__(self, data, masks, true_intensities, true_Ls):
        self.data = data
        self.masks = masks
        self.true_intensities = true_intensities
        self.true_Ls = true_Ls

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        shoebox = self.data[idx]
        mask = self.masks[idx]
        true_I = self.true_intensities[idx]
        true_L = self.true_Ls[idx]
        return shoebox, mask, true_I, true_L


simulated_data = SimulatedData(dataset, masks_, true_intensities, true_Ls)


# train loader
subset_ratio = 0.004
subset_size = int(len(simulated_data) * subset_ratio)
subset_indices = list(range(subset_size))
subset_data = torch.utils.data.Subset(simulated_data, subset_indices)
train_subset, test_subset = torch.utils.data.random_split(subset_data, [0.8, 0.2])
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)


# Variational distributions
intensity_dist = rsd.FoldedNormal
background_dist = rsd.FoldedNormal

prior_I = torch.distributions.log_normal.LogNormal(
    loc=torch.tensor(7.0, requires_grad=False),
    scale=torch.tensor(1.5, requires_grad=False),
)

# prior_I = torch.distributions.exponential.Exponential(
# rate=torch.tensor(10, requires_grad=False)
# )

p_I_scale = 1
# prior_bg = torch.distributions.normal.Normal(
# loc=torch.tensor(0, requires_grad=False),
# scale=torch.tensor(1, requires_grad=False),
# )
prior_bg = rsd.FoldedNormal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
# prior_bg = torch.distributions.gamma.Gamma(torch.tensor([1.0]), torch.tensor([0.5]))
p_bg_scale = 1

prior_profile = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(3), torch.eye(3)
)


true_intensities[0]
true_intensities[1]
true_intensities[2]

epochs = 200
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_steps = len(train_loader)

standardization = Standardize(max_counts=len(train_loader))
encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)

poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    prior_profile=prior_profile,
    p_I_scale=0.006,
    p_bg_scale=0.006,
    p_profile_scale=0.006,
)

integrator = Integrator(
    standardize=standardization,
    encoder=encoder,
    distribution_builder=distribution_builder,
    likelihood=poisson_loss,
)
integrator = integrator.to(device)
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)

grad_norms = []

q_I_list = []
q_bg_list = []

train_profile = []
train_avg_loss = []
train_traces = []
train_rate = []
train_L = []

test_avg_loss = []
test_profile = []
test_rate = []
test_traces = []
test_L = []

# evaluate every number of epochs
evaluate_every = 1
num_runs = 50

# for i in range(num_runs):
standardization = Standardize(max_counts=len(train_loader))
encoder = Encoder(depth, dmodel, feature_dim, dropout=None)
distribution_builder = DistributionBuilder(
    dmodel, intensity_dist, background_dist, eps, beta
)

poisson_loss = PoissonLikelihoodV2(
    beta=beta,
    eps=eps,
    prior_I=prior_I,
    prior_bg=prior_bg,
    prior_profile=prior_profile,
    p_I_scale=0.005,
    p_bg_scale=0.005,
    p_profile_scale=0.005,
)

integrator = Integrator(
    standardize=standardization,
    encoder=encoder,
    distribution_builder=distribution_builder,
    likelihood=poisson_loss,
)
integrator = integrator.to(device)
opt = torch.optim.Adam(integrator.parameters(), lr=learning_rate)
trace = []

q_I_mean_train_list = []
q_I_stddev_train_list = []
q_bg_mean_train_list = []
q_bg_stddev_train_list = []
true_L_train_list = []


train_preds = {
    "q_I_mean": [],
    "q_I_stddev": [],
    "q_bg_mean": [],
    "q_bg_stddev": [],
    "L_pred": [],
    "rate_pred": [],
    "profile_pred": [],
    "true_I": [],
    "true_L": [],
}

test_preds = {
    "q_I_mean_list": [],
    "q_I_stddev_list": [],
    "q_bg_mean_list": [],
    "q_bg_stddev_list": [],
    "L_pred_list": [],
    "rate_pred_list": [],
    "profile_pred_list": [],
    "true_I": [],
    "true_L": [],
}

with tqdm(total=epochs * num_steps, desc="training") as pbar:
    for epoch in range(epochs):
        # Train
        integrator.train()
        for step, (sbox, mask, true_I, true_L) in enumerate(train_loader):
            sbox = sbox.to(device)
            mask = mask.to(device)

            opt.zero_grad()
            loss, rate, q_I, profile, q_bg, counts, L = integrator(sbox, mask)
            loss.backward()
            opt.step()
            trace.append(loss.item())

            grad_norm = torch.nn.utils.clip_grad_norm_(
                integrator.parameters(), max_norm=max_size
            )

            grad_norms.append(grad_norm)

            # Update progress bar
            if epoch == epochs - 1:
                train_preds["q_I_mean"].extend(q_I.mean.ravel().tolist())
                train_preds["q_I_stddev"].extend(q_I.stddev.ravel().tolist())
                train_preds["q_bg_mean"].extend(q_bg.mean.ravel().tolist())
                train_preds["q_bg_stddev"].extend(q_bg.stddev.ravel().tolist())
                train_preds["L_pred"].extend(L.cpu())
                train_preds["profile_pred"].extend(profile.cpu())
                train_preds["rate_pred"].extend(rate.cpu())
                train_preds["true_I"].extend(true_I.ravel().tolist())
                train_preds["true_L"].extend(true_L.cpu())

            pbar.set_postfix(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "loss": loss.item(),
                    "grad norm": grad_norm,
                }
            )

            pbar.update(1)

        # store metrics/outputs
        train_avg_loss.append(torch.mean(torch.tensor(trace)))
        train_rate.append(rate)
        train_profile.append(profile)
        train_L.append(L[0])

        # Evaluate
        if (epoch + 1) % evaluate_every == 0 or epoch == epochs - 1:
            integrator.eval()
            test_loss = []

            with torch.no_grad():
                for i, (shoebox, mask, true_I, true_L) in enumerate(test_loader):
                    shoebox = shoebox.to(device)
                    mask = mask.to(device)

                    # Forward pass
                    eval_loss, rate, q_I, profile, q_bg, counts, L = integrator(
                        shoebox, mask
                    )
                    test_loss.append(eval_loss.item())

                    if epoch == epochs - 1:
                        test_preds["q_I_mean_list"].extend(q_I.mean.ravel().tolist())
                        test_preds["q_I_stddev_list"].extend(
                            q_I.stddev.ravel().tolist()
                        )
                        test_preds["q_bg_mean_list"].extend(q_bg.mean.ravel().tolist())
                        test_preds["q_bg_stddev_list"].extend(
                            q_bg.stddev.ravel().tolist()
                        )
                        test_preds["L_pred_list"].extend(L)
                        test_preds["rate_pred_list"].extend(rate)
                        test_preds["true_I"].extend(true_I.ravel().tolist())
                        test_preds["true_L"].extend(true_L)
                        test_preds["profile_pred_list"].extend(profile)

                test_avg_loss.append(torch.mean(torch.tensor(test_loss)))

train_preds

results = {
    "train_preds": train_preds,
    "test_preds": test_preds,
    "train_avg_loss": train_avg_loss,
    "test_avg_loss": test_avg_loss,
}

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
# Optional: Load results for verification
with open("results.pkl", "rb") as f:
    loaded_results = pickle.load(f)
    # print(loaded_results)
train_result_df = pl.DataFrame(loaded_results["train_preds"])
test_result_df = pl.DataFrame(loaded_results["test_preds"])

x_axis = np.linspace(0, train_result_df.height, train_result_df.height)
true_I_train = train_result_df["true_I"]
true_I_test = test_result_df["true_I"]
q_I_mean_train = train_result_df["q_I_mean"]
L_train = train_result_df["L_pred"]
q_I_mean_test = test_result_df["q_I_mean_list"]
L_test = train_result_df["L_pred"]
L_test_true = train_result_df["true_L"]

# %%
# Plot Ground truth Intensity vs predicted Intensity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.scatter(q_I_mean_train, true_I_train, alpha=0.2, color="black")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_title("Ground truth vs Model prediction (Training)")
ax1.set_ylabel("Ground truth I")
ax1.set_xlabel("Predicted I")
ax1.set_ylim(0, 1e4)
ax1.set_xlim(0, 1e4)
# ax1.set_aspect('equal')
ax1.grid()

ax2.scatter(q_I_mean_test, true_I_test, alpha=0.2, color="black")
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_title("Ground truth vs Model prediction (Testing)")
ax2.set_ylabel("Ground truth I")
ax2.set_xlim(0, 1e4)
ax2.set_ylim(0, 1e4)
# ax2.set_aspect('equal')
ax2.grid()
ax2.set_xlabel("Predicted I")

plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

x_axis = np.linspace(
    0, len(loaded_results["train_avg_loss"]), len(loaded_results["train_avg_loss"])
)
loaded_results["test_avg_loss"]

train_avg_loss = torch.tensor(loaded_results["train_avg_loss"])
test_avg_loss = torch.tensor(loaded_results["test_avg_loss"])


plt.title("Normalized test/train loss")
plt.plot(x_axis, train_avg_loss / train_avg_loss.max(), label="train")
plt.plot(x_axis, test_avg_loss / test_avg_loss.max(), label="test")
plt.xlabel("epoch")
plt.ylabel("avg loss")
plt.legend()
plt.show()

# %%
# Plot train trace

# x_axis = np.linspace(0, len(trace), len(trace))
# plt.plot(x_axis, torch.tensor(trace) / torch.max(torch.tensor(trace)), label="trace")
# plt.title("Training loss over iteration")
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.show()


# Plot average train/test loss
x_axis = np.linspace(0, len(train_avg_loss), len(train_avg_loss))
plt.plot(x_axis, torch.tensor(train_avg_loss), label="train", alpha=0.8)

plt.plot(x_axis, test_avg_loss, label="test", alpha=0.8)
plt.title("Average train/test loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

# %%
# Generate samples from test/train variational distributions

# Getting variational distribution mean and stddev vectors
q_I_mean_stddev_train = torch.tensor(
    train_result_df[["q_I_mean", "q_I_stddev"]].to_numpy()
)
q_bg_mean_stddev_train = torch.tensor(
    train_result_df[["q_bg_mean", "q_bg_stddev"]].to_numpy()
)

q_I_mean_stddev_test = torch.tensor(
    test_result_df[["q_I_mean_list", "q_I_stddev_list"]].to_numpy()
)
q_bg_mean_stddev_test = torch.tensor(
    test_result_df[["q_bg_mean_list", "q_bg_stddev_list"]].to_numpy()
)


# Lower trils for MVN(torch.zeros(3),)
L_train = torch.stack(list(train_result_df["L_pred"]))

# Mean and standard deviation
q_I_mean_train = q_I_mean_stddev_train[:, 0]
q_I_stddev_train = q_I_mean_stddev_train[:, 1]

q_bg_mean_train = q_bg_mean_stddev_train[:, 0]
q_bg_stddev_train = q_bg_mean_stddev_train[:, 1]


# Variational distributions
q_I_train = rsd.FoldedNormal(q_I_mean_train, q_I_stddev_train)
q_bg_train = rsd.FoldedNormal(q_bg_mean_train, q_bg_stddev_train)

samples_I_train = q_I_train.sample([50000])
lower_bound_q_I_train = torch.quantile(samples_I_train, 0.025, dim=0)
upper_bound_q_I_train = torch.quantile(samples_I_train, 0.975, dim=0)

samples_bg_train = q_bg_train.sample([50000])
lower_bound_q_bg_train = torch.quantile(samples_bg_train, 0.025, dim=0)
upper_bound_q_bg_train = torch.quantile(samples_bg_train, 0.975, dim=0)

# %%
q_I_mean_test = q_I_mean_stddev_test[:, 0]
q_I_stddev_test = q_I_mean_stddev_test[:, 1]

q_bg_mean_test = q_bg_mean_stddev_test[:, 0]
q_bg_stddev_test = q_bg_mean_stddev_test[:, 1]

# Variational distributions
q_I_test = rsd.FoldedNormal(q_I_mean_test, q_I_stddev_test)
q_bg_test = rsd.FoldedNormal(q_bg_mean_test, q_bg_stddev_test)

samples_I_test = q_I_test.sample([50000])
lower_bound_q_I_test = torch.quantile(samples_I_test, 0.025, dim=0)
upper_bound_q_I_test = torch.quantile(samples_I_test, 0.975, dim=0)

samples_bg_test = q_bg_test.sample([50000])
lower_bound_q_bg_test = torch.quantile(samples_bg_test, 0.025, dim=0)

upper_bound_q_bg_test = torch.quantile(samples_bg_test, 0.975, dim=0)


# %%
# Plot the first histogram
def plot_grid(
    samples,
    num_rows,
    num_cols,
    indices,
    lower_bound,
    upper_bound,
    means,
    sigmas,
    true_param,
    title="",
):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

    for i, idx in enumerate(indices):
        row = i // num_cols
        col = i % num_cols

        counts, bins, patches = axes[row, col].hist(
            samples[..., idx].numpy(),
            bins=100,
            alpha=0.3,
            linewidth=1,
            color="black",
            edgecolor="black",
        )

        axes[row, col].fill_betweenx(
            [0, max(counts)],
            lower_bound[idx],
            upper_bound[idx],
            color="red",
            alpha=0.2,
        )
        axes[row, col].hist(
            samples[..., idx].numpy(),
            bins=100,
            alpha=0.3,
            linewidth=1,
            color="gray",
            edgecolor="black",
        )

        axes[row, col].axvline(
            true_param[idx], color="red", linestyle="dashed", linewidth=1
        )
        axes[row, col].set_title(
            f"mu = {means[idx]:.2f}, sigma = {sigmas[idx]:.2f}\nindex = {idx}"
        )
        if col == 0:
            axes[row, col].set_ylabel("counts")
        axes[row, col].set_xlabel(r"$\lambda$")
        plt.subplots_adjust(
            left=0.05, right=0.98, bottom=0.1, top=0.9, wspace=0.2, hspace=1
        )
    fig.suptitle(title, fontsize=16)


# %%
# plot grid of train set q_bg distributions
true_poiss_rate = (torch.ones(len(samples_bg_train)) * 0.3).tolist()
indices = list(np.linspace(0, 16, 16, dtype=int))

plot_grid(
    samples_bg_train,
    4,
    4,
    indices,
    lower_bound_q_bg_train,
    upper_bound_q_bg_train,
    list(q_bg_mean_train),
    q_bg_stddev_train,
    true_poiss_rate,
    title="Train set q_bg distributions",
)
plt.show()

# %%
# plot grid test set q_bg distributions
true_poiss_rate = (torch.ones(len(samples_bg_test)) * 0.3).tolist()
indices = list(np.linspace(0, 16, 16, dtype=int))

plot_grid(
    samples_bg_test,
    4,
    4,
    indices,
    lower_bound_q_bg_test,
    upper_bound_q_bg_test,
    list(q_bg_mean_test),
    q_bg_stddev_test,
    true_poiss_rate,
    title="Test set q_bg distributions",
)

plt.show()

# %%
# plot grid of train set q_I distributions
plot_grid(
    samples_I_train,
    4,
    4,
    indices,
    lower_bound_q_I_train,
    upper_bound_q_I_train,
    list(q_I_mean_train),
    q_I_stddev_train,
    list(true_I_train),
    title="Train set q_I distributions",
)

plt.show()

# %%
# plot grid of  test set q_I distributions
plot_grid(
    samples_I_test,
    4,
    4,
    indices,
    lower_bound_q_I_test,
    upper_bound_q_I_test,
    list(q_I_mean_test),
    q_I_stddev_test,
    list(true_I_test),
    title="Test set q_I distributions",
)

plt.show()

# %%
# Plot profile of a single reflection
# Viewed from xy, xz, and zy planes
MVN = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(3), scale_tril=true_Ls[0]
)

grid_size = 100
x_ = torch.linspace(-4, 4, grid_size)
y_ = torch.linspace(-3, 3, grid_size)
z_ = torch.linspace(-3, 3, grid_size)
X, Y, Z = torch.meshgrid(x_, y_, z_)
points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)

log_probs = MVN.log_prob(points).reshape(grid_size, grid_size, grid_size)
profile = torch.exp(log_probs)


# Plot contour plots for different pairs of dimensions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# XY plane
axes[0].contour(
    X[:, :, grid_size // 2].numpy(),
    Y[:, :, grid_size // 2].numpy(),
    profile[:, :, grid_size // 2].detach().numpy(),
    levels=30,
)
axes[0].set_title("Contour Plot in XY plane")
axes[0].set_xlabel("X-axis")
axes[0].set_xlim(-4, 4)
axes[0].set_ylabel("Y-axis")

# XZ plane
axes[1].contour(
    X[:, grid_size // 2, :].numpy(),
    Z[:, grid_size // 2, :].numpy(),
    profile[:, grid_size // 2, :].detach().numpy(),
    levels=30,
)
axes[1].set_title("Contour Plot in XZ plane")
axes[1].set_xlabel("X-axis")
axes[1].set_xlim(-4, 4)
axes[1].set_ylabel("Z-axis")

# YZ plane
axes[2].contour(
    Y[grid_size // 2, :, :].numpy(),
    Z[grid_size // 2, :, :].numpy(),
    profile[grid_size // 2, :, :].detach().numpy(),
    levels=30,
)
axes[2].set_title("Contour Plot in YZ plane")
axes[2].set_xlabel("Y-axis")
axes[2].set_ylabel("Z-axis")
axes[2].set_xlim(-4, 4)

plt.show()

# %%
# 3D surface plot for a slice through the distribution (XY plane at Z=0)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    X[:, :, grid_size // 2].detach().numpy(),
    Y[:, :, grid_size // 2].detach().numpy(),
    profile[:, :, grid_size // 2].detach().numpy(),
    cmap="viridis",
)


ax.set_title("3D Surface Plot of Multivariate Normal Distribution (XY plane at Z=0)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Density")
plt.show()

# %%
# Plot a single 3d profile
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

# Create a grid of points
grid_size = 50
x = torch.linspace(-6, 6, grid_size)
y = torch.linspace(-6, 6, grid_size)
z = torch.linspace(-6, 6, grid_size)
X, Y, Z = torch.meshgrid(x, y, z)
points = torch.stack([X.ravel(), Y.ravel(), Z.ravel()], dim=-1)

# Evaluate the log probability density function at the grid points
log_probs = MVN.log_prob(points).reshape(grid_size, grid_size, grid_size)
profile = torch.exp(log_probs)


# Convert tensors to numpy arrays for plotting
X_np = X.numpy()
Y_np = Y.numpy()
Z_np = Z.numpy()
profile_np = profile.detach().numpy()

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Create a custom colormap with transparency for low values
viridis = plt.get_cmap("viridis")
viridis_colors = viridis(np.linspace(0, 1, 256))
viridis_colors[:128, -1] = np.linspace(
    0, 1, 128
)  # Add transparency to the first 50 colors
transparent_viridis = ListedColormap(viridis_colors)

norm = Normalize(vmin=0, vmax=profile_np.max())

# Plot the density points in 3D
# We use scatter for better visualization
sc = ax.scatter(X_np, Y_np, Z_np, c=profile_np, cmap=transparent_viridis, norm=norm)


# Add a color bar with the custom colormap
colorbar = plt.colorbar(
    ScalarMappable(norm=norm, cmap=transparent_viridis), ax=ax, shrink=0.5, aspect=5
)
colorbar.set_label("Density")

# Set plot title and labels
ax.set_title("3D Visualization of Multivariate Normal Distribution")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()

# %%
#
im, mask = next(iter(train_loader))
counts = torch.clamp(im[..., -1], min=0)
im_ = standardization(im, mask.squeeze(-1))
dxyz = im_[..., 3:6]
representation = encoder(im_, mask)
q_bg, q_I, profile, L = distribution_builder(representation, dxyz, mask)
z = q_I.rsample([100])
bg = q_bg.rsample([100])
rate = ((z.permute(1, 0, 2) * profile.unsqueeze(1))) + bg.permute(1, 0, 2)
ll = torch.distributions.Poisson(rate).log_prob(counts.unsqueeze(1))

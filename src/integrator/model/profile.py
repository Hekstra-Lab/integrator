import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.transforms import SoftplusTransform
from integrator.layers import Linear
from rs_distributions.transforms import FillScaleTriL


class SoftmaxProfile(nn.Module):
    def __init__(self, input_dim, rank, channels=3, height=21, width=21):
        super(SoftmaxProfile, self).__init__()
        self.rank = rank
        self.channels = channels
        self.height = height
        self.width = width

        # fully connected layer
        self.fc = nn.Linear(input_dim, rank * (channels + height + width))

    def forward(self, representation, dxyz=None):
        # Input representation shape: (batch_size, 1, input_dim)
        batch_size = representation.size(0)
        representation = representation.view(
            batch_size, -1
        )  # Flatten to (batch_size, input_dim)

        # Linear transformation to produce decomposition parameters
        decomposition_params = self.fc(
            representation
        )  # (batch_size, rank * (3 + 21 + 21))

        # Split the parameters into separate factors
        A = decomposition_params[:, : self.rank * self.channels].view(
            batch_size, self.rank, self.channels
        )  # (batch_size, rank, 3)

        B = decomposition_params[
            :, self.rank * self.channels : self.rank * (self.channels + self.height)
        ].view(
            batch_size, self.rank, self.height
        )  # (batch_size, rank, 21)

        C = decomposition_params[:, self.rank * (self.channels + self.height) :].view(
            batch_size, self.rank, self.width
        )  # (batch_size, rank, 21)

        # A = torch.nn.functional.softmax(A)
        # B = torch.nn.functional.softmax(B)
        # C = torch.nn.functional.softmax(C)
        # A = torch.nn.functional.softplus(A)
        # B = torch.nn.functional.softplus(B)
        # C = torch.nn.functional.softplus(C)

        # Reconstruct the background tensor using CP decomposition
        background = torch.einsum(
            "brc,brh,brw->bchw", A, B, C
        )  # (batch_size, 3, 21, 21)

        background = torch.softmax(
            background.view(batch_size, self.channels * self.height * self.width),
            dim=-1,
        )

        return background


class MVNProfile(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        rank=None,
        num_components=1,  # number of components in the mixture model
    ):
        super().__init__()
        self.dmodel = dmodel
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
        self.num_components = num_components
        if self.num_components == 1:
            self.scale_layer = Linear(self.dmodel, 6)
            self.mean_layer = Linear(self.dmodel, 3)

        else:
            self.mixture_weight_layer = Linear(dmodel, num_components)
            self.mean_layer = Linear(dmodel, (num_components - 1) * 3)
            self.scale_layer = Linear(self.dmodel, num_components * 6)

    def forward(self, representation, dxyz, num_planes=3):
        num_components = self.num_components
        batch_size = representation.size(0)

        if self.num_components == 1:
            # means = torch.zeros(
            # (batch_size, 1, 3), device=representation.device, requires_grad=False
            # ).to(torch.float32)

            means = self.mean_layer(representation).view(batch_size, 1, 3)

            scales = self.scale_layer(representation).view(batch_size, 1, 6)
            scales = torch.sigmoid(scales)

            L = self.L_transform(scales).to(torch.float32)
            mvn = MultivariateNormal(means, scale_tril=L)
            log_probs = mvn.log_prob(dxyz)
            profile = torch.exp(log_probs)

            return profile

        else:
            mixture_weights = self.mixture_weight_layer(representation).view(
                batch_size, self.num_components
            )

            mixture_weights = F.softmax(mixture_weights, dim=-1)

            means = self.mean_layer(representation).view(
                batch_size, self.num_components - 1, 3
            )

            zero_means = torch.zeros((batch_size, 1, 3), device=representation.device)

            means = torch.cat([zero_means, means], dim=1)

            scales = self.scale_layer(representation).view(
                batch_size, self.num_components, 6
            )

            L = self.L_transform(scales).to(torch.float32)

            mvn = MultivariateNormal(means, scale_tril=L)

            mix = Categorical(mixture_weights)

            gmm = MixtureSameFamily(
                mixture_distribution=mix, component_distribution=mvn
            )

            log_probs = gmm.log_prob(dxyz.view(441 * 3, batch_size, 3))

            profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

            return profile

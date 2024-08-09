import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from torch.distributions.transforms import SoftplusTransform
from rs_distributions.transforms import FillScaleTriL
from integrator.layers import Linear


class Profile(torch.nn.Module):
    def __init__(
        self,
        dmodel,
        num_components=5,  # number of components in the mixture model
    ):
        super().__init__()
        self.dmodel = dmodel
        self.L_transform = FillScaleTriL(diag_transform=SoftplusTransform())
        self.mixture_weight_layer = Linear(dmodel, num_components)
        self.mean_layer = Linear(dmodel, (num_components - 1) * 3)
        self.scale_layer = Linear(self.dmodel, num_components * 6)
        self.num_components = num_components

    def forward(self, representation, dxyz, num_planes=3):
        num_components = self.num_components

        batch_size = representation.size(0)

        mixture_weights = self.mixture_weight_layer(representation).view(
            batch_size, num_components
        )

        mixture_weights = F.softmax(mixture_weights, dim=-1)

        means = self.mean_layer(representation).view(batch_size, num_components - 1, 3)

        zero_means = torch.zeros((batch_size, 1, 3), device=representation.device)

        means = torch.cat([zero_means, means], dim=1)

        scales = self.scale_layer(representation).view(batch_size, num_components, 6)

        L = FillScaleTriL(diag_transform=SoftplusTransform())(scales)

        mvn = MultivariateNormal(means, scale_tril=L)

        mix = Categorical(mixture_weights)

        gmm = MixtureSameFamily(mixture_distribution=mix, component_distribution=mvn)

        log_probs = gmm.log_prob(dxyz.view(441 * 3, batch_size, 3))

        profile = torch.exp(log_probs).view(batch_size, num_planes * 441)

        return profile, L

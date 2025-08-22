import torch
from torch import Tensor
from torch.distributions import Gamma

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution, MetaData


class GammaDistribution(BaseDistribution[Gamma]):
    def __init__(
        self,
        dmodel: int,
        out_features: int = 2,
        use_metarep: bool = False,
    ):
        super().__init__()

        self.use_metarep = use_metarep

        if self.use_metarep:
            # separate layers for params1 and params2
            self.fc1 = Linear(
                in_features=dmodel,
                out_features=1,
            )
            self.fc2 = Linear(
                in_features=dmodel * 2,
                out_features=1,
            )
        else:
            # single layer for both params
            self.fc = Linear(
                in_features=dmodel,
                out_features=out_features,
            )

    def distribution(
        self,
        concentration: Tensor,
        rate: Tensor,
    ) -> Gamma:
        concentration = self.constraint(concentration)
        rate = self.constraint(rate)
        return Gamma(concentration.flatten(), rate.flatten())

    def forward(
        self,
        x: Tensor,
        *,
        meta_data: MetaData | None = None,
    ) -> Gamma:
        if meta_data is not None and meta_data.metadata is not None:
            assert metarep is not None, "metarep required when use_metarep=True"
            params1 = self.fc1(x)
            combined_rep = torch.cat([x, meta_data.metadata], dim=1)
            params2 = self.fc2(combined_rep)
            gamma = self.distribution(params1, params2)
        else:
            params = self.fc(x)
            gamma = self.distribution(params[..., 0], params[..., 1])

        return gamma


if __name__ == "__main__":
    # Example usage
    dmodel = 64
    gamma_dist = GammaDistribution(dmodel)
    representation = torch.randn(10, dmodel)  # Example input
    metarep = torch.randn(10, dmodel * 2)  # Example metadata representation

    # use without metadata
    qbg = gamma_dist(representation)

    # use with metadata
    qbg = gamma_dist(representation, meta_data=MetaData(metarep))

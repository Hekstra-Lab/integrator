import torch
from torch import Tensor
from torch.distributions import LogNormal

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution, MetaData


class LogNormalDistribution(BaseDistribution[LogNormal]):
    def __init__(
        self,
        dmodel: int,
        out_features: int = 2,
        use_metarep: bool = False,
    ):
        """

        Args:
            dmodel:
            out_features:
            use_metarep:
        """
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
            self.fc = Linear(
                in_features=dmodel,
                out_features=out_features,
            )

    def distribution(self, loc, scale) -> LogNormal:
        """
        Args:
            loc ():
            scale ():

        Returns:

        """
        scale = self.constraint(scale)
        return LogNormal(loc=loc.flatten(), scale=scale.flatten())

    def forward(self, x: Tensor, *, meta_data: MetaData | None = None) -> LogNormal:
        """

        Args:
            x: Tensor representation of the shoebox
            meta_data: Additional Tensors besides the shoebox representation

        Returns:

        """
        if meta_data is not None and meta_data.metadata is not None:
            x_metadata = meta_data.metadata
            params1 = self.fc1(x)
            combined_rep = torch.cat([x, x_metadata], dim=1)
            params2 = self.fc2(combined_rep)
            lognormal = self.distribution(params1, params2)

        else:
            params = self.fc(x)
            lognormal = self.distribution(params[..., 0], params[..., 1])

        return lognormal


if __name__ == "__main__":
    # generate a batch of 10 representation vectors
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)

    # initialize a LogNormalDistribution object
    lognormal = LogNormalDistribution(dmodel=64, use_metarep=True)

    # Use with metadata
    q = lognormal(representation, meta_data=MetaData(metadata=metarep))

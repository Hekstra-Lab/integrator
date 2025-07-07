import torch
from torch.distributions import LogNormal

from integrator.layers import Constraint, Linear
from integrator.model.distributions import BaseDistribution


class LogNormalDistribution(BaseDistribution):
    def __init__(
        self,
        dmodel,
        constraint=Constraint(),
        out_features: int = 2,
        use_metarep: bool = False,
    ):
        """
        Args:
            dmodel (int):
            constraint ():
            out_features (int):
            use_metarep (bool): Boolean indicating if metadata is being used
        """
        super().__init__(
            q=LogNormal,
        )
        self.use_metarep = use_metarep

        self.constraint = constraint

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

    def distribution(self, loc, scale):
        """
        Args:
            loc ():
            scale ():

        Returns:

        """
        scale = self.constraint(scale)
        return self.q(loc=loc.flatten(), scale=scale.flatten())

    def forward(self, representation, metarep=None):
        """

        Args:
            representation ():
            metarep ():

        Returns:

        """
        if self.use_metarep:
            assert metarep is not None, "metarep required when use_metarep=True"
            params1 = self.fc1(representation)
            combined_rep = torch.cat([representation, metarep], dim=1)
            params2 = self.fc2(combined_rep)
            lognormal = self.distribution(params1, params2)

        else:
            params = self.fc(representation)
            lognormal = self.distribution(params[..., 0], params[..., 1])
        return lognormal


if __name__ == "__main__":
    # generate a batch of 10 representation vectors
    representation = torch.randn(10, 64)
    metarep = torch.randn(10, 64)

    # initialize a LogNormalDistribution object
    model = LogNormalDistribution(dmodel=64, use_metarep=True)

    # get the parameterized torch.distributions.LogNormal object
    lognormal = model(representation, metarep)

    # sample from the distribution
    lognormal.rsample([100])

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Dirichlet

from integrator.layers import Linear
from integrator.model.distributions import BaseDistribution

# Shape of the sheobox
# For 2D: [height, width]
# For 3D: [depth ,height, width]
type InputShape = tuple[int, int] | tuple[int, int, int]


class DirichletDistribution(BaseDistribution[Dirichlet]):
    """

    Attributes:
        dmodel: The dimension of the shoebox representation
    """

    def __init__(
        self,
        dmodel: int = 64,
        input_shape: InputShape = (3, 21, 21),
    ):
        """

        Args:
            dmodel: Dimension of input features
            input_shape:
        """
        super().__init__(eps=1e-6)

        if len(input_shape) == 3:
            self.num_components = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            self.num_components = input_shape[0] * input_shape[1]

        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, self.num_components)
        self.dmodel = dmodel

    def forward(
        self,
        x: Tensor,
    ) -> Dirichlet:
        x = self.alpha_layer(x)
        x = F.softplus(x) + self.eps
        qp = Dirichlet(x)

        return qp


if __name__ == "__main__":
    x = torch.rand(10, 64)
    dirichlet = DirichletDistribution(dmodel=64, input_shape=(21, 21))

    qp = dirichlet(x)

    # 3D Case

    dirichlet = DirichletDistribution(dmodel=64, input_shape=(3, 21, 21))
    qp = dirichlet(x)

    dirichlet = DirichletDistribution(dmodel=64, input_shape=(21, 21))
    q = dirichlet(rep)

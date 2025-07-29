import torch
import torch.nn.functional as F

from integrator.layers import Linear


class DirichletDistribution(torch.nn.Module):
    def __init__(self, dmodel=None, input_shape=(3, 21, 21)):
        """

        Args:
            dmodel (int): Integer specifying the dimensions of the shoebox representation
            input_shape (tuple): Tuple of integers specifying the depth, height, and width of the input shoebox
        """
        super().__init__()

        if len(input_shape) == 3:
            self.num_components = input_shape[0] * input_shape[1] * input_shape[2]
        elif len(input_shape) == 2:
            self.num_components = input_shape[0] * input_shape[1]

        if dmodel is not None:
            self.alpha_layer = Linear(dmodel, self.num_components)
        self.dmodel = dmodel
        self.eps = 1e-6

    def forward(self, x):
        """
        Args:
            x (torch.tensor): input representation tensor

        Returns: `torch.distributions.Dirichlet`

        """
        x = self.alpha_layer(x)
        x = F.softplus(x) + self.eps
        q_p = torch.distributions.Dirichlet(x)

        return q_p


if __name__ == "__main__":
    rep = torch.rand(10, 64)
    dirichlet = DirichletDistribution(dmodel=64, input_shape=(21, 21))

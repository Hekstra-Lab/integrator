import torch
import torch.nn.functional as F

class Constraint(torch.nn.Module):
    def __init__(self, eps=1e-12, beta=1.0):
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("beta", torch.tensor(beta))

    def forward(self, x):
        return F.softplus(x, beta=self.beta) + self.eps

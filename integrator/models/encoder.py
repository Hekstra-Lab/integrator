from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class MLPEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel, d_in=5):
        super().__init__()
        # self.mlp_1 = MLP(dmodel, depth, d_in=d_in, output_dims=dmodel + 1)
        self.mlp_1 = MLP(dmodel, depth, d_in=d_in, output_dims=dmodel + 2)
        # self.mlp_2 = MLP(dmodel, depth)

    def forward(self, xy, dxy, counts, mask=None):
        per_pixel = torch.concat(
            (
                xy,
                dxy,
                counts[..., None],
            ),
            axis=-1,
        )

        out = self.mlp_1(per_pixel)
        score = out[..., :1]
        ps = torch.special.expit(out[...,1:2].permute(2,0,1))
        refl = out[..., 2:]
        if mask is not None:
            score = torch.where(mask[..., None], score, -np.inf)
        score = torch.softmax(score, axis=-2)

        refl = torch.matmul(score.transpose(-2, -1), refl)
        # refl = self.mlp_2(refl)
        return refl, ps

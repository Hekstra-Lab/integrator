from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP, MLPImage, MLPOut1, MLPPij, MLPPij2


def convexcombination(tnsr1):
    score = tnsr1[..., :1]
    score = torch.softmax(score, axis=-2)
    rep = tnsr1[..., 1:]
    rep = torch.matmul(rep.transpose(-2, -1), score)
    return rep


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
        # ps = torch.special.expit(out[...,1:2])
        p = out[..., 1:2]
        # ps = p / p.sum(axis=1)[..., None] #normalizes p-array
        ps = torch.softmax(p, axis=-2)
        refl = out[..., 2:]
        if mask is not None:
            score = torch.where(mask[..., None], score, -np.inf)
        score = torch.softmax(score, axis=-2)

        refl = torch.matmul(score.transpose(-2, -1), refl)
        # refl = self.mlp_2(refl)
        return refl, ps


class MLPPixelEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel, d_in_refl=4):
        super().__init__()
        self.mlp_1 = MLP(dmodel, depth, d_in=d_in_refl, output_dims=dmodel)

    def forward(self, xy, dxy, mask=None):
        per_pixel = torch.concat(
            (
                xy,
                dxy,
            ),
            axis=-1,
        )
        out = self.mlp_1(per_pixel)
        pixel_rep = torch.relu(out)
        return pixel_rep


class MLPReflEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel, d_in_refl=5):
        super().__init__()
        self.mlp_1 = MLP(dmodel, depth, d_in=d_in_refl, output_dims=dmodel + 1)

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
        # ConvexCombination
        score = out[..., :1]
        refl = out[..., 1:]
        if mask is not None:
            score = torch.where(mask[..., None], score, -np.inf)
        score = torch.softmax(score, axis=-2)

        refl = torch.matmul(score.transpose(-2, -1), refl)
        # refl = self.mlp_2(refl)
        return refl


class MLPImageEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel):
        super().__init__()
        self.mlp_1 = MLPImage(dmodel, depth, output_dims=dmodel + 1)

    def forward(self, refl_representation):
        out = self.mlp_1(refl_representation)
        out = out.view(out.shape[0], out.shape[-1])
        score = out[..., :1]
        score = torch.softmax(score, axis=-2)
        image_rep = out[..., 1:]
        image_rep = torch.matmul(image_rep.transpose(-2, -1), score)
        return image_rep


class MLPOut1Encoder(torch.nn.Module):
    def __init__(self, depth, dmodel):
        super().__init__()
        self.mlp_1 = MLPOut1(dmodel, depth, output_dims=2 + 2 + 1)

    def forward(self, image_rep, refl_representation):
        summed = refl_representation + image_rep.transpose(-2, -1)
        out1 = self.mlp_1(summed)
        out1 = out1.view(out1.shape[0], out1.shape[-1])
        return out1


class MLPPijEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel):
        super().__init__()
        self.mlp_1 = MLPPij(dmodel, depth, output_dims=dmodel + 1)
        self.mlp_2 = MLPPij2(dmodel, output_dims=1024)

    def forward(self, image_rep, refl_representation, pixel_rep):
        refl_representation = refl_representation.view(
            refl_representation.shape[0], refl_representation.shape[-1]
        )
        image_rep = image_rep.transpose(-2, -1)
        sum1 = refl_representation + image_rep
        sum1 = sum1.unsqueeze(1).expand_as(pixel_rep)
        sum2 = pixel_rep + sum1
        out = self.mlp_1(sum2)
        out = convexcombination(out)
        out = out.view(out.shape[0], out.shape[1])
        out2 = self.mlp_2(out)
        out2 = torch.softmax(out2, axis=1)  # pij matrix
        return out2

from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer, Transformer
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


class RotationReflectionEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel, feature_dim, dropout=None):
        super().__init__()
        self.dropout = None
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=self.dropout, output_dims=dmodel
        )
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask=None):
        out = self.mlp_1(shoebox_data)
        pooled_out = self.mean_pool(out, mask)
        return pooled_out


class Encoder(torch.nn.Module):
    def __init__(self, depth, dmodel, feature_dim, dropout=None):
        super().__init__()
        self.dropout = None
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=self.dropout, output_dims=dmodel
        )
        self.mean_pool = MeanPool()
        # I + bg + MVN
        # 2 + 2 + 6 + 3
        self.linear = Linear(64, 2 + 2 + 6 + 3)

    def forward(self, shoebox_data, mask=None):
        out = self.mlp_1(shoebox_data)
        pooled_out = self.mean_pool(out, mask)
        # outputs = self.linear(pooled_out)
        return pooled_out


class ReflectionTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        depth,
        dmodel,
        feature_dim,
        dropout=None,
        d_hid=2000,
        nhead=8,
        nlayers=6,
        batch_first=True,
    ):
        super().__init__()
        self.dropout = None
        self.d_hid = d_hid
        self.nhead = nhead
        self.nlayers = nlayers
        self.batch_first = batch_first
        self.dmodel = dmodel
        self.mlp_1 = MLP(
            dmodel, depth, d_in=feature_dim, dropout=None, output_dims=dmodel
        )
        self.transformer = Transformer(
            d_model=self.dmodel,
            d_hid=self.d_hid,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=self.batch_first,
            nlayers=self.nlayers,
        )
        self.mean_pool = MeanPool()

    def forward(self, shoebox_data, mask=None):
        out = self.mlp_1(shoebox_data)
        transformer_out = self.transformer(out, src_mask=mask)
        pooled_out = self.mean_pool(transformer_out, mask)
        return pooled_out


# Embed shoeboxes
class MeanPool(torch.nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        # self.register_buffer(
        # "dim",
        # torch.tensor(),
        # )

    def forward(self, data, mask=None):
        out = data.sum(1, keepdim=True)
        if mask is None:
            denom = data.shape[-1]
        else:
            denom = mask.sum(-1, keepdim=True)
        out = out / denom.unsqueeze(-1)

        return out


class RotationPixelEncoder(torch.nn.Module):
    """
    Encodes pixels into (num_reflection x max_voxel_sixe x d_model)
    """

    def __init__(self, depth, dmodel, d_in_refl=4, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLP(
            dmodel, depth, d_in=d_in_refl, dropout=self.dropout, output_dims=dmodel
        )

    def forward(self, shoebox, mask=None):
        out = self.mlp_1(shoebox)
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
    def __init__(self, depth, dmodel, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLPImage(dmodel, depth, dropout=dropout, output_dims=dmodel + 1)

    def forward(self, refl_representation):
        out = self.mlp_1(refl_representation)
        out = out.view(out.shape[0], out.shape[-1])
        score = out[..., :1]
        score = torch.softmax(score, axis=-2)
        image_rep = out[..., 1:]
        image_rep = torch.matmul(image_rep.transpose(-2, -1), score)
        return image_rep


class MLPOut1Encoder(torch.nn.Module):
    def __init__(self, depth, dmodel, dropout=None):
        super().__init__()
        k
        self.dropout = dropout
        self.mlp_1 = MLPOut1(dmodel, depth, dropout=self.dropout, output_dims=2 + 2 + 1)

    def forward(self, image_rep, refl_representation):
        summed = refl_representation + image_rep.transpose(-2, -1)
        out1 = self.mlp_1(summed)
        # out1 = out1.view(out1.shape[0], out1.shape[-1])
        return out1


class IntensityBgPredictor(torch.nn.Module):
    """
    Ouputs mu and sigma for LogNorm(mu,sigma) and Background
    """

    def __init__(self, depth, dmodel, dropout=None, beta=1.0, eps=1e-4):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLPOut1(dmodel, depth, dropout=self.dropout, output_dims=2 + 1)

    def forward(self, refl_representation):
        out1 = self.mlp_1(refl_representation)
        out1 = out1.view(out1.shape[0], out1.shape[-1])
        return out1


class MLPPijEncoder(torch.nn.Module):
    def __init__(self, depth, dmodel, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLPPij(dmodel, depth, dropout=self.dropout, output_dims=1)
        # self.mlp_2 = MLPPij2(dmodel, output_dims=1024)

    def forward(self, image_rep, refl_representation, pixel_rep):
        refl_representation = refl_representation.view(
            refl_representation.shape[0], refl_representation.shape[-1]
        )
        image_rep = image_rep.transpose(-2, -1)
        sum1 = refl_representation + image_rep
        sum1 = sum1.unsqueeze(1).expand_as(pixel_rep)
        sum2 = pixel_rep + sum1
        out = self.mlp_1(sum2)
        out = torch.softmax(out, axis=-2)
        out.squeeze()
        # out = convexcombination(out)
        # out = out.view(out.shape[0], out.shape[1])
        # out2 = self.mlp_2(out)
        # out2 = torch.softmax(out2, axis=1)  # pij matrix
        return out


class ProfilePredictor(torch.nn.Module):
    """
    Outputs p_ij matrix to scale Intensity values
    """

    def __init__(self, dmodel, depth, max_pixel, dropout=None):
        super().__init__()
        self.dropout = dropout
        self.mlp_1 = MLPPij(dmodel, depth, dropout=self.dropout, output_dims=1)
        self.mean_pool = MeanPool()
        self.linear = Linear(64, max_pixel)

    def forward(self, refl_representation, pixel_rep, mask=None):
        sum = pixel_rep + refl_representation.expand_as(pixel_rep)
        out = self.mlp_1(sum)
        out = torch.softmax(out, axis=-2)
        out = out.squeeze(-1)

        # pooled_out = self.mean_pool(out, mask)
        # out = self.linear(pooled_out)
        # out = torch.softmax(out, axis=-1)
        # out = out.unsqueeze(-2)

        return out

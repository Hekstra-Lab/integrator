import torch
from torch.nn import Linear
from integrator.layers import Residual, MLP, MeanPool
from integrator.model.encoders import BaseEncoder


class FcEncoder(BaseEncoder):
    def __init__(self, depth=10, dmodel=64, feature_dim=7, dropout=None):
        super().__init__()
        self.linear = Linear(feature_dim, dmodel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.batch_norm = nn.BatchNorm1d(dmodel)
        self.layer_norm = torch.nn.LayerNorm(dmodel)
        self.mlp_1 = MLP(dmodel, depth, dropout=dropout, output_dims=dmodel)

    def forward(self, shoebox_data):
        # shoebox_data shape: [batch_size, num_pixels, feature_dim]
        batch_size, features = shoebox_data.shape

        # Initial linear transformation
        out = self.linear(shoebox_data)
        out = self.relu(out)
        out = self.layer_norm(out)
        out = self.mlp_1(out)
        return out

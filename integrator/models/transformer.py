from pylab import *
import torch
import torch.nn as nn
from integrator.layers import Linear, Transformer


class TransformerModel(nn.Module):
    def __init__(self, d_model: int, d_hid: int, nhead: int, nlayers: int):
        super().__init__()
        # Layers
        self.d_model = d_model
        self.linear1 = Linear(5, self.d_model)
        self.linear2 = Linear(self.d_model, self.d_model + 1)
        self.linear3 = Linear(self.d_model, 1)
        self.linear4 = Linear(1024, 8)
        self.transformer = Transformer(d_model, d_hid, nhead, nlayers)

    def forward(self, xy, dxy, counts):
        per_pixel = torch.concat(
            (
                xy,
                dxy,
                counts[..., None],
            ),
            axis=-1,
        )
        output = self.linear1(per_pixel)  # batch_size x pixel x d_model ,
        output = self.transformer(output)  # batch_size x pixel x d_model ,
        output = self.linear2(output)  # batch_size x pixel x d_model + 1 ,
        # output1 = output[..., -1:].view(output.shape[0], output.shape[1])  # p_ij matrix
        output1 = output[..., -1:]  # p_ij matrix
        output2 = self.linear3(
            output[..., :-1].view(output.shape[0], output.shape[1], output.shape[2] - 1)
        )  # batch_size x pixel x 1 ,
        # output2 = output2.view(output2.shape[0], output2.shape[1])
        output2 = output2.view(output2.shape[0], output2.shape[-1], output2.shape[1])
        # output2 = self.linear4(output2)
        output1 = torch.softmax(output1, axis=-1)
        return output2, output1  # (representation,pij)

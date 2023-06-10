from pylab import *
from integrator.layers.util import weight_initializer
from integrator.layers import Linear
import torch
import math




class Attention(torch.nn.Module):
    """
    A modified variant of self attention
    """
    def __init__(self, heads, rank, dropout=None):
        super().__init__()
        self.heads = heads
        self.rank = rank
        self.built = False
        self.dropout = dropout

    def build(self, data):
        weight_shape = (self.heads, data.shape[-1], self.rank)
        weights = torch.empty(weight_shape, device=data.device, dtype=data.dtype)
        weights = weight_initializer(weights)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)

        w_out = torch.ones(self.heads, device=data.device, dtype=data.dtype) / self.heads
        self.w_out = torch.nn.Parameter(w_out, requires_grad=True)
        self.built = True

    @staticmethod
    def sequence_lengths_to_mask(lengths, max_length=None):
        if max_length is None:
            max_length = lengths.max()

        mask = torch.where(
            torch.arange(max_length)[...,None,:] <= lengths[...,:,None], 
            True, 
            False,
        )
        return mask

    def forward(self, data, mask=None):
        if not self.built:
            self.build(data)

        proj = torch.einsum("nld,hda->nlha", data, self.weights)
        score = torch.einsum("nlha,nxha->nhlx", proj, proj)
        if mask is None:
            mask = torch.ones_like(data[...,0], dtype=torch.bool)

        if self.dropout is not None:
            diag_mask = torch.diag(torch.ones(score.shape[-1], dtype=torch.bool, device=score.device))[...,:,:]
            keep = (torch.rand_like(score[-2, -1]) > self.dropout) 

            if mask is not None:
                keep = keep & mask[...,None,:] 

            keep = keep | diag_mask #In case there are only False
            score = torch.where(keep[...,None,:,:], score, -np.inf)
        else:
            score = torch.where(mask[...,None,None,:], score, -np.inf)

        score = torch.softmax(score, axis=-1)
        out = torch.einsum("nhll,nld->nhld", score, data)
        out = torch.einsum("nhld,h->nld", out, self.w_out)
        return out + data

if __name__=="__main__":
    b = 100
    l = 1024
    h = 8
    dmodel = 32
    rank = 16
    min_length = 50
    dropout = 0.3
    lengths = np.random.choice(np.arange(min_length, l), b)
    lengths = torch.tensor(lengths)
    att = Attention(h, rank, dropout=dropout)
    mask = att.sequence_lengths_to_mask(lengths, l)
    x = torch.rand((b, l, dmodel))
    z = att(x, mask=mask)
    z = att.cuda()(x.cuda(), mask.cuda())

    dropout = None
    att = Attention(h, rank, dropout=dropout)
    mask = att.sequence_lengths_to_mask(lengths, l)
    x = torch.rand((b, l, dmodel))
    z = att(x, mask=mask)
    z = att.cuda()(x.cuda(), mask.cuda())

    from IPython import embed
    embed(colors='linux')

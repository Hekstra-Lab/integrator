from pylab import *
import torch
from integrator.layers import Linear,ResidualLayer,Attention

class Block(torch.nn.Module):
    def __init__(self, heads, rank, dropout=None):
        super().__init__()
        self.dmodel = heads * rank
        self.attention = Attention(heads, rank, dropout)
        self.linear = ResidualLayer(self.dmodel, dropout=dropout)

    def forward(self, data, mask=None):
        out = self.attention(data, mask) + data
        out = self.linear(out)
        return out

class Transformer(torch.nn.Module):
    def __init__(self, heads, rank, depth, dropout=None):
        super().__init__()
        self.dmodel = heads * rank
        layers = []
        self.linear = None
        for i in range(depth):
            layers.append(Block(heads, rank, dropout))
        self.main = torch.nn.Sequential(*layers)

    def build(self, data):
        d_in = data.shape[-1]
        self.linear = Linear(d_in, self.dmodel, device=data.device, dtype=data.dtype)

    def forward(self, data, mask=None, **kwargs):
        if self.linear is None:
            self.build(data)

        out = self.linear(data)
        for block in self.main:
            out = block(out, mask)

        return out

if __name__=="__main__":
    b = 100
    d = 12
    l = 512
    h = 8
    depth = 5
    rank = 8
    min_length = 50
    dropout = None
    lengths = np.random.choice(np.arange(min_length, l), b)
    lengths = torch.tensor(lengths)
    mask = Attention.sequence_lengths_to_mask(lengths, l)
    transformer = Transformer(h, rank, depth, dropout)
    x = torch.rand((b, l, d))
    #z = transformer(x, mask=mask)
    z = transformer.cuda()(x.cuda(), mask.cuda())
    from IPython import embed
    embed(colors='linux')


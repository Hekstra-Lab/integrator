from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class Integrator(torch.nn.Module):
    def __init__(self, encoder, profile, likelihood):
        super().__init__()
        self.encoder = encoder
        self.profile = profile
        self.likelihood = likelihood
        self.counts_std = None

    def set_counts_std(self, value):
        self.counts_std = torch.nn.Parameter(value, requires_grad=False)

    def get_intensity_sigma(self, xy, dxy, counts, mask=None, batch_size=100):
        I, SigI = [], []
        for batch in zip(
            torch.split(xy, batch_size, dim=0),
            torch.split(dxy, batch_size, dim=0),
            torch.split(counts, batch_size, dim=0),
            torch.split(mask, batch_size, dim=0),
        ):
            device = next(self.parameters()).device
            i, s = self.get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
            I.append(i.detach().cpu().numpy())
            SigI.append(s.detach().cpu().numpy())
        I, SigI = np.concatenate(I), np.concatenate(SigI)
        return I, SigI

    def get_per_spot_normalization(self, counts, mask=None):
        if mask is None:
            return counts.mean(-1, keepdims=True)
        return counts.sum(-1, keepdims=True) / mask.sum(-1, keepdims=True)

    def get_intensity_sigma_batch(self, xy, dxy, counts, mask=None):
        norm_factor = self.get_per_spot_normalization(counts, mask)
        representation, p = self.encoder(xy, dxy, counts / norm_factor, mask)
        params = self.profile.get_params(representation)
        q = self.profile.distribution(params)
        I, SigI = q.mean, q.stddev
        I, SigI = I * norm_factor, SigI * norm_factor
        return I, SigI

    def forward(self, xy, dxy, counts, mask=None, mc_samples=100):
        norm_factor = self.get_per_spot_normalization(counts, mask=mask)
        representation, p = self.encoder(xy, dxy, counts / norm_factor, mask=mask)
        bg, q = self.profile(representation, dxy, mask=mask)

        # profile = profile * norm_factor
        bg = bg * norm_factor
        # p = p * norm_factor[..., None]

        ll, kl_term = self.likelihood(norm_factor, counts, p, bg, q, mc_samples)
        if mask is None:
            nll = -ll.mean()
        else:
            nll = -torch.where(mask, ll, 0.0).sum() / mask.sum()

        return nll + kl_term, p

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm

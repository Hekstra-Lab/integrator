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
        # I, SigI = np.concatenate(I), np.concatenate(SigI)
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

        return nll + kl_term, p.detach(), bg

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm


class IntegratorBern(torch.nn.Module):
    def __init__(
        self,
        reflencoder,
        imageencoder,
        paramencoder,
        pixelencoder,
        pijencoder,
        bglognorm,
        likelihood,
    ):
        super().__init__()
        self.reflencoder = reflencoder
        self.imageencoder = imageencoder
        self.paramencoder = paramencoder
        self.pixelencoder = pixelencoder
        self.pijencoder = pijencoder
        self.bglognorm = bglognorm
        self.likelihood = likelihood
        self.counts_std = None

    ################################
    def set_counts_std(self, value):
        self.counts_std = torch.nn.Parameter(value, requires_grad=False)

    def get_intensity_sigma(self, xy, dxy, counts, batch_size=100):
        I, SigI = [], []
        for batch in zip(
            torch.split(xy, batch_size, dim=0),
            torch.split(dxy, batch_size, dim=0),
            torch.split(counts, batch_size, dim=0),
        ):
            device = next(self.parameters()).device
            i, s = self.get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
            I.append(i.detach().cpu().numpy())
            SigI.append(s.detach().cpu().numpy())
        I, SigI = np.concatenate(I), np.concatenate(SigI)
        return I, SigI

    # def get_per_spot_normalization(self, counts):
    # return counts.mean(-1, keepdims=True)

    def get_per_spot_normalization(counts):
        return torch.clamp(counts[..., -1], min=0).sum(-1)

    def get_intensity_sigma_batch(self, xy, dxy, counts):
        norm_factor = self.get_per_spot_normalization(counts)

        reflrep = self.reflencoder(xy, dxy, counts / norm_factor)
        imagerep = self.imageencoder(reflrep)
        paramrep = self.paramencoder(imagerep, reflrep)
        pixelrep = self.pixelencoder(xy, dxy)
        pijrep = self.pijencoder(imagerep, reflrep, pixelrep)

        q = self.bglognorm.distribution(paramrep)
        I, SigI = q.mean, q.stddev
        I, SigI = I * norm_factor, SigI * norm_factor
        return I, SigI

    def forward(self, xy, dxy, counts, mc_samples=100):
        norm_factor = self.get_per_spot_normalization(counts)
        reflrep = self.reflencoder(xy, dxy, counts / norm_factor)
        imagerep = self.imageencoder(reflrep)
        paramrep = self.paramencoder(imagerep, reflrep)
        pixelrep = self.pixelencoder(xy, dxy)
        pijrep = self.pijencoder(imagerep, reflrep, pixelrep)

        bg, q = self.bglognorm(paramrep, dxy)

        bg = bg * norm_factor
        # p = p * norm_factor[..., None]

        ll, kl_term = self.likelihood(norm_factor, counts, pijrep, bg, q, mc_samples)
        nll = -ll.mean()

        return nll + kl_term, bg

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm


class IntegratorV2(torch.nn.Module):
    def __init__(
        self,
        reflencoder,
        paramencoder,
        pixelencoder,
        pijencoder,
        bglognorm,
        likelihood,
    ):
        super().__init__()
        self.reflencoder = reflencoder
        self.paramencoder = paramencoder
        self.pixelencoder = pixelencoder
        self.pijencoder = pijencoder
        self.bglognorm = bglognorm
        self.likelihood = likelihood
        self.counts_std = None

    def set_counts_std(self, value):
        self.counts_std = torch.nn.Parameter(value, requires_grad=False)

    def get_intensity_sigma(self, shoebox, batch_size=100):
        I, SigI = [], []
        xyz = shoebox[..., 0:3]
        dxyz = shoebox[..., 3:6]
        counts = shoebox[..., -1]
        for batch in zip(
            torch.split(xyz, batch_size, dim=0),
            torch.split(dxyz, batch_size, dim=0),
            torch.split(counts, batch_size, dim=0),
        ):
            device = next(self.parameters()).device
            i, s = self.get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
            I.append(i.detach().cpu().numpy())
            SigI.append(s.detach().cpu().numpy())
        I, SigI = np.concatenate(I), np.concatenate(SigI)
        return I, SigI

    def get_per_spot_normalization(self, counts):
        return torch.clamp(counts[:, :, -1], min=0).sum(-1)

    def get_intensity_sigma_batch(self, shoebox):
        norm_factor = self.get_per_spot_normalization(shoebox)
        shoebox[:, :, -1] = shoebox[:, :, -1] / norm_factor.unsqueeze(-1)
        reflrep = self.reflencoder(shoebox)
        paramrep = self.paramencoder(reflrep)
        pixelrep = self.pixelencoder(shoebox[:, :, 0:-1])
        pijrep = self.pijencoder(reflrep, pixelrep)

        q = self.bglognorm.distribution(paramrep)
        I, SigI = q.mean, q.stddev
        I, SigI = I * norm_factor, SigI * norm_factor
        return I, SigI

    def forward(self, shoebox, mask, mc_samples=5):
        norm_factor = self.get_per_spot_normalization(shoebox)
        shoebox[..., -1] = shoebox[..., -1] / norm_factor.unsqueeze(-1)
        shoebox[..., -1][shoebox[..., -1].isnan()] = 0
        shoebox[..., -1][[shoebox[..., -1] == -float("inf")]] = 0
        counts = torch.clamp(shoebox[:, :, -1], min=0)
        reflrep = self.reflencoder(shoebox, mask)
        paramrep = self.paramencoder(reflrep)
        pixelrep = self.pixelencoder(shoebox[:, :, 0:-1])
        pijrep = self.pijencoder(reflrep, pixelrep)

        # Removing nans
        paramrep[paramrep.isnan()] = 0
        pijrep[pijrep.isnan()] = 0

        bg, q = self.bglognorm(paramrep)

        bg = bg * norm_factor.unsqueeze(1)
        # p = p * norm_factor[..., None]

        ll, kl_term = self.likelihood(norm_factor, counts, pijrep, bg, q, mc_samples)
        nll = -ll.mean()

        return nll + kl_term, bg

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm

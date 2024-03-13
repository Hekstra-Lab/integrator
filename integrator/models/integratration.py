from pylab import *
import torch
from integrator.layers import Linear, ResidualLayer
from integrator.models import MLP


class IntegratorTransformer(torch.nn.Module):
    """
    Attributes:
        refl_encoder: encodes shoebox as 1 x d_model
        param_encoder: outputs mu,sigma,and background
        pixel_encoder: encodes shoebox into pixel x d_model
        profile_model: outputs per-pixel profile values
        bglognorm:
        likelihood:
    """

    def __init__(
        self,
        reflection_encoder,
        param_encoder,
        pixel_encoder,
        pixel_transfomer,
        profile_model,
        bglognorm,
        likelihood,
    ):
        super().__init__()
        self.refl_encoder = reflection_encoder
        self.param_encoder = param_encoder
        self.pixel_encoder = pixel_encoder
        self.pixel_transfomer = pixel_transfomer
        self.profile_model = profile_model
        self.bglognorm = bglognorm
        self.likelihood = likelihood
        self.counts_std = None

    def set_counts_std(self, value):
        self.counts_std = torch.nn.Parameter(value, requires_grad=False)

    def get_intensity_sigma(self, shoebox):
        I, SigI = [], []
        xyz = shoebox[..., 0:3]
        dxyz = shoebox[..., 3:6]
        counts = torch.clamp(shoebox[..., -1], min=0)
        for batch in zip(
            torch.split(xyz, batch_size, dim=0),
            torch.split(dxyz, batch_size, dim=0),
            torch.split(counts, batch_size, dim=0),
        ):
            device = next(
                self.parameters()
            ).device  # device where parameters are located
            i, s = self.get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
            I.append(i.detach().cpu().numpy())
            SigI.append(s.detach().cpu().numpy())
        I, SigI = np.concatenate(I), np.concatenate(SigI)
        return I, SigI

    def get_per_spot_normalization(self, counts):
        return torch.clamp(counts[:, :, -1], min=0).sum(-1)

    def get_intensity_sigma_batch(
        self,
        shoebox,
        mask,
        emp_bg,
        kl_lognorm_scale=None,
        bg_penalty_scaling=None,
        mc_samples=100,
    ):
        # norm_factor = self.get_per_spot_normalization(shoebox)
        # shoebox[:, :, -1] = shoebox[:, :, -1] / norm_factor.unsqueeze(-1)
        reflrep = self.refl_encoder(shoebox, mask)
        paramrep = self.param_encoder(reflrep)
        pixelrep = self.pixel_encoder(shoebox[:, :, 0:-1])
        pixelrep = self.pixel_transfomer(pixelrep, mask)
        pijrep = self.profile_model(reflrep, pixelrep)
        counts = torch.clamp(shoebox[:, :, -1], min=0)

        bg, q = self.bglognorm(paramrep)
        I, SigI = q.mean, q.stddev
        # I, SigI = I * norm_factor, SigI * norm_factor

        # ll, kl_term, bg_penalty = self.likelihood(
        # counts,
        # pijrep,
        # bg,
        # q,
        # emp_bg,
        # kl_lognorm_scale,
        # mc_samples=mc_samples,
        # kl_bern_scale=None,
        # bg_penalty_scaling=None,
        # mask=mask,
        # )
        # nll = -ll.mean()

        return I, SigI, bg, pijrep, counts

    def forward(
        self,
        shoebox,
        mask,
        emp_bg,
        kl_lognorm_scale,
        kl_bern_scale=None,
        bg_penalty_scaling=None,
        mc_samples=100,
    ):
        # norm_factor = self.get_per_spot_normalization(shoebox)
        # shoebox[..., -1] = shoebox[..., -1] / norm_factor.unsqueeze(-1)

        counts = torch.clamp(shoebox[:, :, -1], min=0)
        reflrep = self.refl_encoder(shoebox, mask)
        paramrep = self.param_encoder(reflrep)
        pixelrep = self.pixel_encoder(shoebox[:, :, 0:-1])
        pixelrep = self.pixel_transfomer(pixelrep, mask)
        pijrep = self.profile_model(reflrep, pixelrep)

        bg, q = self.bglognorm(paramrep)

        ll, kl_term, bg_penalty = self.likelihood(
            counts,
            pijrep,
            bg,
            q,
            emp_bg,
            kl_lognorm_scale,
            mc_samples=mc_samples,
            kl_bern_scale=None,
            bg_penalty_scaling=None,
            mask=mask,
        )
        nll = -ll.mean()

        return nll + kl_term

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm


class IntegratorV2(torch.nn.Module):
    """
    Attributes:
        reflencoder: Reflection encoder
        paramencoder:
        pixelencoder:
        pijencoder:
        bglognorm:
        likelihood:
        counts_std:
        counts_std:
    """

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

    def get_intensity_sigma(self, shoebox):
        I, SigI = [], []
        xyz = shoebox[..., 0:3]
        dxyz = shoebox[..., 3:6]
        counts = torch.clamp(shoebox[..., -1], min=0)
        for batch in zip(
            torch.split(xyz, batch_size, dim=0),
            torch.split(dxyz, batch_size, dim=0),
            torch.split(counts, batch_size, dim=0),
        ):
            device = next(
                self.parameters()
            ).device  # device where parameters are located
            i, s = self.get_intensity_sigma_batch(*(i.to(device=device) for i in batch))
            I.append(i.detach().cpu().numpy())
            SigI.append(s.detach().cpu().numpy())
        I, SigI = np.concatenate(I), np.concatenate(SigI)
        return I, SigI

    def get_per_spot_normalization(self, counts):
        return torch.clamp(counts[:, :, -1], min=0).sum(-1)

    def get_intensity_sigma_batch(self, shoebox, mask):
        # norm_factor = self.get_per_spot_normalization(shoebox)
        # shoebox[:, :, -1] = shoebox[:, :, -1] / norm_factor.unsqueeze(-1)
        reflrep = self.reflencoder(shoebox, mask)
        paramrep = self.paramencoder(reflrep)
        pixelrep = self.pixelencoder(shoebox[:, :, 0:-1])
        pijrep = self.pijencoder(reflrep, pixelrep)
        counts = torch.clamp(shoebox[:, :, -1], min=0)

        bg, q = self.bglognorm(paramrep)
        I, SigI = q.mean, q.stddev
        # I, SigI = I * norm_factor, SigI * norm_factor
        return I, SigI, bg, pijrep, counts

    def forward(
        self,
        shoebox,
        mask,
        emp_bg,
        kl_lognorm_scale,
        kl_bern_scale=None,
        bg_penalty_scaling=None,
        mc_samples=100,
    ):
        # norm_factor = self.get_per_spot_normalization(shoebox)
        # shoebox[..., -1] = shoebox[..., -1] / norm_factor.unsqueeze(-1)

        counts = torch.clamp(shoebox[:, :, -1], min=0)
        reflrep = self.reflencoder(shoebox, mask)
        paramrep = self.paramencoder(reflrep)
        pixelrep = self.pixelencoder(shoebox[:, :, 0:-1])
        pijrep = self.pijencoder(reflrep, pixelrep)

        bg, q = self.bglognorm(paramrep)

        # bg = bg * norm_factor.unsqueeze(1)
        # p = p * norm_factor[..., None]

        ll, kl_term, bg_penalty = self.likelihood(
            counts,
            pijrep,
            bg,
            q,
            emp_bg,
            kl_lognorm_scale,
            mc_samples=mc_samples,
            kl_bern_scale=None,
            bg_penalty_scaling=None,
            mask=mask,
        )
        nll = -ll.mean()

        return nll + kl_term

    def grad_norm(self):
        grads = [
            param.grad.detach().flatten()
            for param in self.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        return norm


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

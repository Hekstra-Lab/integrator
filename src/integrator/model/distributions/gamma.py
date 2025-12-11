import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from integrator.layers import Linear


class AttentionPoolingPerImage(nn.Module):
    """
    Attention pooling that produces one r per image.
    emb: (B, F)
    img_ids: (B,) with integer image indices (used only for grouping, not for parameters)
    """

    def __init__(
        self,
        emb_dim=64,
        hidden_dim=64,
        out_dim=1,
    ):
        super().__init__()

        # attention score per reflection: h -> scalar
        self.attn = nn.Linear(emb_dim, 1)

        # value transform for pooling: h -> hidden_dim
        self.value = nn.Linear(emb_dim, hidden_dim)

        # image-level output: hidden_dim -> r_dim
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, emb, img_ids):
        """
        emb: (B, emb_dim)
        img_ids: (B,) giving group assignments (one image id per reflection)

        Returns:
            r_reflections: (B, out_dim), r broadcast to each reflection in that image
            r_images: (N_images_in_batch, out_dim)
            unique_img_ids: (N_images_in_batch,)
        """

        unique_ids = torch.unique(img_ids)
        pooled_reprs = []
        id_to_index = {}

        # Compute per-image pooled representation
        for idx, uid in enumerate(unique_ids):
            id_to_index[uid.item()] = idx

            mask = img_ids == uid
            emb_img = emb[mask]  # reflections for this image

            # Attention weights within this image
            attn_scores = self.attn(emb_img)  # (n_reflections, 1)
            attn_weights = F.softmax(attn_scores, dim=0)

            # Values to pool
            values = self.value(emb_img)  # (n_reflections, hidden_dim)

            # Weighted sum → pooled representation
            pooled = torch.sum(attn_weights * values, dim=0)
            pooled_reprs.append(pooled)

        # Stack pooled image representations
        pooled_reprs = torch.stack(
            pooled_reprs, dim=0
        )  # (N_images_in_batch, hidden_dim)
        r_images = self.output(pooled_reprs)  # (N_images_in_batch, out_dim)

        # Broadcast r back to each reflection
        B = emb.size(0)
        r_reflections = torch.zeros((B, r_images.size(-1)), device=emb.device)

        for uid in unique_ids:
            j = id_to_index[uid.item()]
            r_reflections[img_ids == uid] = r_images[j]

        return r_reflections, r_images, unique_ids


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        out_features: int = 2,
    ):
        super().__init__()
        self.fc1 = Linear(
            in_dim,
            hidden,
            bias=False,
        )
        self.act = nn.SiLU()
        self.fc2 = Linear(
            hidden,
            out_features,
            bias=False,
        )

    def forward(self, x):
        h = self.act(self.fc1(x))
        return self.fc2(h)


class GammaDistribution(nn.Module):
    def __init__(
        self,
        estimand: Literal["background", "intensity"],
        in_features: int,
        n_images: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        if estimand == "intensity":
            self.mu_min, self.mu_max = 1e-3, 1e6
            self.fano_min, self.fano_max = 0.2, 2.0
        else:
            self.mu_min, self.mu_max = 1e-3, 100.0
            self.fano_min, self.fano_max = 0.2, 5.0

        self.log_mu_min = math.log(self.mu_min)
        self.log_mu_max = math.log(self.mu_max)
        self.log_fano_min = math.log(self.fano_min)
        self.log_fano_max = math.log(self.fano_max)

        self.mlp = MLP(in_dim=in_features, out_features=1)

        self.log_phi_table = nn.Parameter(torch.zeros(n_images))  # (n_images,)
        self.attn = AttentionPoolingPerImage()

        self.eps = eps

    def _bound(self, raw, log_min, log_max):
        return torch.exp(log_min + (log_max - log_min) * torch.sigmoid(raw))

    def forward(self, x, img_ids):
        """
        x: (batch, features)
        img_ids:(batch,) integer indices 0...n_images-1
        """

        raw_alpha = self.mlp(x)

        raw_r, _, _ = self.attn(x, img_ids)

        # mu = self._bound(raw_mu, self.log_mu_min, self.log_mu_max)  # (B,1)
        alpha = torch.nn.functional.softplus(raw_alpha) + 0.0001
        rate = torch.nn.functional.softplus(raw_alpha) + 0.0001

        # log_phi_img = self.log_phi_table[img_ids[:, 2].long()]
        # phi = torch.exp(log_phi_img).unsqueeze(-1)
        # phi = torch.clamp(phi, self.fano_min, self.fano_max)

        # beta = 1.0 / (phi + self.eps)
        # alpha = mu * beta

        # dist = Gamma(concentration=alpha.flatten(), rate=beta.flatten())
        dist = Gamma(concentration=alpha.flatten(), rate=rate.flatten())
        return dist


if __name__ == "__main__":
    import torch

    from integrator.model.distributions import (
        DirichletDistribution,
        FoldedNormalDistribution,
    )
    from integrator.model.loss import LossConfig
    from integrator.utils import (
        create_data_loader,
        create_integrator,
        load_config,
    )
    from utils import CONFIGS

    cfg = list(CONFIGS.glob("*"))[-1]
    cfg = load_config(cfg)

    integrator = create_integrator(cfg)
    data = create_data_loader(cfg)

    losscfg = LossConfig(pprf=None, pi=None, pbg=None, shape=(1, 21, 21))

    # hyperparameters
    mc_samples = 100
    shape = (1, 21, 21)

    # distributions
    qbg_ = FoldedNormalDistribution(in_features=64)
    qi_ = FoldedNormalDistribution(in_features=64)
    qp_ = DirichletDistribution(in_features=64, out_features=(1, 21, 21))

    # load a batch
    counts, sbox, mask, meta = next(iter(data.train_dataloader()))

    gamma_dist = GammaDistribution(
        in_features=in_features, n_images=1000, estimand="intensity"
    )

    x_intensity = torch.randn(100, 64)
    img_ids = torch.randint(0, 10, (100,))

    qi = gamma_dist(x_intensity, img_ids)

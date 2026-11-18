import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

OPERATIONS = {
    "2d": {
        "conv": nn.Conv2d,
        "max_pool": nn.MaxPool2d,
        "adaptive_pool": nn.AdaptiveAvgPool2d,
        "dropout": nn.Dropout2d,
    },
    "3d": {
        "conv": nn.Conv3d,
        "max_pool": nn.MaxPool3d,
        "adaptive_pool": nn.AdaptiveAvgPool3d,
        "dropout": nn.Dropout3d,
    },
}

class ProfileEncoder(nn.Module):
    """CNN encoder producing a fixed-length embedding from a shoebox volume.

    Args:
        dropout: Probability of zeroing entire channels after each block
        (using `DropoutNd`, not `Dropout`).
        output_activation: What follows the final linear layer. `relu` (the
            default) is the original encoder. Its output can be identically zero:
            on sim_geom_both every one of the 32 pre-activations was negative on
            every shoebox, so the profile latent received no gradient and could not
            recover. `none` leaves the projection linear; `layernorm` normalizes it,
            which also keeps its scale fixed. Neither has a dead state.
    """

    def __init__(
        self,
        data_dim: str,
        input_shape: tuple[int, ...] = (3, 21, 21),
        in_channels: int = 1,
        encoder_out: int = 64,
        conv1_out_channels: int = 16,
        conv1_kernel_size: tuple[int, int, int] = (1, 3, 3),
        conv1_padding: tuple[int, int, int] = (0, 1, 1),
        norm1_num_groups: int = 4,
        pool_kernel_size: tuple[int, int, int] = (1, 2, 2),
        pool_stride: tuple[int, ...] = (1, 2, 2),
        conv2_out_channels: int = 32,
        conv2_kernel_size: tuple[int, int, int] = (3, 3, 3),
        conv2_padding: tuple[int, int, int] = (0, 0, 0),
        norm2_num_groups: int = 4,
        dropout: float = 0.0,
        output_activation: str = "relu",
        head: str = "flatten",
    ):
        super().__init__()

        if head not in ("flatten", "pool", "moments"):
            raise ValueError(
                f"head must be flatten, pool or moments, got {head!r}"
            )
        self.head = head
        if output_activation not in ("relu", "none", "layernorm"):
            raise ValueError(
                "output_activation must be relu, none or layernorm, "
                f"got {output_activation!r}"
            )
        self.output_activation = output_activation
        self.encoder_out = encoder_out
        self.dropout_p = float(dropout)
        if self.dropout_p > 0:
            dropout_cls = OPERATIONS[data_dim]["dropout"]
            self.dropout1 = dropout_cls(self.dropout_p)
            self.dropout2 = dropout_cls(self.dropout_p)
        else:
            self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()

        effective_in_channels = in_channels

        self.conv1 = OPERATIONS[data_dim]["conv"](
            in_channels=effective_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=norm1_num_groups,
            num_channels=conv1_out_channels,
        )

        self.pool = OPERATIONS[data_dim]["max_pool"](
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            ceil_mode=True,
        )
        self.conv2 = OPERATIONS[data_dim]["conv"](
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=norm2_num_groups,
            num_channels=conv2_out_channels,
        )

        # `pool`: one more reduction before the dense layer, same kernel and
        # stride as the first pool
        self.head_pool = (
            OPERATIONS[data_dim]["max_pool"](
                kernel_size=pool_kernel_size, stride=pool_stride, ceil_mode=True
            )
            if head == "pool"
            else nn.Identity()
        )
        self.flattened_size = self._infer_flattened_size(
            input_shape=input_shape,
            in_channels=effective_in_channels,
        )
        self._n_moments = 6 if data_dim == "2d" else 7  # mass, centroids, spreads, xy covariance
        self.fc = torch.nn.Linear(
            in_features=(conv2_out_channels * self._n_moments
                         if head == "moments" else self.flattened_size),
            out_features=encoder_out,
        )
        # created only when used, so `relu` checkpoints keep their exact keys
        if output_activation == "layernorm":
            self.out_norm = nn.LayerNorm(encoder_out)

    def _infer_flattened_size(self, input_shape, in_channels):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            x = self.pool(F.relu(self.norm1(self.conv1(dummy))))
            x = F.relu(self.norm2(self.conv2(x)))
            return self.head_pool(x).numel()

    def _moments(self, x: Tensor) -> Tensor:
        """Mass, centroid and second moments of each feature map.

        A profile is a normalized image, so what the profile head needs from a
        feature map is where its mass sits and how it is spread -- the same
        quantities `laue.integrate` computes by hand from the counts. Taking
        them directly costs `6 * channels` inputs instead of flattening to
        23,104, and unlike a global average it keeps position. Coordinates run
        over [-1, 1] so the features do not depend on the box size.
        """
        b, ch = x.shape[0], x.shape[1]
        spatial = tuple(x.shape[2:])
        w = F.relu(x).reshape(b, ch, -1)
        tot = w.sum(-1, keepdim=True)
        p = w / tot.clamp_min(1e-6)
        axes = []
        for i, s in enumerate(spatial):
            g = torch.arange(s, device=x.device, dtype=x.dtype)
            g = 2.0 * g / max(s - 1, 1) - 1.0
            shape = [1] * len(spatial)
            shape[i] = s
            axes.append(g.reshape(shape).expand(*spatial).reshape(1, 1, -1))
        cen = [(p * a).sum(-1) for a in axes]
        feats = [tot.squeeze(-1).log1p()] + cen
        # `+ eps` inside the sqrt, not `clamp_min(0)` after: a feature map whose
        # mass sits on one pixel has zero spread, and sqrt has infinite
        # derivative there, which is a NaN in the backward pass on the first
        # such shoebox
        feats += [((p * (a - c.unsqueeze(-1)) ** 2).sum(-1) + 1e-6).sqrt()
                  for a, c in zip(axes, cen)]
        if len(axes) == 2:
            feats.append((p * (axes[0] - cen[0].unsqueeze(-1))
                          * (axes[1] - cen[1].unsqueeze(-1))).sum(-1))
        return torch.stack(feats, dim=-1).reshape(b, -1)

    def forward(self, x: Tensor, position: Tensor | None = None) -> Tensor:
        """Embed a shoebox.

        `position` is accepted and ignored. The scaling/merging integrators pass
        detector position as a separate encoder argument; this branch instead
        conditions on position by concatenating coordinate channels onto `x`
        upstream (`coordinate_channels` in `integrator_utils`, wired by the
        factory from `profile_coord_channels`), so the information arrives
        through the input. Accepting the argument lets one encoder serve both
        call conventions.
        """
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.dropout2(x)

        if self.head == "moments":
            x = self._moments(x)
        else:
            x = self.head_pool(x).reshape(x.size(0), -1)

        x = self.fc(x)
        if self.output_activation == "relu":
            x = F.relu(x)
        elif self.output_activation == "layernorm":
            x = self.out_norm(x)

        return x


class IntensityEncoder(nn.Module):
    def __init__(
        self,
        data_dim: str,
        in_channels=1,
        encoder_out=64,
        conv1_out_channels=16,
        conv1_kernel_size=(3, 3),
        conv1_padding=(1, 1),
        norm1_num_groups=4,
        pool_kernel_size=(2, 2),
        pool_stride=(2, 2),
        conv2_out_channels=32,
        conv2_kernel_size=(3, 3),
        conv2_padding=(0, 0),
        norm2_num_groups=4,
        conv3_out_channels=64,
        conv3_kernel_size=(3, 3),
        conv3_padding=(1, 1),
        norm3_num_groups=8,
    ):
        super().__init__()

        self.conv1 = OPERATIONS[data_dim]["conv"](
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            padding=conv1_padding,
        )
        self.norm1 = nn.GroupNorm(
            num_groups=norm1_num_groups,
            num_channels=conv1_out_channels,
        )

        self.pool = OPERATIONS[data_dim]["max_pool"](
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            ceil_mode=True,
        )

        self.conv2 = OPERATIONS[data_dim]["conv"](
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            padding=conv2_padding,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=norm2_num_groups,
            num_channels=conv2_out_channels,
        )

        self.conv3 = OPERATIONS[data_dim]["conv"](
            in_channels=conv2_out_channels,
            out_channels=conv3_out_channels,
            kernel_size=conv3_kernel_size,
            padding=conv3_padding,
        )
        self.norm3 = nn.GroupNorm(
            num_groups=norm3_num_groups,
            num_channels=conv3_out_channels,
        )

        self.adaptive_pool = OPERATIONS[data_dim]["adaptive_pool"](
            1
        )  # Output: (batch, channels, 1, 1)

        self.fc = torch.nn.Linear(
            in_features=conv3_out_channels,
            out_features=encoder_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.flatten(1)  # From (B, C, 1, ...) to (B, C)
        x = self.fc(x)
        x = F.relu(x)

        return x


class SummaryStatsEncoder(nn.Module):
    """Encoder over hand-built shoebox statistics, not the pixels.

    The CNN encoder spends 128,576 parameters reading the box. This reads a
    short vector of order statistics, spatial moments and detector metadata, and
    pushes it through an MLP of about a thousand parameters. It is the smallest
    thing that can still carry what the profile needs: how peaked the box is,
    where the mass sits, how elongated it is, and where on the detector it is.

    Random projections and random Fourier features were tried in place of the
    CNN and lost on every axis, most of all off-centring, where the slope
    against the true offset fell from 0.421 to 0.114. Both are built to be
    nearly invariant, so they destroy exactly the position information a
    profile needs; the moments here are the opposite choice.

    **Standardization.** Every feature is scale-free by construction rather
    than by a dataset constant, which is what lets the same encoder run on a
    dataset it was not fitted to:

    - intensity shape: quantiles and tail fractions of `c / (median + 1)`, so
      a brighter crystal or a longer exposure moves nothing
    - position and shape: first and second moments of the positive excess
      `max(c - median, 0)`, divided by the box half-width, so a 3x21x21 and a
      1x41x41 box give comparable numbers
    - detector: offset from the beam centre over the beam-centre radius, and
      the azimuth as `cos`/`sin` so it is continuous across the branch cut
    - resolution as `1/d` in inverse angstroms, which is already order one

    The one exception is the overall scale `log1p(sum c)`, which has no
    scale-free form because it *is* the scale; it gets an explicit affine whose
    defaults suit 1e2 to 1e5 counts and which is reported in the signature
    rather than fitted silently.

    Args:
        data_dim: Unused; accepted so the registry can build this like the
            other encoders.
        input_shape: Shoebox grid `(D, H, W)`, used for the moment scaling.
        encoder_out: Width of the embedding handed to the profile surrogate.
        hidden: Width of the single hidden layer.
        log_total_loc: Subtracted from `log1p(sum c)` before scaling.
        log_total_scale: Divides `log1p(sum c) - log_total_loc`.
        beam_center: Detector beam centre in pixels, for the radial features.
            Without it the four detector features are zero and the encoder sees
            the box alone.
        quantiles: Which quantiles of the normalized counts to take.
        tail_multiples: Report the fraction of pixels above each of these
            multiples of the median.
    """

    QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    TAIL_MULTIPLES = (2.0, 4.0, 8.0)

    def __init__(
        self,
        data_dim: str | None = None,
        input_shape: tuple[int, ...] = (3, 21, 21),
        encoder_out: int = 32,
        hidden: int = 16,
        log_total_loc: float = 7.0,
        log_total_scale: float = 3.0,
        beam_center: tuple[float, float] | list[float] | None = None,
        quantiles: tuple[float, ...] | list[float] | None = None,
        tail_multiples: tuple[float, ...] | list[float] | None = None,
    ):
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.encoder_out = int(encoder_out)
        self.log_total_loc = float(log_total_loc)
        self.log_total_scale = float(log_total_scale)
        self.beam_center = (
            None if beam_center is None else tuple(float(v) for v in beam_center)
        )
        self.quantiles = tuple(
            float(q) for q in (quantiles if quantiles is not None else self.QUANTILES)
        )
        self.tail_multiples = tuple(
            float(k)
            for k in (
                tail_multiples if tail_multiples is not None else self.TAIL_MULTIPLES
            )
        )

        d, h, w = self.input_shape
        zz, yy, xx = torch.meshgrid(
            torch.arange(d, dtype=torch.float32),
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        # contiguous, because meshgrid returns expanded views whose storage has
        # one element: a non-contiguous buffer saves but will not load
        # not `half`, which is nn.Module.half, the dtype cast
        extent = torch.tensor(
            [max((d - 1) / 2.0, 0.5), (h - 1) / 2.0, (w - 1) / 2.0]
        )
        self.register_buffer("half_extent", extent, persistent=False)
        for nm, g, c in (("pz", zz, (d - 1) / 2.0), ("py", yy, (h - 1) / 2.0),
                         ("px", xx, (w - 1) / 2.0)):
            self.register_buffer(nm, (g.reshape(-1) - c).contiguous(), persistent=False)

        self.n_features = (
            1                              # log total
            + len(self.quantiles)          # order statistics
            + len(self.tail_multiples)     # tail fractions
            + 3                            # first moments
            + 4                            # second moments and one cross term
            + 6                            # detector metadata
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.encoder_out),
            nn.LayerNorm(self.encoder_out),
        )

    def features(
        self, counts: Tensor, mask: Tensor | None = None, metadata: dict | None = None
    ) -> Tensor:
        """The statistics vector, shape `(B, n_features)`."""
        c = counts.reshape(counts.shape[0], -1).float()
        m = (
            torch.ones_like(c)
            if mask is None
            else mask.reshape(mask.shape[0], -1).to(c.dtype)
        )
        n = m.sum(-1, keepdim=True).clamp_min(1.0)
        tot = (c * m).sum(-1, keepdim=True).clamp_min(0.0)

        # median over VALID pixels only; counting masked pixels as zero drags it
        # down in proportion to the masked fraction
        cm = c.masked_fill(m <= 0, float("nan"))
        med = cm.nanmedian(dim=-1, keepdim=True).values
        med = torch.nan_to_num(med, nan=0.0)
        r = (c - med) / (med.abs() + 1.0)          # scale-free excess
        rq = torch.nan_to_num(
            torch.where(m > 0, r, torch.full_like(r, float("nan"))), nan=0.0
        )

        q = torch.quantile(
            rq, torch.tensor(self.quantiles, device=c.device, dtype=c.dtype), dim=-1
        ).transpose(0, 1)
        # asinh, not the raw ratio: a bright peak over a thin background gives
        # an excess near 20, and asinh is signed, linear at zero and
        # logarithmic in the tail, so one bright reflection does not dominate
        # the first layer while the weak end keeps its resolution
        q = torch.asinh(q)
        tails = torch.cat(
            [((c > k * (med + 1.0)) & (m > 0)).sum(-1, keepdim=True) / n
             for k in self.tail_multiples],
            dim=-1,
        )

        ex = (c - med).clamp_min(0.0) * m           # positive excess as weights
        wsum = ex.sum(-1, keepdim=True).clamp_min(1e-6)
        pz, py, px = self.pz[None], self.py[None], self.px[None]
        mz = (ex * pz).sum(-1, keepdim=True) / wsum
        my = (ex * py).sum(-1, keepdim=True) / wsum
        mx = (ex * px).sum(-1, keepdim=True) / wsum
        hz, hy, hx = (self.half_extent[i] for i in range(3))
        first = torch.cat([mz / hz, my / hy, mx / hx], dim=-1)
        vz = (ex * (pz - mz) ** 2).sum(-1, keepdim=True) / wsum / hz**2
        vy = (ex * (py - my) ** 2).sum(-1, keepdim=True) / wsum / hy**2
        vx = (ex * (px - mx) ** 2).sum(-1, keepdim=True) / wsum / hx**2
        cyx = (ex * (py - my) * (px - mx)).sum(-1, keepdim=True) / wsum / (hy * hx)
        second = torch.cat([vz.sqrt(), vy.sqrt(), vx.sqrt(), cyx], dim=-1)

        meta = torch.zeros(c.shape[0], 6, device=c.device, dtype=c.dtype)
        if metadata is not None and self.beam_center is not None:
            x0, y0 = self.beam_center
            if "xyzcal.px.0" in metadata and "xyzcal.px.1" in metadata:
                dx = metadata["xyzcal.px.0"].reshape(-1).to(c) - x0
                dy = metadata["xyzcal.px.1"].reshape(-1).to(c) - y0
                rad = torch.hypot(dx, dy).clamp_min(1e-6)
                scale = math.hypot(x0, y0) or 1.0
                meta[:, 0] = rad / scale
                meta[:, 1] = dx / rad
                meta[:, 2] = dy / rad
                # the two-fold part, so an axis-like dependence is continuous
                meta[:, 3] = (dx**2 - dy**2) / rad**2
            if "resolution" in metadata:
                d_sp = metadata["resolution"].reshape(-1).to(c).clamp_min(1e-3)
                meta[:, 4] = 1.0 / d_sp
            if "wavelength" in metadata:
                lam = metadata["wavelength"].reshape(-1).to(c)
                meta[:, 5] = lam / lam.mean().clamp_min(1e-6) - 1.0

        log_tot = (torch.log1p(tot) - self.log_total_loc) / self.log_total_scale
        return torch.cat([log_tot, q, tails, first, second, meta], dim=-1)

    def forward(self, counts: Tensor, mask: Tensor | None = None,
                metadata: dict | None = None, **kwargs) -> Tensor:
        return self.mlp(self.features(counts, mask=mask, metadata=metadata))

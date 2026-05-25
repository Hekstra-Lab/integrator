from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

import gemmi
import numpy as np

import gemmi

if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )

from integrator import configs
from integrator.model.integrators.base_integrator import (
    BaseIntegrator,
    _log_loss,
)
from integrator.model.integrators.hierarchical_integrator import (
    _add_group_outputs,
    _get_normalized_position,
    _sample_profile,
)
from integrator.model.integrators.integrator_utils import (
    IntegratorBaseOutputs,
    _assemble_outputs,
)
from integrator.model.scaling.chebyshev_scale import ChebyshevScale


class DeterministicIntensity:
    """Duck-typed wrapper around deterministic F^2 that satisfies the
    interface expected by ``_assemble_outputs`` and predict_step."""

    def __init__(self, F_sq: Tensor):
        self.F_sq = F_sq

    @property
    def mean(self) -> Tensor:
        return self.F_sq

    @property
    def variance(self) -> Tensor:
        return torch.zeros_like(self.F_sq)

    @property
    def arg_constraints(self) -> dict:
        return {}


def _build_hasu_lookup(
    Hasu_array: np.ndarray, sg: gemmi.SpaceGroup, anomalous: bool = True
) -> dict[tuple[int, int, int], int]:
    """Build a hash map from canonical (H,K,L) → index in Hasu_array.

    Always includes Friedel mates because SFcalculator's Hasu_array
    contains only one member per Friedel pair even with anomalous=True.
    Both F(+) and F(-) map to the same index; the anomalous flag is
    preserved in the data's asu_id for downstream use.
    """
    op_list = list(sg.operations())
    lookup: dict[tuple[int, int, int], int] = {}
    for idx in range(len(Hasu_array)):
        h, k, l = int(Hasu_array[idx, 0]), int(Hasu_array[idx, 1]), int(Hasu_array[idx, 2])
        for op in op_list:
            hkl_rot = op.apply_to_hkl([h, k, l])
            lookup[tuple(hkl_rot)] = idx
            lookup[(-hkl_rot[0], -hkl_rot[1], -hkl_rot[2])] = idx
    return lookup


def _build_hasu_lookup_anomalous(
    Hasu_array: np.ndarray, sg: gemmi.SpaceGroup
) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, int], bool]]:
    """Build lookup that distinguishes Friedel mates.

    Returns:
        idx_lookup: HKL → index in Hasu_array (same as _build_hasu_lookup)
        is_plus: HKL → True if this HKL matches Hasu_array directly,
                 False if it matches only via Friedel negation.
                 F(+) uses Fc_plus[idx], F(-) uses Fc_minus[idx].
    """
    op_list = list(sg.operations())
    idx_lookup: dict[tuple[int, int, int], int] = {}
    is_plus: dict[tuple[int, int, int], bool] = {}
    for idx in range(len(Hasu_array)):
        h, k, l = int(Hasu_array[idx, 0]), int(Hasu_array[idx, 1]), int(Hasu_array[idx, 2])
        for op in op_list:
            hkl_rot = op.apply_to_hkl([h, k, l])
            key_plus = tuple(hkl_rot)
            key_minus = (-hkl_rot[0], -hkl_rot[1], -hkl_rot[2])
            if key_plus not in idx_lookup:
                idx_lookup[key_plus] = idx
                is_plus[key_plus] = True
            if key_minus not in idx_lookup:
                idx_lookup[key_minus] = idx
                is_plus[key_minus] = False
    return idx_lookup, is_plus


class RefinementIntegrator(BaseIntegrator):
    """End-to-end integrator that refines an atomic model against raw pixel counts.

    Replaces the per-HKL lookup table with a differentiable structure
    factor calculation via SFC_Torch.  Atomic coordinates and B-factors
    are nn.Parameters optimized jointly with profile, background, and
    scale through the pixel-level Poisson ELBO.

    rate = s(frame)/lp × |F_calc(atoms)|² × profile + background

    Requires SFC_Torch to be installed.
    """

    REQUIRED_ENCODERS = {
        "profile": configs.ProfileEncoderArgs,
        "k_bg": configs.IntensityEncoderArgs,
        "r_bg": configs.IntensityEncoderArgs,
    }

    def __init__(
        self,
        cfg: configs.IntegratorCfg,
        loss: nn.Module,
        encoders: dict[str, nn.Module],
        surrogates: dict[str, nn.Module],
    ):
        super().__init__(cfg, loss, encoders, surrogates)

        if cfg.pdb_path is None:
            raise ValueError("RefinementIntegrator requires pdb_path.")
        if cfg.asu_id_to_hkl_path is None:
            raise ValueError("RefinementIntegrator requires asu_id_to_hkl_path.")

        from SFC_Torch import SFcalculator as SFC

        self.sfcalc = SFC(
            pdbmodel=cfg.pdb_path,
            dmin=cfg.dmin,
            anomalous=cfg.anomalous,
            wavelength=cfg.wavelength,
        )
        self.sfcalc.inspect_data()

        self.atom_pos = nn.Parameter(self.sfcalc.atom_pos_orth.clone())
        self.atom_b_iso = nn.Parameter(self.sfcalc.atom_b_iso.clone())

        self.register_buffer("atom_pos_init", self.sfcalc.atom_pos_orth.clone())
        self.register_buffer("atom_b_iso_init", self.sfcalc.atom_b_iso.clone())

        self.restraint_w_xyz = cfg.restraint_w_xyz
        self.restraint_w_biso = cfg.restraint_w_biso
        self.atom_lr = cfg.atom_lr if cfg.atom_lr is not None else cfg.lr

        self.scale_fn = ChebyshevScale(
            degree=cfg.scale_degree,
            frame_min=cfg.scale_frame_min,
            frame_max=cfg.scale_frame_max,
        )

        self._build_hkl_map(cfg)

    def _build_hkl_map(self, cfg: configs.IntegratorCfg) -> None:
        """Build mapping from our asu_id → SFcalculator's Hasu index."""
        id_to_hkl = torch.load(
            cfg.asu_id_to_hkl_path, weights_only=False, map_location="cpu"
        )
        n_asu_ids = len(id_to_hkl)

        hasu_lookup = _build_hasu_lookup(
            self.sfcalc.Hasu_array, self.sfcalc.space_group, cfg.anomalous
        )

        sfc_idx = torch.full((n_asu_ids,), -1, dtype=torch.long)
        for aid in range(n_asu_ids):
            h, k, l = int(id_to_hkl[aid, 0]), int(id_to_hkl[aid, 1]), int(id_to_hkl[aid, 2])
            if (h, k, l) in hasu_lookup:
                sfc_idx[aid] = hasu_lookup[(h, k, l)]

        n_mapped = (sfc_idx >= 0).sum().item()
        n_missing = n_asu_ids - n_mapped
        if n_missing > 0:
            import logging
            logging.getLogger(__name__).warning(
                "%d of %d asu_ids could not be mapped to SFcalculator HKLs "
                "(likely beyond dmin=%.1f). These will get F_sq=0.",
                n_missing, n_asu_ids, cfg.dmin,
            )
            sfc_idx[sfc_idx < 0] = 0

        self.register_buffer("sfc_idx", sfc_idx)
        self.register_buffer(
            "sfc_valid", torch.tensor([1 if s >= 0 else 0 for s in sfc_idx.tolist()], dtype=torch.bool)
        )

    def _compute_F_sq(self) -> Tensor:
        """Compute |F_hkl|² for all ASU HKLs from the atomic model."""
        self.sfcalc.atom_pos_orth = self.atom_pos
        self.sfcalc.atom_b_iso = self.atom_b_iso
        F = self.sfcalc.calc_fprotein(Return=True)
        return (F * F.conj()).real

    def _forward_impl(
        self,
        counts: Tensor,
        shoebox: Tensor,
        mask: Tensor,
        metadata: dict,
    ) -> dict[str, Any]:
        counts = torch.clamp(counts, min=0)

        b = shoebox.shape[0]
        device = shoebox.device
        shoebox_masked = shoebox * mask
        shoebox_reshaped = shoebox_masked.reshape(b, 1, *self.shoebox_shape)

        # Profile
        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )

        # Background
        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)

        # Profile surrogate
        prf_labels = metadata.get(
            "profile_group_label", metadata.get("group_label")
        )
        prf_labels = prf_labels.long() if prf_labels is not None else None
        qp = self.surrogates["qp"](
            x_profile,
            mc_samples=self.mc_samples,
            group_labels=prf_labels,
            metadata=metadata,
        )

        # Structure factor from atomic model (deterministic)
        F_sq_all = self._compute_F_sq()
        asu_ids = metadata["asu_id"].long().to(device)
        F_sq = F_sq_all[self.sfc_idx[asu_ids]]
        F_sq = F_sq.view(b, 1, 1)

        # MC samples for profile and background
        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)

        # Per-observation scale: s(frame) / lp
        frame = metadata["xyzcal.px.2"].to(device).float()
        s = self.scale_fn(frame)
        lp = metadata["lp"].to(device).float().clamp(min=1e-8)
        scale = (s / lp).view(b, 1, 1)

        rate = scale * F_sq * zp + zbg

        if "is_coset" in metadata:
            coset = metadata["is_coset"].bool().view(-1, 1, 1)
            rate = torch.where(coset, zbg, rate)

        qi = DeterministicIntensity(F_sq.squeeze())

        out = IntegratorBaseOutputs(
            rates=rate,
            counts=counts,
            mask=mask,
            qbg=qbg,
            qp=qp,
            qi=qi,
            zp=zp,
            zbg=zbg,
            metadata=metadata,
        )
        out = _assemble_outputs(out)
        out["asu_id"] = asu_ids
        _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

    def _geometry_restraint(self) -> Tensor:
        n_atoms = self.atom_pos.shape[0]
        xyz_penalty = (
            (self.atom_pos - self.atom_pos_init).pow(2).sum() / n_atoms
            * self.restraint_w_xyz
        )
        b_penalty = (
            (self.atom_b_iso - self.atom_b_iso_init).pow(2).sum() / n_atoms
            * self.restraint_w_biso
        )
        return xyz_penalty + b_penalty

    def _step(self, batch, step: Literal["train", "val"]):
        counts, shoebox, mask, metadata = batch
        outputs = self(counts, shoebox, mask, metadata)
        forward_out = outputs["forward_out"]

        group_labels = metadata["group_label"].long()

        loss_dict = self.loss(
            rate=forward_out["rates"],
            counts=forward_out["counts"],
            qp=outputs["qp"],
            qi=outputs["qi"],
            qbg=outputs["qbg"],
            mask=forward_out["mask"],
            group_labels=group_labels,
            metadata=metadata,
        )

        total_loss = loss_dict["loss"]

        _log_loss(
            self,
            kl=loss_dict["kl_mean"],
            nll=loss_dict["neg_ll_mean"],
            total_loss=total_loss,
            step=step,
            kl_components={
                k.removesuffix("_mean"): v
                for k, v in loss_dict.items()
                if k in ("kl_prf_mean", "kl_i_mean", "kl_bg_mean")
            },
        )

        # Profile basis penalty
        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        # Geometry restraints
        restraint = self._geometry_restraint()
        self.log(f"{step} restraint", restraint.detach(), on_step=False, on_epoch=True)
        total_loss = total_loss + restraint

        return {
            "loss": total_loss,
            "forward_out": forward_out,
            "loss_components": {
                "loss": total_loss.detach(),
                "nll": loss_dict["neg_ll_mean"].detach(),
                "kl": loss_dict["kl_mean"].detach(),
                "kl_prf": loss_dict["kl_prf_mean"].detach(),
                "kl_i": loss_dict["kl_i_mean"].detach(),
                "kl_bg": loss_dict["kl_bg_mean"].detach(),
            },
        }

    def _build_optimizer(self) -> torch.optim.Optimizer:
        atom_params = [self.atom_pos, self.atom_b_iso]
        atom_ids = {id(p) for p in atom_params}
        other_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in atom_ids
        ]
        return torch.optim.Adam(
            [
                {"params": other_params, "weight_decay": self.weight_decay},
                {"params": atom_params, "lr": self.atom_lr, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

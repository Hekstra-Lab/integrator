import logging
import math
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
from integrator.model.scaling.chebyshev_scale import (
    ChebyshevScale,
    SpatialChebyshevScale,
)
from integrator.model.scaling.refinement_integrator import (
    DeterministicIntensity,
)

logger = logging.getLogger(__name__)

EIGHT_PI_SQ = 8.0 * math.pi**2


def _softplus_inverse(x: Tensor) -> Tensor:
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))


class VariationalRefinementIntegrator(BaseIntegrator):
    """Refinement integrator with variational inference over atomic positions.

    q(x_j) = N(mu_j, sigma_j^2 I),  B_j = 8 pi^2 sigma_j^2

    The Debye-Waller factor is the analytic Fourier average over the
    position posterior, so F_calc is deterministic given (mu, sigma) and
    no MC sampling over atom positions is needed.

    The isotropic Gaussian KL replaces the ad-hoc L2 geometry restraint:
      KL = sum_j [3 log(s0/s) + (3 s^2 + ||mu-mu0||^2)/(2 s0^2) - 3/2]

    Optional bulk solvent via the flat mask model (Jiang & Brunger 1994):
      F_total = F_protein + k_sol * exp(-B_sol * s^2) * F_mask
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
            raise ValueError(
                "VariationalRefinementIntegrator requires pdb_path."
            )
        if cfg.asu_id_to_hkl_path is None:
            raise ValueError(
                "VariationalRefinementIntegrator requires asu_id_to_hkl_path."
            )

        from SFC_Torch import SFcalculator as SFC

        self.sfcalc = SFC(
            pdbmodel=cfg.pdb_path,
            dmin=cfg.dmin,
            anomalous=cfg.anomalous,
            wavelength=cfg.wavelength,
        )
        self.sfcalc.inspect_data()

        pos_init = self.sfcalc.atom_pos_orth.clone()
        b_init = self.sfcalc.atom_b_iso.clone().clamp(min=0.5)

        self.atom_pos_mu = nn.Parameter(pos_init)

        sigma_init = (b_init / EIGHT_PI_SQ).sqrt()
        self.atom_raw_log_sigma = nn.Parameter(_softplus_inverse(sigma_init))

        self.register_buffer("atom_pos_prior_mu", pos_init.clone())

        if cfg.atom_sigma_prior is not None:
            sigma_prior = torch.full_like(b_init, cfg.atom_sigma_prior)
        else:
            sigma_prior = sigma_init.clone()
        self.register_buffer("atom_sigma_prior", sigma_prior)

        self.kl_atom_weight = cfg.kl_atom_weight
        self.atom_lr = cfg.atom_lr if cfg.atom_lr is not None else cfg.lr

        if cfg.scale_spatial:
            self.scale_fn = SpatialChebyshevScale(
                degree_frame=cfg.scale_degree,
                degree_radius=cfg.scale_degree_radius,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
                beam_center=cfg.scale_beam_center,
                r_min=cfg.scale_r_min,
                r_max=cfg.scale_r_max,
            )
        else:
            self.scale_fn = ChebyshevScale(
                degree=cfg.scale_degree,
                frame_min=cfg.scale_frame_min,
                frame_max=cfg.scale_frame_max,
            )

        self.use_bulk_solvent = cfg.bulk_solvent
        if self.use_bulk_solvent:
            self._init_bulk_solvent(cfg)

        self.use_geometry_restraints = cfg.geometry_restraints
        if self.use_geometry_restraints:
            self._init_geometry_restraints(cfg)

        self._build_hkl_map(cfg)

    @property
    def atom_sigma(self) -> Tensor:
        return F.softplus(self.atom_raw_log_sigma) + 1e-6

    @property
    def atom_b_iso(self) -> Tensor:
        return EIGHT_PI_SQ * self.atom_sigma.pow(2)

    # ------------------------------------------------------------------
    # Bulk solvent
    # ------------------------------------------------------------------

    def _init_bulk_solvent(self, cfg: configs.IntegratorCfg) -> None:
        """Compute initial solvent mask and set up k_sol / B_sol parameters."""
        self.sfcalc.calc_fprotein()
        self.sfcalc.calc_fsolvent()

        self.register_buffer(
            "F_mask", self.sfcalc.Fmask_asu.detach().clone()
        )

        d_hkl = torch.from_numpy(self.sfcalc.dHasu).float()
        s_squared = 1.0 / (4.0 * d_hkl.pow(2))
        self.register_buffer("s_squared", s_squared)

        self.raw_k_sol = nn.Parameter(
            _softplus_inverse(torch.tensor(cfg.k_sol_init))
        )
        self.raw_B_sol = nn.Parameter(
            _softplus_inverse(torch.tensor(cfg.B_sol_init))
        )

        logger.info(
            "Bulk solvent enabled: solvent_pct=%.1f%%, %d ASU HKLs, "
            "k_sol_init=%.3f, B_sol_init=%.1f",
            self.sfcalc.solventpct * 100,
            len(self.F_mask),
            cfg.k_sol_init,
            cfg.B_sol_init,
        )

    @property
    def k_sol(self) -> Tensor:
        return F.softplus(self.raw_k_sol)

    @property
    def B_sol(self) -> Tensor:
        return F.softplus(self.raw_B_sol)

    def update_solvent_mask(self) -> None:
        """Recompute solvent mask from current atomic positions."""
        with torch.no_grad():
            self.sfcalc.atom_pos_orth = self.atom_pos_mu.detach()
            self.sfcalc.atom_b_iso = self.atom_b_iso.detach()
            self.sfcalc.calc_fprotein()
            self.sfcalc.calc_fsolvent()
            self.F_mask.copy_(self.sfcalc.Fmask_asu.detach())

    # ------------------------------------------------------------------
    # Geometry restraints
    # ------------------------------------------------------------------

    def _init_geometry_restraints(self, cfg: configs.IntegratorCfg) -> None:
        """Build bond/angle restraint tensors from gemmi topology."""
        pdb_path = cfg.pdb_path
        st = gemmi.read_structure(pdb_path)
        st.setup_entities()

        topo = gemmi.prepare_topology(st, gemmi.MonLib())

        # Build atom → index map (same ordering as SFcalculator)
        all_atoms = []
        for model in st:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        all_atoms.append(atom)
        atom_id_to_idx = {id(a): i for i, a in enumerate(all_atoms)}

        # Bonds
        bond_i, bond_j, bond_ideal, bond_sigma = [], [], [], []
        for bond in topo.bonds:
            a1, a2 = bond.atoms
            i1 = atom_id_to_idx.get(id(a1), -1)
            i2 = atom_id_to_idx.get(id(a2), -1)
            if i1 >= 0 and i2 >= 0 and bond.restr.esd > 0:
                bond_i.append(i1)
                bond_j.append(i2)
                bond_ideal.append(bond.restr.value)
                bond_sigma.append(bond.restr.esd)

        self.register_buffer(
            "bond_idx",
            torch.tensor([bond_i, bond_j], dtype=torch.long).T,
        )
        self.register_buffer("bond_ideal", torch.tensor(bond_ideal))
        self.register_buffer("bond_sigma", torch.tensor(bond_sigma))
        self.geometry_w_bond = cfg.geometry_w_bond

        # Angles
        ang_i, ang_j, ang_k, ang_ideal, ang_sigma = [], [], [], [], []
        for angle in topo.angles:
            a1, a2, a3 = angle.atoms
            i1 = atom_id_to_idx.get(id(a1), -1)
            i2 = atom_id_to_idx.get(id(a2), -1)
            i3 = atom_id_to_idx.get(id(a3), -1)
            if i1 >= 0 and i2 >= 0 and i3 >= 0 and angle.restr.esd > 0:
                ang_i.append(i1)
                ang_j.append(i2)
                ang_k.append(i3)
                ang_ideal.append(math.radians(angle.restr.value))
                ang_sigma.append(math.radians(angle.restr.esd))

        self.register_buffer(
            "angle_idx",
            torch.tensor([ang_i, ang_j, ang_k], dtype=torch.long).T,
        )
        self.register_buffer("angle_ideal", torch.tensor(ang_ideal))
        self.register_buffer("angle_sigma", torch.tensor(ang_sigma))
        self.geometry_w_angle = cfg.geometry_w_angle

        logger.info(
            "Geometry restraints: %d bonds, %d angles",
            len(bond_ideal), len(ang_ideal),
        )

    def _geometry_penalty(self) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute bond length and angle penalties from monomer library ideals."""
        pos = self.atom_pos_mu
        components: dict[str, Tensor] = {}

        # Bond penalty: Σ ((d - d_ideal) / sigma)²
        p1 = pos[self.bond_idx[:, 0]]
        p2 = pos[self.bond_idx[:, 1]]
        d = (p1 - p2).norm(dim=1)
        bond_z = (d - self.bond_ideal) / self.bond_sigma
        bond_loss = bond_z.pow(2).mean() * self.geometry_w_bond
        components["bond_rmsd"] = (d - self.bond_ideal).pow(2).mean().sqrt()
        components["bond_loss"] = bond_loss.detach()

        # Angle penalty: Σ ((θ - θ_ideal) / sigma)²
        p1 = pos[self.angle_idx[:, 0]]
        p2 = pos[self.angle_idx[:, 1]]
        p3 = pos[self.angle_idx[:, 2]]
        v1 = p1 - p2
        v2 = p3 - p2
        cos_theta = (v1 * v2).sum(dim=1) / (
            v1.norm(dim=1) * v2.norm(dim=1) + 1e-8
        )
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        angle_z = (theta - self.angle_ideal) / self.angle_sigma
        angle_loss = angle_z.pow(2).mean() * self.geometry_w_angle
        components["angle_rmsd"] = (
            (theta - self.angle_ideal).pow(2).mean().sqrt()
            * (180.0 / math.pi)
        )
        components["angle_loss"] = angle_loss.detach()

        return bond_loss + angle_loss, components

    # ------------------------------------------------------------------

    def _build_hkl_map(self, cfg: configs.IntegratorCfg) -> None:
        """Build asu_id → Hasu_array index mapping.

        Since Hasu_array with anomalous=True already contains both
        Friedel mates as separate entries, each asu_id should map to
        exactly one index.  We build a symmetry-only lookup (no Friedel
        negation) so that plus and minus asu_ids map to their correct,
        distinct Hasu entries.
        """
        id_to_hkl = torch.load(
            cfg.asu_id_to_hkl_path, weights_only=False, map_location="cpu"
        )
        n_asu_ids = len(id_to_hkl)

        # Build lookup from Hasu_array using ONLY symmetry operations
        # (no Friedel negation — both mates are already in Hasu_array)
        op_list = list(self.sfcalc.space_group.operations())
        hasu_lookup: dict[tuple[int, int, int], int] = {}
        for idx in range(len(self.sfcalc.Hasu_array)):
            h, k, l = (
                int(self.sfcalc.Hasu_array[idx, 0]),
                int(self.sfcalc.Hasu_array[idx, 1]),
                int(self.sfcalc.Hasu_array[idx, 2]),
            )
            for op in op_list:
                hkl_rot = op.apply_to_hkl([h, k, l])
                hasu_lookup[tuple(hkl_rot)] = idx

        sfc_idx = torch.full((n_asu_ids,), -1, dtype=torch.long)
        for aid in range(n_asu_ids):
            h, k, l = (
                int(id_to_hkl[aid, 0]),
                int(id_to_hkl[aid, 1]),
                int(id_to_hkl[aid, 2]),
            )
            if (h, k, l) in hasu_lookup:
                sfc_idx[aid] = hasu_lookup[(h, k, l)]

        n_mapped = (sfc_idx >= 0).sum().item()
        n_missing = n_asu_ids - n_mapped
        if n_missing > 0:
            logger.warning(
                "%d of %d asu_ids could not be mapped to SFcalculator HKLs "
                "(likely beyond dmin=%.1f or canonicalization mismatch). "
                "These will get F_sq=0.",
                n_missing,
                n_asu_ids,
                cfg.dmin,
            )
            sfc_idx[sfc_idx < 0] = 0

        logger.info(
            "HKL map: %d mapped, %d unmapped, %d Hasu entries",
            n_mapped, n_missing, len(self.sfcalc.Hasu_array),
        )

        self.register_buffer("sfc_idx", sfc_idx)

    def _compute_F_sq(self) -> Tensor:
        """Compute |F|² for all ASU HKLs.

        With anomalous=True, Hasu_array already contains both Friedel
        mates as separate entries, so calc_fprotein returns different
        F values for (h,k,l) and (-h,-k,-l) due to f''.
        """
        self.sfcalc.atom_pos_orth = self.atom_pos_mu
        self.sfcalc.atom_b_iso = self.atom_b_iso
        Fc = self.sfcalc.calc_fprotein(Return=True)

        if self.use_bulk_solvent:
            dampening = torch.exp(-self.B_sol * self.s_squared)
            F_solvent = self.k_sol * dampening * self.F_mask
            Fc = Fc + F_solvent

        return (Fc * Fc.conj()).real

    def _atom_position_kl(self) -> Tensor:
        """KL(q(x) || p(x)) for isotropic Gaussian position posteriors.

        q(x_j) = N(mu_j, sigma_j^2 I_3)
        p(x_j) = N(mu0_j, sigma0_j^2 I_3)

        Per-atom KL in 3D:
          3 log(s0/s) + (3 s^2 + ||mu - mu0||^2) / (2 s0^2) - 3/2
        """
        sigma = self.atom_sigma
        sigma0 = self.atom_sigma_prior
        mu_diff_sq = (self.atom_pos_mu - self.atom_pos_prior_mu).pow(2).sum(
            dim=1
        )

        kl_per_atom = (
            3.0 * torch.log(sigma0 / sigma)
            + (3.0 * sigma.pow(2) + mu_diff_sq) / (2.0 * sigma0.pow(2))
            - 1.5
        )
        return kl_per_atom.sum() * self.kl_atom_weight

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

        position = _get_normalized_position(metadata, device)
        x_profile = self.encoders["profile"](
            shoebox_reshaped, position=position
        )

        x_k_bg = self.encoders["k_bg"](shoebox_reshaped)
        x_r_bg = self.encoders["r_bg"](shoebox_reshaped)
        qbg = self.surrogates["qbg"](x_k_bg, x_r_bg)

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

        F_sq_all = self._compute_F_sq()
        asu_ids = metadata["asu_id"].long().to(device)
        F_sq = F_sq_all[self.sfc_idx[asu_ids]]
        F_sq = F_sq.view(b, 1, 1)

        zbg = qbg.rsample([self.mc_samples]).unsqueeze(-1).permute(1, 0, 2)
        zp = _sample_profile(qp, self.mc_samples)

        frame = metadata["xyzcal.px.2"].to(device).float()
        if isinstance(self.scale_fn, SpatialChebyshevScale):
            x_det = metadata["xyzcal.px.0"].to(device).float()
            y_det = metadata["xyzcal.px.1"].to(device).float()
            s = self.scale_fn(frame, x_det, y_det)
        else:
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
        if "group_label" in metadata:
            _add_group_outputs(out, metadata, self.loss)

        return {
            "forward_out": out,
            "qp": qp,
            "qi": qi,
            "qbg": qbg,
        }

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

        if "neg_ll_free" in loss_dict:
            self.log(
                f"{step} nll_rfree",
                loss_dict["neg_ll_free"].detach(),
                on_step=False,
                on_epoch=True,
            )

        penalty, penalty_components = self._profile_basis_penalty()
        for name, value in penalty_components.items():
            self.log(f"{step} {name}", value, on_step=False, on_epoch=True)
        total_loss = total_loss + penalty

        kl_atoms = self._atom_position_kl()
        self.log(
            f"{step} kl_atoms", kl_atoms.detach(), on_step=False, on_epoch=True
        )
        total_loss = total_loss + kl_atoms

        if self.use_geometry_restraints:
            geom_penalty, geom_components = self._geometry_penalty()
            for name, value in geom_components.items():
                self.log(
                    f"{step} {name}", value, on_step=False, on_epoch=True
                )
            total_loss = total_loss + geom_penalty

        # Log derived quantities for monitoring
        with torch.no_grad():
            b_iso = self.atom_b_iso
            self.log(
                f"{step} B_mean",
                b_iso.mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} B_min",
                b_iso.min(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step} B_max",
                b_iso.max(),
                on_step=False,
                on_epoch=True,
            )
            pos_shift = (
                (self.atom_pos_mu - self.atom_pos_prior_mu)
                .pow(2)
                .sum(dim=1)
                .sqrt()
            )
            self.log(
                f"{step} pos_rmsd",
                pos_shift.mean(),
                on_step=False,
                on_epoch=True,
            )
            if self.use_bulk_solvent:
                self.log(
                    f"{step} k_sol",
                    self.k_sol,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    f"{step} B_sol",
                    self.B_sol,
                    on_step=False,
                    on_epoch=True,
                )

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
        atom_params = [self.atom_pos_mu, self.atom_raw_log_sigma]
        atom_ids = {id(p) for p in atom_params}
        other_params = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in atom_ids
        ]
        return torch.optim.Adam(
            [
                {"params": other_params, "weight_decay": self.weight_decay},
                {
                    "params": atom_params,
                    "lr": self.atom_lr,
                    "weight_decay": 0.0,
                },
            ],
            lr=self.lr,
        )

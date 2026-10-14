from __future__ import annotations

"""Area-aware, physics-aware losses shared by FNO, GNO and GINO.

The primary objective is identical for all six controlled workflows.  The loss
combines normalized field error with conservative-volume, wet/dry support,
dry-cell suppression and admissibility terms.  The wet/dry term is explicitly
area weighted and class balanced; unlike ``binary_cross_entropy(...,
weight=area/area.sum())`` with the default mean reduction, it does not acquire
an accidental extra 1/N factor.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

HF, HSU, HCU, HSL, HCL, MX, MY, DZB = range(8)


@dataclass
class LossConfig:
    # Data term: target-wet cells receive additional weight, but dry cells are
    # retained so false-positive halos remain penalized.
    lambda_early_front: float = 0.0
    early_front_hard_fraction: float = 0.02
    wet_weight: float = 8.0

    # Wet/dry support uses the solver dry threshold (1 mm in this study).
    h_wet_m: float = 1.0e-3
    wet_temperature_m: float = 2.0e-4
    max_support_class_weight: float = 50.0

    # Physical regularizers.  These defaults are intentionally moderate and
    # are common to all six FNO/GNO/GINO Cartesian and adaptive workflows.
    lambda_mass: float = 0.05
    lambda_wet: float = 0.20
    lambda_dry_depth: float = 0.05
    lambda_dry_momentum: float = 0.02
    lambda_admissible: float = 0.05

    # A target-dry cell may contain a tiny spectral/numerical ripple without
    # being penalized until it exceeds this tolerance.
    dry_depth_tolerance_m: float = 2.5e-4

    # speed of the bulk mixture is MOMENTUM / MASS PER AREA, not q/h.
    # This term complements (does not replace) normalized conservative MSE.
    lambda_velocity: float = 0.05
    lambda_momentum_consistency: float = 0.02
    lambda_admissible_l1: float = 0.02
    velocity_min_depth_m: float = 0.05
    velocity_scale_ms: float = 2.0
    velocity_huber_delta_ms: float = 1.0
    rho_f_kg_m3: float = 1000.0
    rho_s_kg_m3: float = 2650.0


def _spatial_dims(y: torch.Tensor):
    # FNO: B,C,H,W. GNO: B,C,N.
    return tuple(range(2, y.ndim))


def _sample_spatial_dims(y: torch.Tensor):
    # Tensors without channel dimension: B,H,W or B,N.
    return tuple(range(1, y.ndim))


def _weighted_mean(v: torch.Tensor, w: torch.Tensor, dims) -> torch.Tensor:
    """Return per-sample/channel weighted mean over spatial dimensions."""
    while w.ndim < v.ndim:
        w = w.unsqueeze(1)
    num = (v * w).sum(dim=dims)
    den = w.sum(dim=dims).clamp_min(1.0e-12)
    return num / den


def _area_mean(v: torch.Tensor, area: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Area-weighted mean of a B,*space tensor, averaged across the batch."""
    a = area.to(v.device, dtype=v.dtype)
    if mask is not None:
        a = a * mask.to(v.device, dtype=v.dtype)
    dims = _sample_spatial_dims(v)
    num = (v * a).sum(dim=dims)
    den = a.sum(dim=dims)
    # Samples with no cells in the requested class contribute zero rather than
    # NaN; this matters for all-dry/all-wet edge cases.
    mean = torch.where(den > 0, num / den.clamp_min(1.0e-12), torch.zeros_like(num))
    return mean.mean()


def _project_phases_torch(y: torch.Tensor):
    """Differentiable admissible projection of the five thickness channels."""
    hf = torch.relu(y[:, HF])
    hsu = torch.relu(y[:, HSU])
    hsl = torch.relu(y[:, HSL])
    hcu = torch.minimum(torch.relu(y[:, HCU]), hsu)
    hcl = torch.minimum(torch.relu(y[:, HCL]), hsl)
    return hf, hsu, hcu, hsl, hcl


def _balanced_support_bce(
    occ: torch.Tensor,
    wet: torch.Tensor,
    area: torch.Tensor,
    max_class_weight: float,
) -> torch.Tensor:
    """Area-weighted BCE with equal wet/dry class mass per sample.

    The old implementation normalized area and then used BCE's default mean
    reduction, introducing an additional division by the number of cells.  On
    large rasters this made the support penalty almost ineffective.  Here the
    integral is performed explicitly.
    """
    eps = 1.0e-6
    p = occ.clamp(eps, 1.0 - eps)
    bce = -(wet * torch.log(p) + (1.0 - wet) * torch.log(1.0 - p))
    a = area.to(p.device, dtype=p.dtype)
    dims = _sample_spatial_dims(p)
    total = a.sum(dim=dims).clamp_min(1.0e-12)
    aw = (a * wet).sum(dim=dims)
    ad = (a * (1.0 - wet)).sum(dim=dims)

    # If both classes are present, each receives ~50% of the total class mass.
    # Missing classes are harmless because their masks are zero.
    pos_w = (0.5 * total / aw.clamp_min(1.0e-12)).clamp(max=float(max_class_weight))
    neg_w = (0.5 * total / ad.clamp_min(1.0e-12)).clamp(max=float(max_class_weight))
    view = (-1,) + (1,) * (p.ndim - 1)
    class_w = wet * pos_w.view(view) + (1.0 - wet) * neg_w.view(view)
    return ((a * class_w * bce).sum(dim=dims) / total).mean()


def operator_loss(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    area: torch.Tensor,
    target_scale: torch.Tensor,
    cfg: LossConfig,
    return_tensor_parts: bool = False,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Common physics-aware loss for FNO, GNO and GINO.

    ``pred_n`` and ``target_n`` are ``B,C,*space`` and ``area`` is
    ``B,*space``.  Field MSE is evaluated in normalized space; all physical
    regularizers use dimensional quantities reconstructed with the common
    target scale.
    """
    scale = target_scale.to(pred_n.device, dtype=pred_n.dtype).view(
        1, -1, *([1] * (pred_n.ndim - 2))
    )
    pred = pred_n * scale
    target = target_n * scale
    a = area.to(pred.device, dtype=pred.dtype)

    # Target support is defined with the same threshold used by the FV solver.
    thf, thsu, thcu, thsl, thcl = _project_phases_torch(target)
    h_t = thf + thsu + thsl
    wet = (h_t >= float(cfg.h_wet_m)).to(pred.dtype)
    dry = 1.0 - wet

    # 1) Shared normalized field error.  Dry cells remain in the objective;
    # wet cells are emphasized so the sparse flow body is not overwhelmed by
    # the large dry far field.
    weight = a * (1.0 + (float(cfg.wet_weight) - 1.0) * wet)
    dims = _spatial_dims(pred_n)
    mse_ch = _weighted_mean((pred_n - target_n) ** 2, weight, dims)
    data = mse_ch.mean()

    # Predicted admissible phase states used only by physical regularizers;
    # the raw outputs are still penalized by the data and admissibility terms.
    phf, phsu, phcu, phsl, phcl = _project_phases_torch(pred)
    h_p = phf + phsu + phsl

    # 2) Integrated conservative volumes.  Projection prevents negative phase
    # depths from spuriously cancelling positive volume errors.
    def integ(q):
        return (q * a).flatten(1).sum(1)

    pf, tf = integ(phf), integ(thf)
    pfi = integ((phsu - phcu) + (phsl - phcl))
    tfi = integ((thsu - thcu) + (thsl - thcl))
    pc, tc = integ(phcu + phcl), integ(thcu + thcl)
    mass = (
        ((pf - tf) / tf.abs().clamp_min(1.0)).pow(2).mean()
        + ((pfi - tfi) / tfi.abs().clamp_min(1.0)).pow(2).mean()
        + ((pc - tc) / tc.abs().clamp_min(1.0)).pow(2).mean()
    ) / 3.0

    # 3) Wet/dry support.  With h_wet=1e-3 and T=2e-4, a perfectly dry
    # prediction h=0 has occupancy sigmoid(-5)=0.0067 rather than ~0.5.
    temp = max(float(cfg.wet_temperature_m), 1.0e-8)
    occ_p = torch.sigmoid((h_p - float(cfg.h_wet_m)) / temp)
    support = _balanced_support_bce(occ_p, wet, a, cfg.max_support_class_weight)

    # Optional FNO-C-only hard-negative dry-front regularizer.
    # A target-dry cell predicted wet is a candidate for premature arrival.
    # Mine the hardest target-dry pixels per snapshot; this is only a proxy
    # for arrival time (a true onset loss requires temporally paired frames).
    early_front = pred_n.new_zeros(())
    if cfg.lambda_early_front > 0.0:
        dry_occupancy = occ_p.square().flatten(1)
        dry_mask = (dry > 0.5).flatten(1)
        candidate = dry_occupancy.masked_fill(~dry_mask, -1.0)
        frac = max(min(float(cfg.early_front_hard_fraction), 1.0), 1.0e-6)
        n_dry = dry_mask.sum(1)
        k_per_sample = torch.clamp((n_dry.float() * frac).ceil().long(), min=1)
        k_max = int(k_per_sample.max().item())
        vals = candidate.topk(k_max, dim=1).values
        rank = torch.arange(k_max, device=pred.device).unsqueeze(0)
        active = (rank < k_per_sample.unsqueeze(1)) & (vals >= 0.0)
        sample_penalty = (vals.clamp_min(0.0) * active).sum(1) / active.sum(1).clamp_min(1)
        early_front = sample_penalty.mean()

    # 4) Suppress the spectral low-amplitude halo in target-dry cells.  Smooth
    # L1 gives a strong but non-explosive penalty for centimetric false depth.
    h_ref = max(float(cfg.h_wet_m), 1.0e-8)
    excess = torch.relu(h_p - float(cfg.dry_depth_tolerance_m)) / h_ref
    dry_depth = _area_mean(F.smooth_l1_loss(excess, torch.zeros_like(excess), reduction="none"), a, dry)

    # 5) Momentum should vanish where the target is dry.  Use normalized
    # momentum channels so this term is dimensionless and shared across data
    # representations.
    dry_mom_field = pred_n[:, MX].pow(2) + pred_n[:, MY].pow(2)
    dry_momentum = _area_mean(dry_mom_field, a, dry)

    # 6) Raw-state admissibility.  Keep this diagnostic/penalty separate from
    # projection so the network is encouraged to learn valid states directly.
    neg_n = torch.relu(-pred_n[:, [HF, HSU, HCU, HSL, HCL]])
    neg_pen = _weighted_mean(neg_n.pow(2), a, dims).mean()
    # Layer coarse thickness cannot exceed corresponding total solid thickness.
    hsu_scale = target_scale[HSU].to(pred.device, dtype=pred.dtype).clamp_min(1.0e-8)
    hsl_scale = target_scale[HSL].to(pred.device, dtype=pred.dtype).clamp_min(1.0e-8)
    vu = torch.relu(pred[:, HCU] - pred[:, HSU]) / hsu_scale
    vl = torch.relu(pred[:, HCL] - pred[:, HSL]) / hsl_scale
    layer_pen = _area_mean(vu.pow(2) + vl.pow(2), a)
    admissible = neg_pen + layer_pen
    # L1-type supplement yields a non-vanishing gradient for widespread small
    # negative thickness/phase-excess halos (the v10 term was quadratic only).
    admissible_l1 = _weighted_mean(neg_n, a, dims).mean() + _area_mean(vu + vl, a)

    # 7) Mixture velocity (the same physical conversion as ml.metrics).
    # Momentum channels MX/MY are MASS FLUX per unit horizontal area,
    # so u=MX/(rho_f*HF + rho_s*(HSU+HSL)), not MX / flow thickness.
    mass_p = float(cfg.rho_f_kg_m3) * phf + float(cfg.rho_s_kg_m3) * (phsu + phsl)
    mass_t = float(cfg.rho_f_kg_m3) * thf + float(cfg.rho_s_kg_m3) * (thsu + thsl)
    min_depth = max(float(cfg.velocity_min_depth_m), 1.0e-6)
    min_mass = min(float(cfg.rho_f_kg_m3), float(cfg.rho_s_kg_m3)) * min_depth
    target_flow = (h_t >= min_depth).detach()
    both_flow = (target_flow & (h_p >= min_depth)).detach()

    velocity = pred.new_zeros(())
    consistency = pred.new_zeros(())
    if float(cfg.lambda_velocity) != 0.0 or float(cfg.lambda_momentum_consistency) != 0.0:
        up = pred[:, MX] / mass_p.clamp_min(min_mass)
        vp = pred[:, MY] / mass_p.clamp_min(min_mass)
        ut = target[:, MX] / mass_t.clamp_min(min_mass)
        vt = target[:, MY] / mass_t.clamp_min(min_mass)
        # Smooth L1 on normalized velocities: threshold in physical m/s.
        u_scale = max(float(cfg.velocity_scale_ms), 1.0e-6)
        delta = max(float(cfg.velocity_huber_delta_ms), 1.0e-6) / u_scale
        vdiff = (
            F.huber_loss(up / u_scale, ut / u_scale, delta=delta, reduction="none")
            + F.huber_loss(vp / u_scale, vt / u_scale, delta=delta, reduction="none")
        ) * 0.5
        velocity = _area_mean(vdiff, a, both_flow)

        # Avoid a shortcut in which pred depth approaches zero so that the
        # velocity term becomes masked. Penalize momentum inconsistent with
        # target velocity, using predicted mixture mass, on target-wet cells.
        mx_scale = target_scale[MX].to(pred.device, dtype=pred.dtype).clamp_min(1e-6)
        my_scale = target_scale[MY].to(pred.device, dtype=pred.dtype).clamp_min(1e-6)
        cx = (pred[:, MX] - mass_p * ut) / mx_scale
        cy = (pred[:, MY] - mass_p * vt) / my_scale
        c_field = (
            F.smooth_l1_loss(cx, torch.zeros_like(cx), reduction="none")
            + F.smooth_l1_loss(cy, torch.zeros_like(cy), reduction="none")
        ) * 0.5
        consistency = _area_mean(c_field, a, target_flow)

    total = (
        data
        + float(cfg.lambda_mass) * mass
        + float(cfg.lambda_wet) * support
        + float(cfg.lambda_early_front) * early_front
        + float(cfg.lambda_dry_depth) * dry_depth
        + float(cfg.lambda_dry_momentum) * dry_momentum
        + float(cfg.lambda_admissible) * admissible
        + float(cfg.lambda_admissible_l1) * admissible_l1
        + float(cfg.lambda_velocity) * velocity
        + float(cfg.lambda_momentum_consistency) * consistency
    )

    tensor_parts = {
        "data": data.detach(),
        "mass": mass.detach(),
        "wet_support": support.detach(),
        "early_front": early_front.detach(),
        "dry_depth": dry_depth.detach(),
        "dry_momentum": dry_momentum.detach(),
        "admissible": admissible.detach(),
        "admissible_l1": admissible_l1.detach(),
        "velocity": velocity.detach(),
        "momentum_consistency": consistency.detach(),
        "total": total.detach(),
    }
    # Training can aggregate the scalars on device and transfer them only once
    # per epoch; external callers retain the v11 float-dictionary API.
    if return_tensor_parts:
        return total, tensor_parts
    return total, {k: float(v) for k, v in tensor_parts.items()}


# Backward-compatible alias retained for any external imports.
def reduce_area(a: torch.Tensor) -> torch.Tensor:
    den = a.flatten(1).sum(1).clamp_min(1.0e-12)
    return den.view(-1, *([1] * (a.ndim - 1)))

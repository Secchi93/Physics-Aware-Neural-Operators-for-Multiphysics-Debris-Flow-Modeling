from __future__ import annotations

"""Reviewer-oriented field, hazard, temporal and physical metrics.

Derived hazard metrics use explicit physical thresholds.  Wet/dry support is
classified at ``h_wet_m`` (1 mm by default, matching the solver H_DRY), while
velocity is reported only above ``h_velocity_m`` (1 cm by default) to avoid
q/h amplification in vanishingly shallow spectral tails.  Raw admissibility
violations remain reported separately, so thresholding does not hide invalid
network outputs.
"""

from typing import Dict, Tuple
import math
import numpy as np
from scipy.spatial import cKDTree

HF, HSU, HCU, HSL, HCL, MX, MY, DZB = range(8)


def project_admissible_state(Y: np.ndarray) -> np.ndarray:
    """Return a copy with physically admissible phase thicknesses.

    Momentum and bed change are left untouched.  This projection is used for
    derived physical quantities only; raw violation areas are evaluated from
    the original prediction in :func:`frame_metrics`.
    """
    Q = np.asarray(Y, dtype=float).copy()
    Q[HF] = np.maximum(Q[HF], 0.0)
    Q[HSU] = np.maximum(Q[HSU], 0.0)
    Q[HSL] = np.maximum(Q[HSL], 0.0)
    Q[HCU] = np.minimum(np.maximum(Q[HCU], 0.0), Q[HSU])
    Q[HCL] = np.minimum(np.maximum(Q[HCL], 0.0), Q[HSL])
    return Q


def primitive_from_target(
    Y: np.ndarray,
    rho_f: float = 1000.0,
    rho_s: float = 2650.0,
    h_wet_m: float = 1.0e-3,
    h_velocity_m: float = 1.0e-2,
    h_dry: float | None = None,
):
    """Convert conservative targets to depth and velocity primitives.

    ``h_dry`` is accepted only for backward compatibility; when supplied it
    overrides ``h_wet_m``.  Velocity is deliberately masked below
    ``h_velocity_m`` because momentum/depth ratios are not physically useful
    in the near-dry spectral halo.
    """
    if h_dry is not None:
        h_wet_m = float(h_dry)
    Q = project_admissible_state(Y)
    hf, hsu, hsl = Q[HF], Q[HSU], Q[HSL]
    h = hf + hsu + hsl
    mass = rho_f * hf + rho_s * (hsu + hsl)
    wet = h >= float(h_wet_m)
    vel_valid = (h >= float(h_velocity_m)) & (mass > 0.0)
    u = np.zeros_like(h)
    v = np.zeros_like(h)
    u[vel_valid] = Q[MX][vel_valid] / np.maximum(mass[vel_valid], 1.0e-30)
    v[vel_valid] = Q[MY][vel_valid] / np.maximum(mass[vel_valid], 1.0e-30)
    return h, u, v, np.hypot(u, v), wet


def _wstats(p, t, w, mask=None, prefix="", relative_floor: float = 1.0e-12) -> Dict[str, float]:
    p = np.asarray(p, float).ravel()
    t = np.asarray(t, float).ravel()
    w = np.asarray(w, float).ravel()
    m = np.isfinite(p) & np.isfinite(t) & np.isfinite(w) & (w > 0)
    if mask is not None:
        m &= np.asarray(mask, bool).ravel()
    if not np.any(m):
        return {
            f"{prefix}mae": float("nan"),
            f"{prefix}rmse": float("nan"),
            f"{prefix}nrmse": float("nan"),
            f"{prefix}rel_l2": float("nan"),
            f"{prefix}r2": float("nan"),
        }
    pp, tt, ww = p[m], t[m], w[m]
    W = ww.sum()
    err = pp - tt
    mae = np.sum(ww * np.abs(err)) / W
    rmse = math.sqrt(np.sum(ww * err * err) / W)
    rms_t = math.sqrt(np.sum(ww * tt * tt) / W)
    mean = np.sum(ww * tt) / W
    sse = np.sum(ww * err * err)
    sst = np.sum(ww * (tt - mean) ** 2)

    # Relative metrics are mathematically undefined when the target field is
    # essentially zero; report NaN rather than astronomically large numbers.
    if rms_t <= float(relative_floor):
        nrmse = rel = float("nan")
    else:
        nrmse = rmse / rms_t
        rel = math.sqrt(np.sum(ww * err * err) / np.sum(ww * tt * tt))
    r2 = float("nan") if sst <= (float(relative_floor) ** 2) * W else 1.0 - sse / sst
    return {
        f"{prefix}mae": float(mae),
        f"{prefix}rmse": float(rmse),
        f"{prefix}nrmse": float(nrmse),
        f"{prefix}rel_l2": float(rel),
        f"{prefix}r2": float(r2),
    }


def wet_metrics(pred_wet: np.ndarray, true_wet: np.ndarray, area: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred_wet, bool).ravel()
    t = np.asarray(true_wet, bool).ravel()
    a = np.asarray(area, float).ravel()
    inter = np.sum(a[p & t])
    union = np.sum(a[p | t])
    ap = np.sum(a[p])
    at = np.sum(a[t])
    tp = inter
    fp = np.sum(a[p & ~t])
    fn = np.sum(a[~p & t])
    return {
        "wet_iou": float(inter / max(union, 1.0e-30)),
        "wet_dice": float(2 * inter / max(ap + at, 1.0e-30)),
        "wet_precision": float(tp / max(tp + fp, 1.0e-30)),
        "wet_recall": float(tp / max(tp + fn, 1.0e-30)),
        "false_positive_wet_area_m2": float(fp),
        "false_negative_wet_area_m2": float(fn),
        "pred_inundated_area_m2": float(ap),
        "true_inundated_area_m2": float(at),
        "inundated_area_error_m2": float(ap - at),
    }


def _coords_flat(coords_m: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    c = np.asarray(coords_m)
    if c.ndim == 2 and c.shape[-1] == 2:
        return c.reshape((-1, 2))
    if c.ndim == 3 and c.shape[-1] == 2:
        return c.reshape((-1, 2))
    raise ValueError(f"unexpected coordinate shape {c.shape}")


def hausdorff_metrics(pwet: np.ndarray, twet: np.ndarray, coords_m: np.ndarray) -> Dict[str, float]:
    coords = _coords_flat(coords_m, pwet.shape)
    P = coords[np.asarray(pwet).ravel()]
    T = coords[np.asarray(twet).ravel()]
    if len(P) == 0 or len(T) == 0:
        return {"hausdorff_m": float("nan"), "hd95_m": float("nan")}
    tp = cKDTree(P)
    tt = cKDTree(T)
    dpt = tt.query(P, k=1)[0]
    dtp = tp.query(T, k=1)[0]
    both = np.concatenate([dpt, dtp])
    return {"hausdorff_m": float(np.max(both)), "hd95_m": float(np.percentile(both, 95))}


def release_centroid(U0: np.ndarray, area: np.ndarray, coords_m: np.ndarray, h_wet_m: float = 1.0e-3) -> np.ndarray:
    Q = project_admissible_state(U0)
    h = Q[HF] + Q[HSU] + Q[HSL]
    m = h.ravel() >= float(h_wet_m)
    coords = _coords_flat(coords_m, h.shape)
    w = (np.asarray(area).ravel() * h.ravel())[m]
    if not np.any(m):
        return np.nanmean(coords, axis=0)
    return np.sum(coords[m] * w[:, None], axis=0) / max(np.sum(w), 1.0e-30)


def runout_distance(wet: np.ndarray, coords_m: np.ndarray, origin: np.ndarray) -> float:
    coords = _coords_flat(coords_m, wet.shape)
    m = np.asarray(wet).ravel()
    if not np.any(m):
        return 0.0
    return float(np.max(np.linalg.norm(coords[m] - origin[None, :], axis=1)))


def _safe_rel_error(pred: float, true: float, floor: float) -> float:
    if abs(true) < float(floor):
        return float("nan")
    return float((pred - true) / abs(true))


def integral_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    area: np.ndarray,
    initial_mobile_volume_m3: float | None = None,
    relative_floor_m3: float = 1.0,
) -> Dict[str, float]:
    """Integrated volume metrics with stable small-denominator handling."""
    a = np.asarray(area, float)
    P = project_admissible_state(pred)
    T = project_admissible_state(target)

    def I(q):
        return float(np.sum(q * a))

    out: Dict[str, float] = {}
    pairs = {
        "fluid": (P[HF], T[HF]),
        "fine": ((P[HSU] - P[HCU]) + (P[HSL] - P[HCL]), (T[HSU] - T[HCU]) + (T[HSL] - T[HCL])),
        "coarse": (P[HCU] + P[HCL], T[HCU] + T[HCL]),
    }
    for name, (p, t) in pairs.items():
        ip, it = I(p), I(t)
        out[f"{name}_volume_pred_m3"] = ip
        out[f"{name}_volume_true_m3"] = it
        out[f"{name}_volume_error_m3"] = ip - it
        out[f"{name}_volume_rel_error"] = _safe_rel_error(ip, it, relative_floor_m3)

    dep_p, dep_t = I(np.maximum(P[DZB], 0.0)), I(np.maximum(T[DZB], 0.0))
    ero_p, ero_t = I(np.maximum(-P[DZB], 0.0)), I(np.maximum(-T[DZB], 0.0))
    out.update(
        {
            "deposit_volume_pred_m3": dep_p,
            "deposit_volume_true_m3": dep_t,
            "deposit_volume_error_m3": dep_p - dep_t,
            "deposit_volume_rel_error": _safe_rel_error(dep_p, dep_t, relative_floor_m3),
            "erosion_volume_pred_m3": ero_p,
            "erosion_volume_true_m3": ero_t,
            "erosion_volume_error_m3": ero_p - ero_t,
            "erosion_volume_rel_error": _safe_rel_error(ero_p, ero_t, relative_floor_m3),
        }
    )
    if initial_mobile_volume_m3 is not None and initial_mobile_volume_m3 > 0:
        V0 = float(initial_mobile_volume_m3)
        out["deposit_volume_error_over_release"] = float((dep_p - dep_t) / V0)
        out["erosion_volume_error_over_release"] = float((ero_p - ero_t) / V0)
    return out


def frame_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    area: np.ndarray,
    coords_m: np.ndarray,
    U0: np.ndarray,
    h_wet_m: float = 1.0e-3,
    h_velocity_m: float = 1.0e-2,
    relative_volume_floor_fraction: float = 1.0e-3,
    h_dry: float | None = None,
) -> Dict[str, float]:
    if h_dry is not None:
        h_wet_m = float(h_dry)
    hp, up, vp, sp, wp = primitive_from_target(pred, h_wet_m=h_wet_m, h_velocity_m=h_velocity_m)
    ht, ut, vt, st, wt = primitive_from_target(target, h_wet_m=h_wet_m, h_velocity_m=h_velocity_m)
    vel_union = (hp >= float(h_velocity_m)) | (ht >= float(h_velocity_m))
    out: Dict[str, float] = {}
    out.update(_wstats(hp, ht, area, prefix="h_", relative_floor=1.0e-6))
    out.update(_wstats(up, ut, area, vel_union, prefix="u_", relative_floor=1.0e-3))
    out.update(_wstats(vp, vt, area, vel_union, prefix="v_", relative_floor=1.0e-3))
    out.update(_wstats(sp, st, area, vel_union, prefix="speed_", relative_floor=1.0e-3))
    out.update(wet_metrics(wp, wt, area))
    out.update(hausdorff_metrics(wp, wt, coords_m))

    U0p = project_admissible_state(U0)
    initial_mobile_volume_m3 = float(np.sum((U0p[HF] + U0p[HSU] + U0p[HSL]) * np.asarray(area, float)))
    volume_floor = max(1.0, float(relative_volume_floor_fraction) * initial_mobile_volume_m3)
    out.update(integral_metrics(pred, target, area, initial_mobile_volume_m3, volume_floor))

    origin = release_centroid(U0, area, coords_m, h_wet_m)
    rp = runout_distance(wp, coords_m, origin)
    rt = runout_distance(wt, coords_m, origin)
    out.update(
        {
            "runout_pred_m": rp,
            "runout_true_m": rt,
            "runout_error_m": rp - rt,
            "max_depth_pred_m": float(np.max(hp)),
            "max_depth_true_m": float(np.max(ht)),
            "max_depth_error_m": float(np.max(hp) - np.max(ht)),
            "max_speed_pred_ms": float(np.max(sp)),
            "max_speed_true_ms": float(np.max(st)),
            "max_speed_error_ms": float(np.max(sp) - np.max(st)),
        }
    )

    # Raw-state diagnostics are deliberately computed before projection.
    raw = np.asarray(pred)
    neg = np.sum(np.asarray(area)[np.any(raw[[HF, HSU, HCU, HSL, HCL]] < -1.0e-8, axis=0)])
    layer = np.sum(np.asarray(area)[(raw[HCU] > raw[HSU] + 1.0e-8) | (raw[HCL] > raw[HSL] + 1.0e-8)])
    out["negative_state_area_m2"] = float(neg)
    out["layer_constraint_violation_area_m2"] = float(layer)
    return out


def arrival_time_metrics(
    pred_h: np.ndarray,
    true_h: np.ndarray,
    times: np.ndarray,
    area: np.ndarray,
    h_wet_m: float = 1.0e-3,
    h_dry: float | None = None,
) -> Dict[str, float]:
    """First wetting-time MAE over cells reached by either trajectory."""
    if h_dry is not None:
        h_wet_m = float(h_dry)
    P = np.asarray(pred_h)
    T = np.asarray(true_h)
    tt = np.asarray(times, float)
    a = np.asarray(area, float).ravel()
    p = P.reshape((P.shape[0], -1)) >= float(h_wet_m)
    t = T.reshape((T.shape[0], -1)) >= float(h_wet_m)
    p_any = p.any(0)
    t_any = t.any(0)
    union = p_any | t_any
    if not np.any(union):
        return {
            "arrival_time_mae_s": 0.0,
            "arrival_time_bias_s": 0.0,
            "arrival_false_reached_area_m2": 0.0,
            "arrival_missed_area_m2": 0.0,
        }
    sentinel = float(tt[-1] + (tt[-1] - tt[-2] if len(tt) > 1 else 1.0))
    pa = np.full(p.shape[1], sentinel)
    ta = np.full(t.shape[1], sentinel)
    for i in np.where(p_any)[0]:
        pa[i] = tt[np.argmax(p[:, i])]
    for i in np.where(t_any)[0]:
        ta[i] = tt[np.argmax(t[:, i])]
    w = a[union]
    d = pa[union] - ta[union]
    return {
        "arrival_time_mae_s": float(np.sum(w * np.abs(d)) / max(np.sum(w), 1.0e-30)),
        "arrival_time_bias_s": float(np.sum(w * d) / max(np.sum(w), 1.0e-30)),
        "arrival_false_reached_area_m2": float(np.sum(a[p_any & ~t_any])),
        "arrival_missed_area_m2": float(np.sum(a[~p_any & t_any])),
    }

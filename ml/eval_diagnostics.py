from __future__ import annotations

"""Additive evaluation diagnostics for Morino neural-operator outputs.

These diagnostics do not alter the v10 field metrics. Physical fields are
admissibility-projected only for *derived* quantities; raw-state violations are
measured on the unprojected network prediction.
"""

from typing import Dict, Sequence

import numpy as np
from scipy import ndimage

from .metrics import (
    HF, HSU, HCU, HSL, HCL,
    _wstats, primitive_from_target, project_admissible_state,
    release_centroid, runout_distance, wet_metrics,
)


def _name_for_threshold(value_m: float) -> str:
    """Stable, collision-free column suffix, e.g. 0.001 -> 1mm."""
    mm = float(value_m) * 1000.0
    if mm <= 0 or not np.isfinite(mm):
        raise ValueError("thresholds must be finite and positive")
    return (f"{mm:g}".replace(".", "p") + "mm")


def _wpercentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    vv, ww = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    m = np.isfinite(vv) & np.isfinite(ww) & (ww > 0)
    if not np.any(m):
        return float("nan")
    vv, ww = vv[m], ww[m]
    order = np.argsort(vv)
    vv, ww = vv[order], ww[order]
    return float(vv[min(np.searchsorted(np.cumsum(ww), percentile / 100.0 * ww.sum(), side="left"), len(vv) - 1)])


def _largest_component(wet: np.ndarray, area: np.ndarray, min_area_m2: float) -> tuple[np.ndarray, float]:
    """Keep the largest 8-connected raster component by cell area.

    Applies to regular Cartesian/rasterized outputs only. For an unstructured
    point cloud, a 1-D cell order is not a meaningful spatial connectivity.
    """
    m = np.asarray(wet, bool) & (np.asarray(area) > 0)
    if m.ndim != 2:
        raise ValueError("largest-component runout needs a 2-D raster")
    labels, n = ndimage.label(m, structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return np.zeros_like(m), 0.0
    sizes = np.bincount(labels.ravel(), weights=np.asarray(area, float).ravel(), minlength=n + 1)
    sizes[0] = 0.0
    idx = int(np.argmax(sizes))
    if sizes[idx] < min_area_m2:
        return np.zeros_like(m), float(np.sum(np.asarray(area)[m]))
    main = labels == idx
    return main, float(np.sum(np.asarray(area)[m & ~main]))


def frame_diagnostics(
    pred: np.ndarray,
    target: np.ndarray,
    area: np.ndarray,
    coords_m: np.ndarray,
    U0: np.ndarray,
    wet_thresholds_m: Sequence[float] = (0.001, 0.01, 0.05, 0.1),
    h_velocity_m: float = 0.01,
    velocity_thresholds_m: Sequence[float] = (0.01, 0.05),
    min_component_area_m2: float = 500.0,
    admissibility_tolerances_m: Sequence[float] = (1.0e-4, 1.0e-3),
) -> Dict[str, float]:
    """Return optional new columns; default v10 metrics remain unchanged."""
    P = project_admissible_state(pred)
    T = project_admissible_state(target)
    hp = P[HF] + P[HSU] + P[HSL]
    ht = T[HF] + T[HSU] + T[HSL]
    a = np.asarray(area, float)
    out: Dict[str, float] = {}

    for threshold in wet_thresholds_m:
        name = _name_for_threshold(threshold)
        wp, wt = hp >= threshold, ht >= threshold
        wm = wet_metrics(wp, wt, a)
        for k in ("wet_iou", "wet_dice", "wet_precision", "wet_recall",
                  "false_positive_wet_area_m2", "false_negative_wet_area_m2"):
            out[f"{k}_{name}"] = wm[k]
        for title, mask in (("truewet", wt), ("wetunion", wt | wp)):
            s = _wstats(hp, ht, a, mask=mask, relative_floor=1.0e-6)
            for metric in ("mae", "rmse", "r2", "rel_l2"):
                out[f"h_{title}_{metric}_{name}"] = s[metric]
            out[f"h_{title}_area_m2_{name}"] = float(np.sum(a[mask & (a > 0)]))

        # Suppress small isolated islands as a SECONDARY runout metric. The
        # original extreme-cell runout remains the primary raw metric.
        if wp.ndim == 2:
            pmain, pdropped = _largest_component(wp, a, min_component_area_m2)
            tmain, tdropped = _largest_component(wt, a, min_component_area_m2)
            origin = release_centroid(U0, a, coords_m, h_wet_m=threshold)
            rp = runout_distance(pmain, coords_m, origin)
            rt = runout_distance(tmain, coords_m, origin)
            out[f"runout_main_pred_m_{name}"] = rp
            out[f"runout_main_true_m_{name}"] = rt
            out[f"runout_main_error_m_{name}"] = rp - rt
            out[f"runout_main_abs_error_m_{name}"] = abs(rp - rt)
            out[f"runout_main_ignored_pred_area_m2_{name}"] = pdropped
            out[f"runout_main_ignored_true_area_m2_{name}"] = tdropped
        else:
            # Do not silently treat unstructured cell order as adjacency.
            for k in ("runout_main_pred_m", "runout_main_true_m", "runout_main_error_m",
                      "runout_main_abs_error_m", "runout_main_ignored_pred_area_m2",
                      "runout_main_ignored_true_area_m2"):
                out[f"{k}_{name}"] = float("nan")

    # Exclude shallow cells symmetrically from the *supplementary* velocity
    # diagnostics; report the retained area so changes are not obscured.
    for threshold in velocity_thresholds_m:
        name = _name_for_threshold(threshold)
        hp2, up, vp, speedp, _ = primitive_from_target(P, h_velocity_m=threshold)
        ht2, ut, vt, speedt, _ = primitive_from_target(T, h_velocity_m=threshold)
        both_wet = (hp2 >= threshold) & (ht2 >= threshold) & (a > 0)
        true_wet = (ht2 >= threshold) & (a > 0)
        out[f"velocity_bothwet_area_m2_{name}"] = float(np.sum(a[both_wet]))
        out[f"velocity_truewet_area_m2_{name}"] = float(np.sum(a[true_wet]))
        out[f"velocity_bothwet_coverage_{name}"] = (
            float(np.sum(a[both_wet]) / np.sum(a[true_wet]))
            if np.any(true_wet) else float("nan")
        )
        for component, pp, tt in (("u", up, ut), ("v", vp, vt), ("speed", speedp, speedt)):
            s = _wstats(pp, tt, a, mask=both_wet, relative_floor=1.0e-3)
            for metric in ("mae", "rmse"):
                out[f"{component}_bothwet_{metric}_{name}"] = s[metric]
        if np.any(both_wet):
            out[f"speed_pred_p95_bothwet_ms_{name}"] = _wpercentile(speedp[both_wet], a[both_wet], 95)
            out[f"speed_pred_p99_bothwet_ms_{name}"] = _wpercentile(speedp[both_wet], a[both_wet], 99)
            out[f"speed_pred_max_bothwet_ms_{name}"] = float(np.max(speedp[both_wet]))
            out[f"speed_true_max_bothwet_ms_{name}"] = float(np.max(speedt[both_wet]))
            out[f"speed_pred_over_10ms_area_m2_{name}"] = float(np.sum(a[both_wet & (speedp > 10.0)]))
        else:
            for key in ("speed_pred_p95_bothwet_ms", "speed_pred_p99_bothwet_ms",
                        "speed_pred_max_bothwet_ms", "speed_true_max_bothwet_ms",
                        "speed_pred_over_10ms_area_m2"):
                out[f"{key}_{name}"] = float("nan")

    raw = np.asarray(pred, float)
    phase = raw[[HF, HSU, HCU, HSL, HCL]]
    negative_depth = np.maximum(-phase, 0.0)
    upper_excess = np.maximum(raw[HCU] - raw[HSU], 0.0)
    lower_excess = np.maximum(raw[HCL] - raw[HSL], 0.0)
    # Thickness integrals add phase coordinates; they are constraint-severity
    # diagnostics, NOT actual extra mobile mass or independent material volume.
    out["raw_negative_thickness_integral_m3"] = float(np.sum(negative_depth * a[None]))
    out["raw_layer_excess_integral_m3"] = float(np.sum((upper_excess + lower_excess) * a))
    for threshold in admissibility_tolerances_m:
        name = _name_for_threshold(threshold)
        neg_mask = np.any(phase < -threshold, axis=0)
        layer_mask = (raw[HCU] - raw[HSU] > threshold) | (raw[HCL] - raw[HSL] > threshold)
        out[f"negative_state_area_m2_tol_{name}"] = float(np.sum(a[neg_mask & (a > 0)]))
        out[f"layer_violation_area_m2_tol_{name}"] = float(np.sum(a[layer_mask & (a > 0)]))

    return out

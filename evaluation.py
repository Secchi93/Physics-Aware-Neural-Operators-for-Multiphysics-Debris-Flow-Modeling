from __future__ import annotations
"""Lean paper-only evaluator for FNO/GNO/GINO; no training or checkpoint changes.

Metrics: h RMSE/R2, wet IoU and false/missed area (1 mm), runout absolute
error, both-wet speed MAE (50 mm), fluid volume relative bias, conditional
arrival-time MAE and bias on mutually reached cells (1 mm), inference timing.
FNO-A is evaluated against the *native* adaptive solver after backprojection.
Per-frame means -> per-scenario means -> 30-scenario split means for fields.
Arrival metrics are computed per whole scenario, then averaged across scenarios.
"""
import argparse
import csv
import json
import os
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data import MorinoFrameDataset
from ml.metrics import (primitive_from_target, _wstats, wet_metrics,
                        release_centroid, runout_distance, project_admissible_state)
from ml.models import build_model
from ml.rasterize import apply_raster_to_source_map


def _save_csv(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _agg(rows, excluded=()):
    keys = list(dict.fromkeys(k for r in rows for k in r if k not in excluded
                               and isinstance(r[k], (int, float, np.integer, np.floating))))
    result = {}
    for k in keys:
        x = np.array([float(r[k]) for r in rows if k in r and np.isfinite(float(r[k]))])
        if x.size:
            result[k] = {'mean': float(x.mean()), 'std': float(x.std()), 'n': int(x.size)}
    return result


def _ref_policy(rep):
    if rep == 'adaptive_rasterized':
        return 'adaptive_native_pre_rasterization', 'area_overlap_raster_to_native'
    return ('adaptive_native' if rep == 'adaptive' else 'cartesian_native'), 'none'


def _native_frame(ds, index):
    ref = ds.frames[index]
    row = ds.rows[ref.row_index]
    pid = int(row['profile_id'])
    path = row['adaptive_path']
    U0, z0, _, _, _, area, _, _, coords_m = ds._static_adapt(path, pid)
    with h5py.File(path, 'r') as handle:
        t = float(handle['times'][ref.time_index])
        U = handle['U'][ref.time_index].astype(np.float32)
        zt = handle['zb'][ref.time_index].astype(np.float32)
    target = np.concatenate((U, (zt-z0)[None]), axis=0).astype(np.float32)
    return target, np.asarray(area), np.asarray(coords_m), np.asarray(U0), t, ds._rmap(pid)


def _coords(batch, rep):
    if rep in ('adaptive', 'cartesian_points'):
        return batch['coords_m'][0].numpy()
    x, y = batch['x_coord'][0].numpy(), batch['y_coord'][0].numpy()
    Y, X = np.meshgrid(y, x, indexing='ij')
    return np.stack((X, Y), axis=-1)


def _forward(model, kind, batch, device):
    x = batch['x'].to(device)
    if kind == 'fno':
        return model(x)
    return model(x=x, coords=batch['coords'].to(device),
                 area=batch['area'].to(device),
                 profile_id=int(batch['profile_id'][0])).permute(0, 2, 1)


def paper_frame_metrics(pred, target, area, coords, U0, h_wet=0.001):
    """Calculate only metrics used by the paper, following ml.metrics semantics."""
    a = np.asarray(area, dtype=float)
    hp, _, _, _, wp = primitive_from_target(pred, h_wet_m=h_wet, h_velocity_m=0.05)
    ht, _, _, _, wt = primitive_from_target(target, h_wet_m=h_wet, h_velocity_m=0.05)
    depth = _wstats(hp, ht, a, prefix='h_', relative_floor=1e-6)
    wet = wet_metrics(wp, wt, a)
    origin = release_centroid(U0, a, coords, h_wet)
    rp = runout_distance(wp, coords, origin)
    rt = runout_distance(wt, coords, origin)

    # Exactly the both-wet 50-mm metric defined in ml.eval_diagnostics.
    _, _, _, sp, _ = primitive_from_target(pred, h_velocity_m=0.05)
    _, _, _, st, _ = primitive_from_target(target, h_velocity_m=0.05)
    both = (hp >= 0.05) & (ht >= 0.05) & (a > 0)
    sm = _wstats(sp, st, a, mask=both, relative_floor=1e-3)['mae']

    # Same project-and-integrate convention as ml.metrics.integral_metrics.
    P = project_admissible_state(pred)
    T = project_admissible_state(target)
    volp = float(np.sum(P[0] * a))
    volt = float(np.sum(T[0] * a))
    U = project_admissible_state(U0)
    v0 = float(np.sum((U[0]+U[1]+U[3])*a))
    floor = max(1.0, 1e-3*v0)
    bias = (volp-volt)/abs(volt) if abs(volt) >= floor else float('nan')

    return {
        'h_rmse': depth['h_rmse'], 'h_r2': depth['h_r2'],
        'wet_iou': wet['wet_iou'],
        'false_positive_wet_area_m2': wet['false_positive_wet_area_m2'],
        'false_negative_wet_area_m2': wet['false_negative_wet_area_m2'],
        'runout_abs_error_m': abs(rp-rt),
        'speed_bothwet_mae_50mm': sm,
        'fluid_volume_rel_error': bias,
    }, hp, ht


def _arrival(pred_h, true_h, times, area, threshold=0.001):
    p = np.asarray(pred_h).reshape(len(times), -1) >= threshold
    t = np.asarray(true_h).reshape(len(times), -1) >= threshold
    a = np.asarray(area, float).ravel()
    tt = np.asarray(times, float)
    common = p.any(axis=0) & t.any(axis=0)
    false = p.any(axis=0) & ~t.any(axis=0)
    missed = ~p.any(axis=0) & t.any(axis=0)
    if common.any():
        dt = tt[np.argmax(p, axis=0)][common] - tt[np.argmax(t, axis=0)][common]
        mae = float(np.average(abs(dt), weights=a[common]))
        bias = float(np.average(dt, weights=a[common]))
    else:
        mae = bias = float('nan')
    return {
        'arrival_time_mae_s': mae, 'arrival_time_bias_s': bias,
        'arrival_intersection_mae_s': mae, 'arrival_intersection_bias_s': bias,
        'arrival_intersection_area_m2': float(a[common].sum()),
        'arrival_false_reached_area_m2': float(a[false].sum()),
        'arrival_missed_area_m2': float(a[missed].sum()),
    }


def main():
    p = argparse.ArgumentParser(description='Fast evaluation of paper-only metrics')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest')
    p.add_argument('--out')
    p.add_argument('--splits', nargs='+', default=['iid_test', 'single_ood_test', 'joint_ood_test'])
    p.add_argument('--frame-stride', type=int, default=1)
    args = p.parse_args()
    if args.frame_stride != 1:
        p.error('Use --frame-stride 1: skipping frames invalidates first-arrival time.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    kind = str(ck['kind']).lower()
    cfg = ck['config']
    rep = str(ck.get('representation', cfg['data']['representation']))
    model = build_model(kind, 13, 8, cfg['model'])
    state = ck['model'].copy()
    state.pop('_metadata', None)
    model.load_state_dict(state)
    model = model.to(device).eval()
    scale = np.asarray(ck['target_scale'], np.float32)
    # Do not trust the dataset path embedded in older checkpoints: it refers
    # to the author's original study directory, not to the published package.
    manifest = os.path.abspath(args.manifest or os.path.join(os.path.dirname(__file__), 'output', 'dataset', 'manifest.csv'))
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f'Manifest not found: {manifest}. Supply --manifest or download the Zenodo dataset into output/dataset/.')
    outdir = os.path.abspath(args.out or os.path.join(os.path.dirname(args.checkpoint)))
    os.makedirs(outdir, exist_ok=True)
    reference, projection = _ref_policy(rep)
    print(f'[FAST] model={kind} rep={rep} device={device} reference={reference}', flush=True)

    frame_rows, scenario_rows, traj_rows = [], [], []
    timing = {}
    for split in args.splits:
        ds = MorinoFrameDataset(manifest, split, rep, scale, frame_stride=1,
                                include_t0=True,
                                raster_dx=float(cfg['data'].get('raster_dx', 10.0)),
                                cache_static_items=int(cfg['data'].get('cache_static_items', 12)))
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        cur = None
        frames, ph, th, times = [], [], [], []
        a_cur = None
        total_inf, total_post, num_frames = 0., 0., 0

        def flush():
            nonlocal frames, ph, th, times, a_cur
            if cur is None or not times:
                return
            # Paper fields exclude t0; arrival includes t0.
            if frames:
                avg = _agg(frames, excluded=('time_s', 'time_index'))
                scenario_rows.append({'split': split, 'scenario_id': cur,
                                      **{k: v['mean'] for k, v in avg.items()}})
            arrival = _arrival(np.stack(ph), np.stack(th), times, a_cur)
            traj_rows.append({'split': split, 'scenario_id': cur, **arrival})
            frames, ph, th, times, a_cur = [], [], [], [], None

        for idx, batch in enumerate(loader):
            sid = batch['scenario_id'][0]
            if cur is None: cur = sid
            if sid != cur:
                flush()
                cur = sid
            time_s = float(batch['time'][0])
            x0 = batch['x'][0].numpy()
            if rep in ('adaptive', 'cartesian_points'):
                U0 = np.moveaxis(x0[..., :7], -1, 0) * scale[:7, None]
            else:
                U0 = x0[:7] * scale[:7, None, None]
            coords = _coords(batch, rep)
            area = batch['area'][0].numpy()
            if device.type == 'cuda': torch.cuda.synchronize()
            st = time.perf_counter()
            with torch.inference_mode():
                output = _forward(model, kind, batch, device)
            if device.type == 'cuda': torch.cuda.synchronize()
            inf = time.perf_counter() - st
            total_inf += inf
            num_frames += 1
            pred_n = output[0].cpu().numpy()
            truth_n = batch['y'][0].numpy().T if kind in ('gno', 'gino') else batch['y'][0].numpy()
            sc = scale.reshape((8,) + (1,)*(pred_n.ndim-1))
            pred, truth = pred_n*sc, truth_n*sc
            if rep == 'adaptive_rasterized':
                native_truth, area, coords, U0, native_time, rmap = _native_frame(ds, idx)
                if abs(native_time-time_s)>1e-3:
                    raise ValueError(f'native frame time mismatch: {sid}')
                st = time.perf_counter()
                pred = apply_raster_to_source_map(rmap, pred)
                total_post += time.perf_counter()-st
                truth, time_s = native_truth, native_time
            is_t0 = int(batch['time_index'][0]) == 0
            if is_t0:
                hp, *_ = primitive_from_target(pred, h_wet_m=0.001, h_velocity_m=0.05)
                ht, *_ = primitive_from_target(truth, h_wet_m=0.001, h_velocity_m=0.05)
                metrics = None
            else:
                metrics, hp, ht = paper_frame_metrics(pred, truth, area, coords, U0)
            ph.append(hp)
            th.append(ht)
            times.append(time_s)
            a_cur = area
            if not is_t0:
                row = {'split': split, 'scenario_id': sid, 'profile_id': int(batch['profile_id'][0]),
                       'time_s': time_s, 'time_index': int(batch['time_index'][0]), **metrics}
                frame_rows.append(row)
                frames.append(row)
        flush()
        timing[split] = {
            'frames': num_frames, 'total_inference_s': total_inf,
            'mean_inference_s_per_frame': total_inf/max(num_frames, 1),
            'total_postprocess_s': total_post,
            'mean_inference_plus_postprocess_s_per_frame': (total_inf+total_post)/max(num_frames, 1),
        }
        ar = [r for r in traj_rows if r['split'] == split]
        am = _agg(ar, excluded=('split', 'scenario_id'))
        print(f'[FAST] {split}: frames={num_frames}, arrival intersection MAE='
              f"{am.get('arrival_time_mae_s', {}).get('mean', float('nan')):.1f} s", flush=True)

    _save_csv(os.path.join(outdir, 'frame_metrics.csv'), frame_rows)
    _save_csv(os.path.join(outdir, 'scenario_metrics_post_t0.csv'), scenario_rows)
    _save_csv(os.path.join(outdir, 'trajectory_metrics.csv'), traj_rows)
    summary = {
        'evaluation_version': 'paper_fast_intersection_v1',
        'kind': kind, 'representation': rep, 'checkpoint': os.path.abspath(args.checkpoint),
        'reference_space': reference, 'prediction_postprocess': projection,
        'evaluation_protocol': {
            'field_aggregation': 'mean of post-t0 frame metrics per scenario; mean across scenarios',
            'arrival_aggregation': 'per-scenario area-weighted intersection timing MAE; mean across scenarios',
            'arrival_threshold_m': 0.001, 'speed_bothwet_threshold_m': 0.05,
            'no_union_sentinel_penalty': True,
        },
        'scenario_metrics_post_t0': {}, 'trajectory_metrics': {}, 'timing': timing,
    }
    for split in args.splits:
        sf = [r for r in scenario_rows if r['split']==split]
        tr = [r for r in traj_rows if r['split']==split]
        summary['scenario_metrics_post_t0'][split] = _agg(sf, excluded=('split','scenario_id'))
        summary['trajectory_metrics'][split] = _agg(tr, excluded=('split','scenario_id'))
    with open(os.path.join(outdir, 'summary.json'), 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2)
    print('[SAVE]', outdir, flush=True)

if __name__=='__main__':
    main()

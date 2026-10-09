"""Pre-flight HDF5 coverage check for the 300x2 Morino controlled benchmark."""
from __future__ import annotations
import csv
import os
import numpy as np
import h5py

from .data import resolve_manifest_data_path


def check_training_inputs(manifest: str, representation: str, scales_path: str,
                          full_run: bool = True, seed: int = 1234,
                          run_fraction: float = 1.0) -> None:
    if not os.path.isfile(manifest):
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    if not os.path.isfile(scales_path):
        raise FileNotFoundError(f"Missing original shared normalization: {scales_path}")
    key = 'cartesian_path' if representation in ('cartesian', 'cartesian_points') else 'adaptive_path'
    with open(manifest, newline='', encoding='utf-8') as fp:
        rows = list(csv.DictReader(fp))
    problems=[]
    for split, expected in (('train',180),('validation',30)):
        selected=[r for r in rows if r['split']==split]
        if split == 'train' and run_fraction < 1.0:
            rng=np.random.default_rng(seed)
            order=rng.permutation(len(selected))
            keep=max(1,int(round(len(selected)*run_fraction)))
            selected=[selected[i] for i in sorted(order[:keep])]
        if full_run and len(selected)!=expected:
            problems.append(f"{split}: expected {expected} scenario rows, found {len(selected)}")
        missing=[]
        for row in selected:
            p=resolve_manifest_data_path(row.get(key,''), manifest)
            if not p or not os.path.isfile(p):
                missing.append((row.get('scenario_id','unknown'),p))
        if missing:
            examples='; '.join(f"{sid}: {p}" for sid,p in missing[:5])
            problems.append(f"{split}: {len(missing)}/{len(selected)} missing {key}; examples: {examples}")
        print(f"[DATA] {split} {representation}: {len(selected)-len(missing)}/{len(selected)} {key} available")
        # Opening the first usable run verifies schema; the actual Dataset opens
        # every run when constructing its frame index in train.py.
        for row in selected:
            p=resolve_manifest_data_path(row.get(key,''), manifest)
            if p and os.path.isfile(p):
                try:
                    with h5py.File(p,'r') as h5:
                        for field in ('times','U','zb'):
                            if field not in h5:
                                raise KeyError(field)
                        if len(h5['times']) < 2:
                            raise ValueError('fewer than two saved frames')
                        if h5['U'].shape[1] != 7:
                            raise ValueError(f"expected 7 conservative channels; found {h5['U'].shape}")
                except Exception as e:
                    problems.append(f"{split} HDF5 schema invalid ({p}): {e}")
                break
    if problems:
        raise RuntimeError('Preflight FAILED:\n  ' + '\n  '.join(problems) + '\n  Restore the dataset runs/ directory or set MORINO_DATASET_ROOT to the dataset root containing runs/.')
    print('[DATA] Preflight passed; the study dataset and shared scales are unchanged.')

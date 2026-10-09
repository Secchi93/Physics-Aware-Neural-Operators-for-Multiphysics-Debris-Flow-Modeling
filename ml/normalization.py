from __future__ import annotations

"""Shared output scaling for FNO, GNO and GINO.

Zero is physically meaningful (dry cell / no bed change), so targets are scaled
without subtracting a mean.  Scales are area-weighted RMS values over wet cells
for the seven mobile conservative channels and over changed-bed cells for dzb.
"""

import csv, json, os
from typing import Dict, Iterable, List, Sequence
import h5py
import numpy as np

from pydebrisflow.constants import HF,HSU,HSL
from .data import resolve_manifest_data_path

TARGET_NAMES=["HF","HSU","HCU","HSL","HCL","MX","MY","DZB"]


def load_scales(path: str) -> np.ndarray:
    with open(path,"r",encoding="utf-8") as f: obj=json.load(f)
    return np.asarray(obj["target_scale"],dtype=np.float32)


def save_scales(path: str, scale: np.ndarray, meta: Dict[str,object]) -> None:
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"target_names":TARGET_NAMES,"target_scale":[float(v) for v in scale],**meta},f,indent=2)


def _mesh_area(dataset_root: str, profile_id: int) -> np.ndarray:
    p=os.path.join(dataset_root,"profiles",f"profile_{profile_id:02d}_adaptive_mesh.npz")
    with np.load(p) as d: return d["area"].astype(np.float64)


def compute_shared_scales(manifest: str, out_json: str, h_dry: float=1e-4, frame_stride: int=2, max_runs: int|None=None) -> np.ndarray:
    """Compute one normalization used by all representations.

    Cartesian and adaptive training runs both contribute, preventing a model
    from receiving a representation-specific normalization advantage.
    """
    root=os.path.dirname(os.path.abspath(manifest)); dataset_root=root
    with open(manifest,newline="",encoding="utf-8") as f:
        rows=[r for r in csv.DictReader(f) if r["split"]=="train"]
    if max_runs is not None: rows=rows[:int(max_runs)]
    ss=np.zeros(8,dtype=np.float64); ww=np.zeros(8,dtype=np.float64)
    for row in rows:
        pid=int(row["profile_id"])
        for rep,key in (("cartesian","cartesian_path"),("adaptive","adaptive_path")):
            path=resolve_manifest_data_path(row[key], manifest)
            if not os.path.isfile(path): continue
            with h5py.File(path,"r") as h5:
                times=h5["times"][:]
                if rep=="cartesian":
                    x=h5["x"][:]; y=h5["y"][:]; cell_area=float(abs(x[1]-x[0])*abs(y[1]-y[0])); area=None
                else:
                    area=_mesh_area(dataset_root,pid); cell_area=1.0
                z0=h5["zb"][0]
                for it in range(1,len(times),max(1,int(frame_stride))):
                    U=h5["U"][it].astype(np.float64); dz=h5["zb"][it].astype(np.float64)-z0
                    h=np.maximum(U[HF],0)+np.maximum(U[HSU],0)+np.maximum(U[HSL],0); wet=h>h_dry
                    if area is None:
                        w=np.full(h.shape,cell_area,dtype=np.float64)
                    else: w=area.reshape(h.shape)
                    for q in range(7):
                        m=wet
                        if np.any(m):
                            ss[q]+=float(np.sum((U[q][m]**2)*w[m])); ww[q]+=float(np.sum(w[m]))
                    m=np.abs(dz)>1e-8
                    if np.any(m): ss[7]+=float(np.sum((dz[m]**2)*w[m])); ww[7]+=float(np.sum(w[m]))
    scale=np.sqrt(ss/np.maximum(ww,1e-30))
    # Safe floors retain stable scaling if a process is disabled.
    floors=np.asarray([1e-3,1e-4,1e-5,1e-4,1e-5,1.0,1.0,1e-4],dtype=float)
    scale=np.maximum(scale,floors)
    save_scales(out_json,scale,{"h_dry":h_dry,"frame_stride":frame_stride,"n_training_runs":len(rows),"normalization":"area_weighted_rms_no_centering"})
    return scale

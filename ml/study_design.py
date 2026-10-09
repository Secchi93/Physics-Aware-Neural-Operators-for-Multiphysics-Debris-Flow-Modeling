from __future__ import annotations

"""Deterministic parameter-space design for the Morino FNO/GNO study."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import json
import math

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Range1D:
    lo: float
    hi: float
    def map(self, u: np.ndarray) -> np.ndarray:
        return self.lo + np.asarray(u) * (self.hi - self.lo)


@dataclass(frozen=True)
class ParameterSpace:
    mu_scale_full: Range1D
    xi_full: Range1D
    volume_full: Range1D
    mu_scale_train: Range1D
    xi_train: Range1D
    volume_train: Range1D


def _sobol(n: int, dim: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0,dim), dtype=np.float64)
    eng=qmc.Sobol(d=dim, scramble=True, seed=int(seed))
    # Generate the next Sobol power-of-two block and slice it. This preserves
    # the sequence balance properties without emitting SciPy warnings for the
    # non-power-of-two subset counts used by the study protocol.
    m=int(math.ceil(math.log2(max(n,1))))
    return eng.random_base2(m)[:n]


def _map_box(u: np.ndarray, ranges: Sequence[Range1D]) -> np.ndarray:
    out=np.empty_like(u,dtype=np.float64)
    for k,r in enumerate(ranges): out[:,k]=r.map(u[:,k])
    return out


def _outer_sample(u: np.ndarray, full: Range1D, inner: Range1D) -> np.ndarray:
    """Map u to the union [full.lo,inner.lo) U (inner.hi,full.hi]."""
    low=max(inner.lo-full.lo,0.0); high=max(full.hi-inner.hi,0.0); total=low+high
    if total <= 0.0:
        raise ValueError("OOD range is empty; inner range must be strictly inside full range")
    x=np.asarray(u,dtype=float); out=np.empty_like(x)
    cut=low/total
    m=x<cut
    if low>0:
        out[m]=full.lo + (x[m]/max(cut,1e-30))*low
    if high>0:
        out[~m]=inner.hi + ((x[~m]-cut)/max(1.0-cut,1e-30))*high
    return out


def make_profile_design(
    profile_id: int,
    space: ParameterSpace,
    counts: Dict[str,int],
    seed: int,
) -> List[Dict[str,object]]:
    """Create train/validation/IID/OOD scenarios for one topographic profile.

    Generalization is deliberately restricted to the physical parameters
    mu-scale, xi and initial volume.  Every topographic profile is represented
    in every subset.
    """
    out: List[Dict[str,object]]=[]
    inner=(space.mu_scale_train,space.xi_train,space.volume_train)
    full=(space.mu_scale_full,space.xi_full,space.volume_full)
    split_order=["train","validation","iid_test","single_ood_test","joint_ood_test"]
    counter=0
    for si,split in enumerate(split_order):
        n=int(counts.get(split,0)); u=_sobol(n,3,seed+1009*profile_id+97*si)
        if split in ("train","validation","iid_test"):
            vals=_map_box(u,inner); axes=["none"]*n
        elif split=="single_ood_test":
            vals=_map_box(u,inner); axes=[]
            for i in range(n):
                # Offset by profile so the global 6-profile design remains
                # balanced across mu, xi and volume even when n is not a
                # multiple of three (production uses 5 cases/profile).
                ax=(i + int(profile_id) - 1) % 3
                axes.append(("mu_scale","xi","volume_m3")[ax])
                vals[i,ax]=_outer_sample(np.asarray([u[i,ax]]),full[ax],inner[ax])[0]
        else:
            vals=np.empty((n,3),dtype=float); axes=["joint"]*n
            for ax in range(3): vals[:,ax]=_outer_sample(u[:,ax],full[ax],inner[ax])
        for i in range(n):
            out.append({
                "scenario_id":f"p{profile_id:02d}_s{counter:04d}",
                "profile_id":int(profile_id),"split":split,"ood_axis":axes[i],
                "mu_scale":float(vals[i,0]),"xi":float(vals[i,1]),"volume_m3":float(vals[i,2]),
            })
            counter+=1
    return out


def make_study_design(n_profiles: int, space: ParameterSpace, counts: Dict[str,int], seed: int=1234) -> List[Dict[str,object]]:
    rows=[]
    for p in range(1,int(n_profiles)+1): rows.extend(make_profile_design(p,space,counts,seed))
    return rows


def save_jsonl(rows: Sequence[Dict[str,object]], path: str) -> None:
    with open(path,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+"\n")

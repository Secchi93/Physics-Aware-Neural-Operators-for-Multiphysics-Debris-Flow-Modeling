from __future__ import annotations

"""Lazy HDF5 datasets for the controlled FNO/GNO/GINO comparison."""

import csv
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .rasterize import RasterMap, apply_raster_map, build_raster_map


FEATURE_NAMES=["U0_HF","U0_HSU","U0_HCU","U0_HSL","U0_HCL","U0_MX","U0_MY",
               "z_relative","slope_x","slope_y","mu_scale","xi_scaled","time_scaled"]
TARGET_NAMES=["HF","HSU","HCU","HSL","HCL","MX","MY","DZB"]


class _TinyCache:
    def __init__(self,max_items=6): self.max_items=max_items; self.d=OrderedDict()
    def get(self,k):
        if k not in self.d: return None
        v=self.d.pop(k); self.d[k]=v; return v
    def put(self,k,v):
        if k in self.d: self.d.pop(k)
        self.d[k]=v
        while len(self.d)>self.max_items: self.d.popitem(last=False)


def resolve_manifest_data_path(raw: str, manifest: str) -> str:
    """Resolve a manifest HDF5 path portably across machines/OSes.

    New manifests use paths relative to the dataset directory. For backward
    compatibility, legacy absolute Windows paths are also mapped by preserving
    their ``runs/...`` tail when the original location no longer exists.
    """
    raw = str(raw or "").strip()
    if not raw:
        return raw
    if os.path.isfile(raw):
        return os.path.abspath(raw)
    root = os.path.dirname(os.path.abspath(manifest))
    normalized = raw.replace("\\", "/")
    if not os.path.isabs(raw):
        candidate = os.path.join(root, *normalized.split("/"))
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        external_root = os.environ.get("MORINO_DATASET_ROOT", "").strip()
        if external_root:
            external = os.path.join(os.path.abspath(external_root), *normalized.split("/"))
            if os.path.isfile(external):
                return external
        if normalized.startswith("runs/"):
            # Return the canonical in-repository location even if the large HDF5
            # files are intentionally omitted; preflight will then report the
            # exact missing file and mention MORINO_DATASET_ROOT.
            return os.path.abspath(candidate)
    low = normalized.lower()
    token = "/runs/"
    if token in low:
        tail = normalized[low.index(token) + 1:]
        return os.path.abspath(os.path.join(root, *tail.split("/")))
    return os.path.abspath(raw) if os.path.isabs(raw) else os.path.abspath(os.path.join(root, raw))


def _relative_topography(z: np.ndarray) -> np.ndarray:
    zz=np.asarray(z,dtype=np.float32); zmin=float(np.nanmin(zz)); zmax=float(np.nanmax(zz))
    return ((zz-zmin)/max(zmax-zmin,1e-6)).astype(np.float32)


def _slope_regular(z: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    dx=float(abs(x[1]-x[0])); dy=float(abs(y[1]-y[0])); gy,gx=np.gradient(np.asarray(z,dtype=np.float64),dy,dx)
    return gx.astype(np.float32),gy.astype(np.float32)


def _feature_stack(U0: np.ndarray,zrel: np.ndarray,sx: np.ndarray,sy: np.ndarray,mu_scale: float,xi: float,t: float,t_end: float,target_scale: np.ndarray,point_last: bool) -> np.ndarray:
    # Preserve zero=dry by scale-only normalization of the initial conservative state.
    u0=U0.astype(np.float32)/target_scale[:7].reshape((7,)+((1,)*(U0.ndim-1)))
    slope_scale=3.0
    scx=np.clip(sx/slope_scale,-1.0,1.0).astype(np.float32); scy=np.clip(sy/slope_scale,-1.0,1.0).astype(np.float32)
    mu=np.full_like(zrel,(float(mu_scale)-1.0)/0.5,dtype=np.float32)
    xiv=np.full_like(zrel,(float(xi)-200.0)/100.0,dtype=np.float32)
    tv=np.full_like(zrel,2.0*float(t)/max(float(t_end),1e-12)-1.0,dtype=np.float32)
    arr=np.concatenate([u0,zrel[None],scx[None],scy[None],mu[None],xiv[None],tv[None]],axis=0)
    if point_last:
        arr=np.moveaxis(arr,0,-1)
    return arr.astype(np.float32)


def _target(U: np.ndarray,zt: np.ndarray,z0: np.ndarray,scale: np.ndarray,point_last: bool) -> np.ndarray:
    y=np.concatenate([U.astype(np.float32),(zt-z0).astype(np.float32)[None]],axis=0)
    y=y/scale.reshape((8,)+((1,)*(U.ndim-1)))
    if point_last: y=np.moveaxis(y,0,-1)
    return y.astype(np.float32)


@dataclass(frozen=True)
class FrameRef:
    row_index: int
    time_index: int


class MorinoFrameDataset(Dataset):
    """Frame-wise operator dataset.

    representation:
      - ``cartesian``: native Cartesian PyDebrisFlow2D -> FNO
      - ``cartesian_points``: same native Cartesian solution exposed as points -> GNO/GINO
      - ``adaptive``: native terrain-adapted cells -> GNO/GINO
      - ``adaptive_rasterized``: adaptive solver conservatively remapped -> FNO
    """
    def __init__(self,manifest: str,split: str,representation: str,target_scale: Sequence[float],frame_stride: int=1,
                 include_t0: bool=False,run_fraction: float=1.0,seed: int=1234,profiles: Optional[Sequence[int]]=None,
                 raster_dx: float=10.0,cache_static_items: int=6):
        self.manifest=os.path.abspath(manifest); self.root=os.path.dirname(self.manifest); self.split=split; self.representation=representation
        self.scale=np.asarray(target_scale,dtype=np.float32); self.raster_dx=float(raster_dx); self.cache=_TinyCache(max(1, int(cache_static_items))); self.rmaps: Dict[int,RasterMap]={}
        with open(self.manifest,newline="",encoding="utf-8") as f: rows=[r for r in csv.DictReader(f) if r["split"]==split]
        if profiles is not None:
            allowed=set(int(p) for p in profiles); rows=[r for r in rows if int(r["profile_id"]) in allowed]
        # Resolve data locations once so every downstream consumer sees portable
        # paths independent of the machine that generated the manifest.
        for row in rows:
            row["cartesian_path"] = resolve_manifest_data_path(row.get("cartesian_path", ""), self.manifest)
            row["adaptive_path"] = resolve_manifest_data_path(row.get("adaptive_path", ""), self.manifest)
        # Learning-curve subsets are selected at run level and identically for representations.
        if run_fraction < 1.0:
            rng=np.random.default_rng(seed); order=rng.permutation(len(rows)); keep=max(1,int(round(len(rows)*run_fraction))); rows=[rows[i] for i in sorted(order[:keep])]
        self.rows=rows; self.frames: List[FrameRef]=[]; self.row_spatial_shapes: Dict[int, Tuple[int, ...]] = {}
        key="cartesian_path" if representation in ("cartesian","cartesian_points") else "adaptive_path"
        for ir,row in enumerate(rows):
            path=row[key]
            if not os.path.isfile(path): continue
            with h5py.File(path,"r") as h5:
                nt=len(h5["times"])
                native_shape=tuple(int(v) for v in h5["zb"].shape[1:])
            if representation=="adaptive_rasterized":
                self.row_spatial_shapes[ir]=tuple(int(v) for v in self._rmap(int(row["profile_id"])).shape)
            else:
                self.row_spatial_shapes[ir]=native_shape
            start=0 if include_t0 else 1
            for it in range(start,nt,max(1,int(frame_stride))): self.frames.append(FrameRef(ir,it))
        if not self.frames: raise RuntimeError(f"no frames found for split={split}, representation={representation}")

    def __len__(self): return len(self.frames)

    def _mesh_path(self,pid: int) -> str:
        return os.path.join(self.root,"profiles",f"profile_{pid:02d}_adaptive_mesh.npz")

    def _rmap(self,pid: int) -> RasterMap:
        if pid not in self.rmaps: self.rmaps[pid]=build_raster_map(self._mesh_path(pid),self.raster_dx)
        return self.rmaps[pid]

    def _static_cart(self,path: str):
        k=("c",path); got=self.cache.get(k)
        if got is not None: return got
        with h5py.File(path,"r") as h5:
            U0=h5["U"][0].astype(np.float32); z0=h5["zb"][0].astype(np.float32); x=h5["x"][:].astype(np.float32); y=h5["y"][:].astype(np.float32); t_end=float(h5["times"][-1])
        sx,sy=_slope_regular(z0,x,y); area=np.full(z0.shape,float(abs(x[1]-x[0])*abs(y[1]-y[0])),dtype=np.float32)
        got=(U0,z0,_relative_topography(z0),sx,sy,area,t_end,x,y); self.cache.put(k,got); return got

    def _static_adapt(self,path: str,pid: int):
        k=("a",path); got=self.cache.get(k)
        if got is not None: return got
        with h5py.File(path,"r") as h5:
            U0=h5["U"][0].astype(np.float32); z0=h5["zb"][0].astype(np.float32); t_end=float(h5["times"][-1])
        run_dir=os.path.dirname(path); sf=os.path.join(run_dir,"static_fields.npz")
        with np.load(sf) as d: sx=d["slope_x"].astype(np.float32); sy=d["slope_y"].astype(np.float32); coords=d["xy_normalized"].astype(np.float32)
        with np.load(self._mesh_path(pid)) as m: area=m["area"].astype(np.float32); coords_m=np.column_stack((m["x_center"],m["y_center"])).astype(np.float32)
        got=(U0,z0,_relative_topography(z0),sx,sy,area,coords,t_end,coords_m); self.cache.put(k,got); return got

    def __getitem__(self,index: int):
        ref=self.frames[index]; row=self.rows[ref.row_index]; pid=int(row["profile_id"]); mu=float(row["mu_scale"]); xi=float(row["xi"])
        if self.representation in ("cartesian","cartesian_points"):
            path=row["cartesian_path"]; U0,z0,zrel,sx,sy,area,t_end,xcoord,ycoord=self._static_cart(path)
            with h5py.File(path,"r") as h5: t=float(h5["times"][ref.time_index]); U=h5["U"][ref.time_index].astype(np.float32); zt=h5["zb"][ref.time_index].astype(np.float32)
            if self.representation=="cartesian":
                x=_feature_stack(U0,zrel,sx,sy,mu,xi,t,t_end,self.scale,False); y=_target(U,zt,z0,self.scale,False)
                return {"x":torch.from_numpy(x),"y":torch.from_numpy(y),"area":torch.from_numpy(area),"time":torch.tensor(t,dtype=torch.float32),
                        "profile_id":pid,"scenario_id":row["scenario_id"],"time_index":ref.time_index,"x_coord":torch.from_numpy(xcoord),"y_coord":torch.from_numpy(ycoord)}
            # Exact same Cartesian target, represented as a coordinate point cloud for GNO.
            feat=_feature_stack(U0,zrel,sx,sy,mu,xi,t,t_end,self.scale,False).reshape(13,-1).T
            targ=_target(U,zt,z0,self.scale,False).reshape(8,-1).T
            X,Y=np.meshgrid(xcoord,ycoord)
            xm=np.column_stack((X.ravel(),Y.ravel())).astype(np.float32)
            xn=(xm[:,0]-float(xcoord.min()))/max(float(xcoord.max()-xcoord.min()),1e-12)
            yn=(xm[:,1]-float(ycoord.min()))/max(float(ycoord.max()-ycoord.min()),1e-12)
            coords=np.column_stack((xn,yn)).astype(np.float32)
            return {"x":torch.from_numpy(feat),"y":torch.from_numpy(targ),"area":torch.from_numpy(area.reshape(-1)),
                    "coords":torch.from_numpy(coords),"coords_m":torch.from_numpy(xm),"time":torch.tensor(t,dtype=torch.float32),
                    "profile_id":pid,"scenario_id":row["scenario_id"],"time_index":ref.time_index}
        path=row["adaptive_path"]; U0,z0,zrel,sx,sy,area,coords,t_end,coords_m=self._static_adapt(path,pid)
        with h5py.File(path,"r") as h5: t=float(h5["times"][ref.time_index]); U=h5["U"][ref.time_index].astype(np.float32); zt=h5["zb"][ref.time_index].astype(np.float32)
        if self.representation=="adaptive":
            x=_feature_stack(U0,zrel,sx,sy,mu,xi,t,t_end,self.scale,True); y=_target(U,zt,z0,self.scale,True)
            return {"x":torch.from_numpy(x),"y":torch.from_numpy(y),"area":torch.from_numpy(area),"coords":torch.from_numpy(coords),"coords_m":torch.from_numpy(coords_m),
                    "time":torch.tensor(t,dtype=torch.float32),"profile_id":pid,"scenario_id":row["scenario_id"],"time_index":ref.time_index}
        if self.representation=="adaptive_rasterized":
            rm=self._rmap(pid); U0r=apply_raster_map(rm,U0); z0r=apply_raster_map(rm,z0); Ur=apply_raster_map(rm,U); ztr=apply_raster_map(rm,zt)
            sxr=apply_raster_map(rm,sx); syr=apply_raster_map(rm,sy); zrelr=_relative_topography(z0r)
            ar=(rm.coverage.reshape(rm.shape)*rm.target_area).astype(np.float32)
            x=_feature_stack(U0r,zrelr,sxr,syr,mu,xi,t,t_end,self.scale,False); y=_target(Ur,ztr,z0r,self.scale,False)
            return {"x":torch.from_numpy(x),"y":torch.from_numpy(y),"area":torch.from_numpy(ar),"time":torch.tensor(t,dtype=torch.float32),
                    "profile_id":pid,"scenario_id":row["scenario_id"],"time_index":ref.time_index,"x_coord":torch.from_numpy(rm.x),"y_coord":torch.from_numpy(rm.y)}
        raise ValueError(self.representation)

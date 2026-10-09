from __future__ import annotations

"""Build paired Cartesian / terrain-adapted Morino simulations.

The script is restartable.  A completed run is represented by ``run_data.h5``
plus ``meta.json``.  Raw per-time snapshots are compacted and removed by
default to keep disk use reasonable on a laptop.
"""

import argparse
import copy
import csv
import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Persist Numba CUDA kernels between Python processes. The first execution
# after installing/updating the package still compiles once; later commands
# reuse the project-local cache.
_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(_PROJECT_ROOT, ".numba_cache"))
os.environ.setdefault("NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", "0")

import numpy as np
import yaml
import h5py

from pydebrisflow.config import SolverConfig
from pydebrisflow.simulation import run_solver
from pydebrisflow.dem import load_or_build_dem
from pydebrisflow.adaptive_mesh import AdaptiveMeshConfig, build_terrain_adapted_mesh, save_mesh_npz
from pydebrisflow.adaptive_solver import run_solver_adaptive
from pydebrisflow.run_visualization import save_final_depth_cartesian, save_final_depth_adaptive
from ml.study_design import ParameterSpace, Range1D, make_study_design, save_jsonl


def _load_yaml(path: str) -> Dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f) or {}


def _abs(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base,p))


def _load_geometry(path: str, n_profiles: int) -> Dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f: g=json.load(f)
    prof=g.get("profiles",[])
    if len(prof) < n_profiles: raise RuntimeError(f"geometry has {len(prof)} profiles; expected {n_profiles}")
    return {"profiles":prof[:n_profiles]}




def _trim_to_valid_rectangular_support(x: np.ndarray, y: np.ndarray, z: np.ndarray, valid: np.ndarray):
    """Trim fully invalid outer DEM rows/columns before adaptive meshing.

    The Cartesian solver already excludes invalid DEM cells through its validity
    mask. The terrain-adapted mesh must cover the same active footprint;
    otherwise bed inventories and domain area differ even when the release is
    identical. For the Morino crops, NoData is expected only on outer border
    rows/columns. Internal NoData holes are rejected explicitly rather than
    silently meshed as active terrain.
    """
    x=np.asarray(x); y=np.asarray(y); z=np.asarray(z); valid=np.asarray(valid,dtype=bool)
    if valid.shape != z.shape or z.shape != (y.size, x.size):
        raise ValueError("x/y/z/valid shapes are inconsistent")
    if not np.any(valid):
        raise RuntimeError("DEM crop contains no valid cells")
    jj,ii=np.where(valid)
    j0,j1=int(jj.min()),int(jj.max())+1
    i0,i1=int(ii.min()),int(ii.max())+1
    xt=x[i0:i1].copy(); yt=y[j0:j1].copy(); zt=z[j0:j1,i0:i1].copy(); vt=valid[j0:j1,i0:i1].copy()
    if not np.all(vt):
        nbad=int((~vt).sum())
        raise RuntimeError(
            f"Adaptive mesh requires rectangular valid DEM support after border trimming; "
            f"found {nbad} internal NoData cells. Resolve the DEM mask before production."
        )
    removed=int(valid.size-vt.size)
    if removed:
        print(f"[ADAPT DOMAIN] trimmed {removed} outer NoData cells; active DEM={vt.shape[1]}x{vt.shape[0]}")
    return xt,yt,zt,vt


def _assert_adaptive_area_matches_cartesian_valid(mesh, x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> None:
    """Fail fast if adaptive and Cartesian domains do not have the same area."""
    dx=float(abs(x[1]-x[0])); dy=float(abs(y[1]-y[0]))
    expected=float(np.asarray(valid,dtype=bool).sum())*dx*dy
    actual=float(np.asarray(mesh.arrays()["area"],dtype=float).sum())
    tol=max(1.0e-6,1.0e-10*max(expected,1.0))
    if abs(actual-expected)>tol:
        raise RuntimeError(
            f"Adaptive mesh area mismatch: mesh={actual:.12g} m2, Cartesian valid={expected:.12g} m2, "
            f"difference={actual-expected:.12g} m2"
        )
    print(f"[ADAPT DOMAIN] area match OK: {actual:.3f} m2")

def _solver_cfg(study: Dict[str,Any], profile: Dict[str,Any], scenario: Dict[str,Any], dem_path: str,
                out_dir: str, backend: str, cache_path: str) -> SolverConfig:
    cfg=SolverConfig(); s=study.get("solver",{})
    bbox=profile["clip_bbox"]
    cfg.compute.backend=backend; cfg.compute.cuda_allow_fallback=True
    cfg.grid.dem_path=dem_path; cfg.grid.zip_member=str(study["paths"].get("zip_member","DEM/DemMosaic_WGS84_33N.asc"))
    cfg.grid.cache_path=cache_path; cfg.grid.target_dx=float(s.get("cartesian_dx_m",10.0)); cfg.grid.clip_enabled=True
    cfg.grid.xmin,cfg.grid.xmax,cfg.grid.ymin,cfg.grid.ymax=[float(v) for v in bbox]
    cfg.numerics.cfl=float(s.get("cfl",0.20)); cfg.numerics.dt_max=float(s.get("dt_max",0.25))
    cfg.numerics.t_end=float(s.get("t_end_s",1000.0)); cfg.numerics.output_dt=float(s.get("output_dt_s",20.0))
    cfg.numerics.h_dry=float(s.get("h_dry",1e-4)); cfg.numerics.space_order=int(s.get("space_order",1)); cfg.numerics.time_order=int(s.get("time_order",1))
    cfg.numerics.flux=str(s.get("flux","hllc")); cfg.numerics.bc_left=str(s.get("bc_left","outflow")); cfg.numerics.bc_right=str(s.get("bc_right","outflow"))
    cfg.numerics.bc_bottom=str(s.get("bc_bottom","outflow")); cfg.numerics.bc_top=str(s.get("bc_top","outflow"))
    cfg.release.enabled=True; cfg.release.mode="polygon"; cfg.release.polygon=profile["release_polygon"]; cfg.release.volume_m3=float(scenario["volume_m3"])
    cfg.release.initial_speed_ms=float(s.get("release_initial_speed_ms",0.0))
    ms=float(scenario["mu_scale"]); xi=float(scenario["xi"])
    mat=study.get("material",{})
    cfg.material.rho_fluid=float(mat.get("rho_fluid",cfg.material.rho_fluid)); cfg.material.rho_solid=float(mat.get("rho_solid",cfg.material.rho_solid))
    cfg.material.initial_solid_fraction=float(mat.get("initial_solid_fraction",cfg.material.initial_solid_fraction))
    cfg.material.initial_coarse_fraction=float(mat.get("initial_coarse_fraction",cfg.material.initial_coarse_fraction))
    cfg.material.initial_upper_solid_fraction=float(mat.get("initial_upper_solid_fraction",cfg.material.initial_upper_solid_fraction))
    cfg.material.max_solid_fraction=float(mat.get("max_solid_fraction",cfg.material.max_solid_fraction))
    # Reference Morino zonation scaled coherently: 0.01 upstream / 0.10 downstream.
    cfg.material.mu_fine=0.01*ms; cfg.material.mu_coarse=0.10*ms; cfg.material.xi_fine=xi; cfg.material.xi_coarse=xi
    cfg.friction.regions=[
        {"polygon":profile["friction_poly1"],"mu":0.01*ms,"xi":xi,"name":"morino_region_1"},
        {"polygon":profile["friction_poly2"],"mu":0.10*ms,"xi":xi,"name":"morino_region_2"},
    ]
    phys=study.get("physics",{})
    cfg.erosion.enabled=bool(phys.get("erosion",True)); cfg.deposition.enabled=bool(phys.get("deposition",True)); cfg.segregation.enabled=bool(phys.get("segregation",True))
    cfg.erosion.update_bed=bool(phys.get("update_bed",True))
    # Every non-varied physical parameter is written explicitly into the study YAML
    # so the paper can distinguish fixed assumptions from the three OOD variables.
    cfg.erosion.erodible_depth_m=float(phys.get("erodible_depth_m",cfg.erosion.erodible_depth_m))
    cfg.erosion.bed_porosity=float(phys.get("bed_porosity",cfg.erosion.bed_porosity))
    cfg.erosion.bed_coarse_fraction=float(phys.get("bed_coarse_fraction",cfg.erosion.bed_coarse_fraction))
    cfg.erosion.critical_shear_pa=float(phys.get("erosion_critical_shear_pa",cfg.erosion.critical_shear_pa))
    cfg.erosion.excess_shear_rate_ms=float(phys.get("excess_shear_rate_ms",cfg.erosion.excess_shear_rate_ms))
    cfg.erosion.excess_shear_exponent=float(phys.get("excess_shear_exponent",cfg.erosion.excess_shear_exponent))
    cfg.erosion.max_erosion_rate_ms=float(phys.get("max_erosion_rate_ms",cfg.erosion.max_erosion_rate_ms))
    cfg.deposition.critical_shear_pa=float(phys.get("deposition_critical_shear_pa",cfg.deposition.critical_shear_pa))
    cfg.deposition.hindered_settling_exponent=float(phys.get("hindered_settling_exponent",cfg.deposition.hindered_settling_exponent))
    cfg.deposition.max_rate_ms=float(phys.get("max_deposition_rate_ms",cfg.deposition.max_rate_ms))
    cfg.deposition.minimum_mobile_depth_m=float(phys.get("minimum_mobile_depth_m",cfg.deposition.minimum_mobile_depth_m))
    cfg.segregation.segregation_length_m=float(phys.get("segregation_length_m",cfg.segregation.segregation_length_m))
    cfg.segregation.max_segregation_speed_ms=float(phys.get("max_segregation_speed_ms",cfg.segregation.max_segregation_speed_ms))
    cfg.segregation.remix_diffusivity_m2s=float(phys.get("remix_diffusivity_m2s",cfg.segregation.remix_diffusivity_m2s))
    cfg.segregation.min_layer_thickness_m=float(phys.get("min_layer_thickness_m",cfg.segregation.min_layer_thickness_m))
    cfg.output.out_dir=out_dir; cfg.output.save_snapshots=True; cfg.output.make_png=False; cfg.output.make_gif=False; cfg.output.save_setup_preview=False
    cfg.output.progress_every_steps=int(s.get("progress_every_steps",500))
    return cfg


def _adaptive_cfg(study: Dict[str,Any]) -> AdaptiveMeshConfig:
    a=study.get("adaptive_mesh",{})
    return AdaptiveMeshConfig(
        grid_unit=float(a.get("grid_unit_m",2.5)),
        dx_levels=tuple(float(v) for v in a.get("dx_levels_m",[15.0,12.5,10.0])),
        refine_slope=float(a.get("refine_slope",0.60)),
        refine_relief_m=float(a.get("refine_relief_m",18.0)),
        refine_release=bool(a.get("refine_release",True)),
        release_buffer_m=float(a.get("release_buffer_m",0.0)),
        edge_decimals=int(a.get("edge_decimals",8)),
        aggregation_mode=str(a.get("aggregation_mode","patch")),
        patch_size_m=float(a.get("patch_size_m",150.0)),
        patch_quantile_adaptation=bool(a.get("patch_quantile_adaptation",False)),
        coarse_patch_fraction=float(a.get("coarse_patch_fraction",0.40)),
        medium_patch_fraction=float(a.get("medium_patch_fraction",0.30)),
    )


def _cache_file(cache_root: str, import_roots: List[str], filename: str) -> str:
    """Return a writable cache path without modifying an existing legacy cache.

    If the new cache does not exist, a same-named legacy cache is copied into
    the new cache root. ``load_or_build_dem`` then validates its signature and
    can rebuild only the copied destination if necessary. The original cache is
    therefore never overwritten.
    """
    os.makedirs(cache_root,exist_ok=True)
    dst=os.path.join(cache_root,filename)
    if not os.path.isfile(dst):
        for root in import_roots:
            src=os.path.join(root,filename)
            if os.path.isfile(src):
                shutil.copy2(src,dst)
                print(f"[CACHE] imported existing DEM cache read-only source: {src}")
                break
    return dst


def _mesh_summary(mesh, adapt_cfg: AdaptiveMeshConfig) -> Dict[str,Any]:
    a=mesh.arrays(); dx=np.asarray(a["dx"],dtype=float); dy=np.asarray(a["dy"],dtype=float)
    area=np.asarray(a["area"],dtype=float); pref=np.asarray(a["preferred_dx"],dtype=float)
    def hist(vals):
        out={}
        for v in sorted(np.unique(np.round(vals,8))): out[f"{float(v):g}"]=int(np.sum(np.isclose(vals,v)))
        return out
    return {
        "aggregation_mode":adapt_cfg.aggregation_mode,
        "patch_size_m":float(adapt_cfg.patch_size_m),
        "patch_quantile_adaptation":bool(adapt_cfg.patch_quantile_adaptation),
        "target_patch_fractions":{
            "15m_coarse":float(adapt_cfg.coarse_patch_fraction),
            "12.5m_medium":float(adapt_cfg.medium_patch_fraction),
            "10m_fine":float(1.0-adapt_cfg.coarse_patch_fraction-adapt_cfg.medium_patch_fraction),
        },
        "requested_dx_levels_m":[float(v) for v in adapt_cfg.dx_levels],
        "n_cells":int(len(mesh.cells)),"n_faces":int(len(mesh.faces)),
        "dx_histogram":hist(dx),"dy_histogram":hist(dy),"preferred_dx_histogram":hist(pref),
        "min_dx_m":float(dx.min()),"max_dx_m":float(dx.max()),
        "min_dy_m":float(dy.min()),"max_dy_m":float(dy.max()),
        "total_area_m2":float(area.sum()),
    }


def _compact(run_dir: str, mode: str, remove_raw: bool=True) -> str:
    """Compact snapshots to a time-chunked HDF5 file for lazy ML loading."""
    files=sorted(glob.glob(os.path.join(run_dir,"state_*.npz")))
    if not files: raise RuntimeError(f"no snapshots in {run_dir}")
    with np.load(files[0]) as d0:
        ushape=d0["U"].shape; zshape=d0["zb"].shape
        has_xy=("x" in d0 and "y" in d0); has_bed=("bed_fine_solid" in d0)
        x=d0["x"].astype(np.float32) if has_xy else None; y=d0["y"].astype(np.float32) if has_xy else None
    out=os.path.join(run_dir,"run_data.h5")
    with h5py.File(out,"w") as h5:
        T=len(files)
        h5.create_dataset("times",shape=(T,),dtype="f4")
        h5.create_dataset("U",shape=(T,*ushape),dtype="f4",chunks=(1,*ushape),compression="lzf")
        h5.create_dataset("zb",shape=(T,*zshape),dtype="f4",chunks=(1,*zshape),compression="lzf")
        if has_bed:
            h5.create_dataset("bed_fine_solid",shape=(T,*zshape),dtype="f4",chunks=(1,*zshape),compression="lzf")
            h5.create_dataset("bed_coarse_solid",shape=(T,*zshape),dtype="f4",chunks=(1,*zshape),compression="lzf")
        if has_xy:
            h5.create_dataset("x",data=x); h5.create_dataset("y",data=y)
        for it,pth in enumerate(files):
            with np.load(pth) as d:
                h5["times"][it]=float(d["t"]); h5["U"][it]=d["U"].astype(np.float32); h5["zb"][it]=d["zb"].astype(np.float32)
                if has_bed:
                    h5["bed_fine_solid"][it]=d["bed_fine_solid"].astype(np.float32); h5["bed_coarse_solid"][it]=d["bed_coarse_solid"].astype(np.float32)
        h5.attrs["mode"]=mode
    if remove_raw:
        for pth in files: os.remove(pth)
    return out


def _write_meta(run_dir: str, row: Dict[str,Any], mode: str, data_path: str, metrics: Dict[str,Any]) -> None:
    obj=dict(row); obj.update({"mode":mode,"data_path":os.path.abspath(data_path),"metrics_path":os.path.abspath(os.path.join(run_dir,"metrics.json")),"final_image_path":os.path.abspath(os.path.join(run_dir,"final_run.png")),"metrics":metrics})
    with open(os.path.join(run_dir,"meta.json"),"w",encoding="utf-8") as f: json.dump(obj,f,indent=2)


def _completed(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir,"run_data.h5")) and os.path.isfile(os.path.join(run_dir,"meta.json"))


def _manifest(root: str, rows: List[Dict[str,Any]]) -> None:
    path=os.path.join(root,"manifest.csv")
    fields=["scenario_id","profile_id","split","ood_axis","mu_scale","xi","volume_m3","cartesian_path","adaptive_path"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            sid=r["scenario_id"]; base=os.path.join(root,"runs",sid)
            w.writerow({**{k:r.get(k,"") for k in fields[:7]},"cartesian_path":os.path.abspath(os.path.join(base,"cartesian","run_data.h5")),
                        "adaptive_path":os.path.abspath(os.path.join(base,"adaptive","run_data.h5"))})


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True,help="site-specific dataset-generation YAML (not bundled with the ML source archive)"); ap.add_argument("--dem",default=None)
    ap.add_argument("--design-only",action="store_true"); ap.add_argument("--limit",type=int,default=None); ap.add_argument("--profile",type=int,default=None)
    ap.add_argument("--mode",choices=["both","cartesian","adaptive"],default="both"); ap.add_argument("--keep-raw",action="store_true")
    ap.add_argument("--mesh-only",action="store_true",help="Build/save adaptive profile mesh summaries without running the solver")
    args=ap.parse_args(); cfgp=os.path.abspath(args.config); base=os.path.dirname(cfgp); study=_load_yaml(cfgp)
    print(f"[CUDA CACHE] persistent Numba cache: {os.environ.get('NUMBA_CACHE_DIR')}")
    print("[CUDA CACHE] first run after a code update may JIT-compile for several minutes; later runs reuse the cache.")
    paths=study["paths"]; root=_abs(base,paths.get("dataset_root","../output/dataset")); os.makedirs(root,exist_ok=True)
    cache_root=_abs(base,paths.get("solver_cache_root",os.path.join(root,"cache")))
    cache_import_roots=[_abs(base,p) for p in paths.get("cache_import_roots",[]) or []]
    geom_path=_abs(base,paths.get("geometry_cache","../output/geometry_morino_6profiles.json")); nprof=int(study.get("n_profiles",6)); geom=_load_geometry(geom_path,nprof)
    dem_path=os.path.abspath(args.dem) if args.dem else _abs(base,paths["dem_path"])
    ps=study["parameter_space"]
    space=ParameterSpace(Range1D(*map(float,ps["mu_scale_full"])),Range1D(*map(float,ps["xi_full"])),Range1D(*map(float,ps["volume_full_m3"])),
                         Range1D(*map(float,ps["mu_scale_train"])),Range1D(*map(float,ps["xi_train"])),Range1D(*map(float,ps["volume_train_m3"])))
    rows=make_study_design(nprof,space,{k:int(v) for k,v in study["counts_per_profile"].items()},int(study.get("seed",1234)))
    save_jsonl(rows,os.path.join(root,"design.jsonl")); _manifest(root,rows)
    if args.design_only: print(f"Wrote {len(rows)} scenarios to {root}"); return
    adapt_cfg=_adaptive_cfg(study)
    if args.mesh_only:
        selected=[int(args.profile)] if args.profile is not None else list(range(1,nprof+1))
        prof_dir=os.path.join(root,"profiles"); os.makedirs(prof_dir,exist_ok=True)
        for p in selected:
            if p < 1 or p > nprof:
                raise RuntimeError(f"profile must be in 1..{nprof}")
            prof=geom["profiles"][p-1]
            row=next(r for r in rows if int(r["profile_id"])==p)
            preview_dir=os.path.join(root,"mesh_preview",f"profile_{p:02d}")
            scfg=_solver_cfg(study,prof,row,dem_path,preview_dir,"cuda",_cache_file(cache_root,cache_import_roots,f"p{p:02d}_adaptive_dem.npz"))
            x,y,z,valid=load_or_build_dem(scfg.grid)
            x,y,z,valid=_trim_to_valid_rectangular_support(x,y,z,valid)
            zf=np.where(valid,z,np.nanmax(z[valid])+scfg.grid.nodata_barrier_m)
            mesh=build_terrain_adapted_mesh(x,y,zf,adapt_cfg,[prof["release_polygon"]])
            _assert_adaptive_area_matches_cartesian_valid(mesh,x,y,valid)
            save_mesh_npz(mesh,os.path.join(prof_dir,f"profile_{p:02d}_adaptive_mesh.npz"))
            summary=_mesh_summary(mesh,adapt_cfg)
            with open(os.path.join(prof_dir,f"profile_{p:02d}_adaptive_mesh_summary.json"),"w",encoding="utf-8") as f:
                json.dump(summary,f,indent=2)
            print(f"[MESH] profile {p:02d}: cells={summary['n_cells']} faces={summary['n_faces']} dx={summary['dx_histogram']}")
        print(f"[MESH] summaries written to {prof_dir}")
        return
    if args.profile is not None: rows=[r for r in rows if int(r["profile_id"])==args.profile]
    if args.limit is not None: rows=rows[:args.limit]
    mesh_cache: Dict[int,Any]={}
    for idx,row in enumerate(rows,1):
        p=int(row["profile_id"]); prof=geom["profiles"][p-1]; sid=str(row["scenario_id"]); base_run=os.path.join(root,"runs",sid)
        print(f"\n=== {idx}/{len(rows)} {sid} split={row['split']} mu_scale={row['mu_scale']:.4g} xi={row['xi']:.4g} V={row['volume_m3']:.4g} ===")
        if args.mode in ("both","cartesian"):
            rd=os.path.join(base_run,"cartesian")
            if not _completed(rd):
                os.makedirs(rd,exist_ok=True); scfg=_solver_cfg(study,prof,row,dem_path,rd,"cuda",_cache_file(cache_root,cache_import_roots,f"p{p:02d}_cartesian.npz"))
                metrics=run_solver(scfg)
                save_final_depth_cartesian(
                    rd, scfg,
                    title=f"{sid} | Cartesian dx={scfg.grid.target_dx:g} m | mu_scale={row['mu_scale']:.3g}, xi={row['xi']:.3g}, V={row['volume_m3']:.3g} m3",
                )
                data=_compact(rd,"cartesian",not args.keep_raw); _write_meta(rd,row,"cartesian",data,metrics)
            else: print("[SKIP] Cartesian already complete")
        if args.mode in ("both","adaptive"):
            rd=os.path.join(base_run,"adaptive")
            if not _completed(rd):
                os.makedirs(rd,exist_ok=True); scfg=_solver_cfg(study,prof,row,dem_path,rd,"cuda",_cache_file(cache_root,cache_import_roots,f"p{p:02d}_adaptive_dem.npz"))
                # Build once per profile from exactly the same cropped DEM.
                if p not in mesh_cache:
                    x,y,z,valid=load_or_build_dem(scfg.grid)
                    x,y,z,valid=_trim_to_valid_rectangular_support(x,y,z,valid)
                    zf=np.where(valid,z,np.nanmax(z[valid])+scfg.grid.nodata_barrier_m)
                    mesh_cache[p]=build_terrain_adapted_mesh(x,y,zf,adapt_cfg,[prof["release_polygon"]])
                    _assert_adaptive_area_matches_cartesian_valid(mesh_cache[p],x,y,valid)
                    prof_dir=os.path.join(root,"profiles"); os.makedirs(prof_dir,exist_ok=True)
                    save_mesh_npz(mesh_cache[p],os.path.join(prof_dir,f"profile_{p:02d}_adaptive_mesh.npz"))
                    with open(os.path.join(prof_dir,f"profile_{p:02d}_adaptive_mesh_summary.json"),"w",encoding="utf-8") as f:
                        json.dump(_mesh_summary(mesh_cache[p],adapt_cfg),f,indent=2)
                metrics=run_solver_adaptive(scfg,adapt_cfg,mesh_cache[p])
                save_final_depth_adaptive(
                    rd, scfg, mesh_cache[p],
                    title=f"{sid} | Terrain-adapted {'/'.join(f'{v:g}' for v in sorted(adapt_cfg.dx_levels))} m | mu_scale={row['mu_scale']:.3g}, xi={row['xi']:.3g}, V={row['volume_m3']:.3g} m3",
                )
                data=_compact(rd,"adaptive",not args.keep_raw); _write_meta(rd,row,"adaptive",data,metrics)
                # Per-run mesh duplicates are not needed; profile mesh is canonical.
                mp=os.path.join(rd,"mesh.npz")
                if os.path.isfile(mp): os.remove(mp)
            else: print("[SKIP] Adaptive already complete")
    print("\nDataset generation pass complete.")

if __name__=="__main__": main()

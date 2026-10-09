from __future__ import annotations

"""
ALL-IN-ONE benchmark for p01_s0048.

ONE script, TWO isolated benchmark phases:

1) PyDebrisFlow2D solver:
   - uses the PERSONAL PyDebrisFlow2D repository passed with --pydebrisflow-root;
   - performs one FULL untimed warm-up simulation in a child Python process;
   - then repeats the identical 0 -> 1000 s simulation in the SAME child process;
   - only the second run is timed, so one-time Numba/CUDA JIT is excluded.

2) Neural operators:
   - loads FNO-C, FNO-A, GNO-C, GNO-A, GINO-C, GINO-A;
   - prepares p01_s0048 at t=1000 s outside the timer;
   - performs one untimed warm-up forward per model;
   - times only the synchronized model forward.

ALL reported times are in SECONDS.
No millisecond output is produced.

The solver runs in a subprocess on purpose: this guarantees that the personal
PyDebrisFlow2D repository is imported before the neural-operator project can
import its own local pydebrisflow package.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

CASES = OrderedDict([
    ("fno_cartesian",  "FNO-C"),
    ("fno_adaptive",   "FNO-A"),
    ("gno_cartesian",  "GNO-C"),
    ("gno_adaptive",   "GNO-A"),
    ("gino_cartesian", "GINO-C"),
    ("gino_adaptive",  "GINO-A"),
])


def read_scenario(manifest: Path, scenario: str) -> dict:
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("scenario_id") == scenario:
                return row
    raise RuntimeError(f"{scenario} not found in {manifest}")


def load_geometry(path: Path, profile_id: int) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj["profiles"][int(profile_id) - 1]


def abs_from(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def make_cfg(SolverConfig, study: dict, profile: dict, scenario: dict,
             dem_path: Path, cache_path: Path, out_dir: Path,
             output_dt: float):
    """Same physical setup used by build_morino_dataset._solver_cfg."""
    cfg = SolverConfig()
    s = study.get("solver", {})
    bbox = profile["clip_bbox"]

    cfg.compute.backend = "cuda"
    cfg.compute.cuda_allow_fallback = True

    cfg.grid.dem_path = str(dem_path)
    cfg.grid.zip_member = str(
        study["paths"].get("zip_member", "DEM/DemMosaic_WGS84_33N.asc")
    )
    cfg.grid.cache_path = str(cache_path)
    cfg.grid.target_dx = float(s.get("cartesian_dx_m", 10.0))
    cfg.grid.clip_enabled = True
    (
        cfg.grid.xmin,
        cfg.grid.xmax,
        cfg.grid.ymin,
        cfg.grid.ymax,
    ) = [float(v) for v in bbox]

    cfg.numerics.cfl = float(s.get("cfl", 0.20))
    cfg.numerics.dt_max = float(s.get("dt_max", 0.25))
    cfg.numerics.t_end = 1000.0
    cfg.numerics.output_dt = float(output_dt)
    cfg.numerics.h_dry = float(s.get("h_dry", 1.0e-4))
    cfg.numerics.space_order = int(s.get("space_order", 1))
    cfg.numerics.time_order = int(s.get("time_order", 1))
    cfg.numerics.flux = str(s.get("flux", "hllc"))
    cfg.numerics.bc_left = str(s.get("bc_left", "outflow"))
    cfg.numerics.bc_right = str(s.get("bc_right", "outflow"))
    cfg.numerics.bc_bottom = str(s.get("bc_bottom", "outflow"))
    cfg.numerics.bc_top = str(s.get("bc_top", "outflow"))

    cfg.release.enabled = True
    cfg.release.mode = "polygon"
    cfg.release.polygon = profile["release_polygon"]
    cfg.release.volume_m3 = float(scenario["volume_m3"])
    cfg.release.initial_speed_ms = float(
        s.get("release_initial_speed_ms", 0.0)
    )

    ms = float(scenario["mu_scale"])
    xi = float(scenario["xi"])

    mat = study.get("material", {})
    cfg.material.rho_fluid = float(
        mat.get("rho_fluid", cfg.material.rho_fluid)
    )
    cfg.material.rho_solid = float(
        mat.get("rho_solid", cfg.material.rho_solid)
    )
    cfg.material.initial_solid_fraction = float(
        mat.get(
            "initial_solid_fraction",
            cfg.material.initial_solid_fraction,
        )
    )
    cfg.material.initial_coarse_fraction = float(
        mat.get(
            "initial_coarse_fraction",
            cfg.material.initial_coarse_fraction,
        )
    )
    cfg.material.initial_upper_solid_fraction = float(
        mat.get(
            "initial_upper_solid_fraction",
            cfg.material.initial_upper_solid_fraction,
        )
    )
    cfg.material.max_solid_fraction = float(
        mat.get("max_solid_fraction", cfg.material.max_solid_fraction)
    )

    # Morino two-zone friction used in the frozen dataset.
    cfg.material.mu_fine = 0.01 * ms
    cfg.material.mu_coarse = 0.10 * ms
    cfg.material.xi_fine = xi
    cfg.material.xi_coarse = xi

    cfg.friction.regions = [
        {
            "polygon": profile["friction_poly1"],
            "mu": 0.01 * ms,
            "xi": xi,
            "name": "morino_region_1",
        },
        {
            "polygon": profile["friction_poly2"],
            "mu": 0.10 * ms,
            "xi": xi,
            "name": "morino_region_2",
        },
    ]

    phys = study.get("physics", {})

    cfg.erosion.enabled = bool(phys.get("erosion", True))
    cfg.deposition.enabled = bool(phys.get("deposition", True))
    cfg.segregation.enabled = bool(phys.get("segregation", True))
    cfg.erosion.update_bed = bool(phys.get("update_bed", True))

    cfg.erosion.erodible_depth_m = float(
        phys.get("erodible_depth_m", cfg.erosion.erodible_depth_m)
    )
    cfg.erosion.bed_porosity = float(
        phys.get("bed_porosity", cfg.erosion.bed_porosity)
    )
    cfg.erosion.bed_coarse_fraction = float(
        phys.get("bed_coarse_fraction", cfg.erosion.bed_coarse_fraction)
    )
    cfg.erosion.critical_shear_pa = float(
        phys.get("erosion_critical_shear_pa", cfg.erosion.critical_shear_pa)
    )
    cfg.erosion.excess_shear_rate_ms = float(
        phys.get("excess_shear_rate_ms", cfg.erosion.excess_shear_rate_ms)
    )
    cfg.erosion.excess_shear_exponent = float(
        phys.get(
            "excess_shear_exponent",
            cfg.erosion.excess_shear_exponent,
        )
    )
    cfg.erosion.max_erosion_rate_ms = float(
        phys.get("max_erosion_rate_ms", cfg.erosion.max_erosion_rate_ms)
    )

    cfg.deposition.critical_shear_pa = float(
        phys.get(
            "deposition_critical_shear_pa",
            cfg.deposition.critical_shear_pa,
        )
    )
    cfg.deposition.hindered_settling_exponent = float(
        phys.get(
            "hindered_settling_exponent",
            cfg.deposition.hindered_settling_exponent,
        )
    )
    cfg.deposition.max_rate_ms = float(
        phys.get("max_deposition_rate_ms", cfg.deposition.max_rate_ms)
    )
    cfg.deposition.minimum_mobile_depth_m = float(
        phys.get(
            "minimum_mobile_depth_m",
            cfg.deposition.minimum_mobile_depth_m,
        )
    )

    cfg.segregation.segregation_length_m = float(
        phys.get(
            "segregation_length_m",
            cfg.segregation.segregation_length_m,
        )
    )
    cfg.segregation.max_segregation_speed_ms = float(
        phys.get(
            "max_segregation_speed_ms",
            cfg.segregation.max_segregation_speed_ms,
        )
    )
    cfg.segregation.remix_diffusivity_m2s = float(
        phys.get(
            "remix_diffusivity_m2s",
            cfg.segregation.remix_diffusivity_m2s,
        )
    )
    cfg.segregation.min_layer_thickness_m = float(
        phys.get(
            "min_layer_thickness_m",
            cfg.segregation.min_layer_thickness_m,
        )
    )

    cfg.output.out_dir = str(out_dir)
    cfg.output.save_snapshots = True
    cfg.output.make_png = False
    cfg.output.make_gif = False
    cfg.output.save_setup_preview = False
    cfg.output.progress_every_steps = int(
        s.get("progress_every_steps", 500)
    )

    return cfg


def numba_cuda_sync() -> None:
    try:
        from numba import cuda
        if cuda.is_available():
            cuda.synchronize()
    except Exception:
        pass


def solver_child(args) -> int:
    """
    Warm-up + measured solver run.
    No neural-operator module is imported before the personal solver path.
    """
    import yaml

    personal_root = args.pydebrisflow_root.resolve()
    if not (personal_root / "pydebrisflow" / "simulation.py").is_file():
        raise FileNotFoundError(
            f"Not a PyDebrisFlow2D repository: {personal_root}"
        )

    sys.path.insert(0, str(personal_root))

    import pydebrisflow
    from pydebrisflow.config import SolverConfig
    from pydebrisflow.simulation import run_solver

    print(f"[PYDEBRISFLOW SOURCE] {Path(pydebrisflow.__file__).resolve()}")

    config = args.config.resolve()
    manifest = args.manifest.resolve()
    geometry = args.geometry.resolve()
    dem = args.dem.resolve()

    if not dem.is_file():
        raise FileNotFoundError(dem)

    study = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    row = read_scenario(manifest, args.scenario)
    profile = load_geometry(geometry, int(row["profile_id"]))

    solver_out = args.out.resolve() / "solver"
    solver_out.mkdir(parents=True, exist_ok=True)

    dataset_root = abs_from(config.parent, study["paths"]["dataset_root"])
    cache_root = dataset_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"p{int(row['profile_id']):02d}_cartesian.npz"

    print("\n=== SOLVER SCENARIO ===")
    print(f"scenario : {row['scenario_id']}")
    print(f"profile  : {row['profile_id']}")
    print(f"split    : {row['split']}")
    print(f"mu_scale : {float(row['mu_scale']):.12g}")
    print(f"xi       : {float(row['xi']):.12g}")
    print(f"V0 [m3]  : {float(row['volume_m3']):.12g}")
    print(f"t_end    : {args.time:g} s")
    print(f"output_dt: {args.output_dt:g} s")
    print(f"DEM      : {dem}")
    print(f"cache    : {cache_path}")

    warmup_dir = solver_out / "warmup_run"
    warmup_dir.mkdir(parents=True, exist_ok=True)

    warmup_cfg = make_cfg(
        SolverConfig=SolverConfig,
        study=study,
        profile=profile,
        scenario=row,
        dem_path=dem,
        cache_path=cache_path,
        out_dir=warmup_dir,
        output_dt=args.output_dt,
    )
    warmup_cfg.numerics.t_end = float(args.time)

    print("\n=== SOLVER WARM-UP: NOT TIMED ===")
    run_solver(warmup_cfg)
    numba_cuda_sync()
    print("[WARM-UP COMPLETE] JIT/framework initialization excluded.")

    measured_dir = solver_out / "measured_run"
    measured_dir.mkdir(parents=True, exist_ok=True)

    measured_cfg = make_cfg(
        SolverConfig=SolverConfig,
        study=study,
        profile=profile,
        scenario=row,
        dem_path=dem,
        cache_path=cache_path,
        out_dir=measured_dir,
        output_dt=args.output_dt,
    )
    measured_cfg.numerics.t_end = float(args.time)

    print("\n=== SOLVER MEASURED RUN ===")
    numba_cuda_sync()
    t0 = time.perf_counter()
    metrics = run_solver(measured_cfg)
    numba_cuda_sync()
    solver_s = time.perf_counter() - t0

    print("\n=== SOLVER TIME ===")
    print(f"PyDebrisFlow2D 0 -> {args.time:g} s : {solver_s:.6f} s")

    result = {
        "scenario_id": row["scenario_id"],
        "profile_id": int(row["profile_id"]),
        "split": row["split"],
        "mu_scale": float(row["mu_scale"]),
        "xi": float(row["xi"]),
        "volume_m3": float(row["volume_m3"]),
        "target_time_s": float(args.time),
        "output_dt_s": float(args.output_dt),
        "solver_wall_s": float(solver_s),
        "timing_definition": (
            "Second identical full solver run in the same process; "
            "first full run is untimed warm-up. CUDA/Numba JIT and one-time "
            "framework initialization are excluded."
        ),
        "pydebrisflow_source": str(Path(pydebrisflow.__file__).resolve()),
        "dem": str(dem),
        "solver_reported_elapsed_wall_s": (
            float(metrics.get("elapsed_wall_s"))
            if isinstance(metrics, dict) and metrics.get("elapsed_wall_s") is not None
            else None
        ),
    }

    args.solver_result.parent.mkdir(parents=True, exist_ok=True)
    args.solver_result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[SAVED] {args.solver_result}")
    return 0


def run_solver_subprocess(args) -> dict:
    result_path = args.out.resolve() / "_solver_result.json"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_solver-child",
        "--pydebrisflow-root", str(args.pydebrisflow_root.resolve()),
        "--dem", str(args.dem.resolve()),
        "--config", str(args.config.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--geometry", str(args.geometry.resolve()),
        "--scenario", args.scenario,
        "--split", args.split,
        "--time", str(args.time),
        "--output-dt", str(args.output_dt),
        "--out", str(args.out.resolve()),
        "--solver-result", str(result_path),
    ]

    print("\n" + "=" * 78)
    print("PHASE 1/2 - PYDEBRISFLOW2D")
    print("=" * 78)

    subprocess.run(cmd, check=True)

    if not result_path.is_file():
        raise RuntimeError(f"Solver result file not created: {result_path}")

    return json.loads(result_path.read_text(encoding="utf-8"))


def checkpoint_path(tier: str, case: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "output"
        / "models"
        / tier
        / f"{case}_seed{seed}"
        / "best.pt"
    )


def manifest_from_checkpoint(tier: str, seed: int, torch) -> Path:
    ckpt = checkpoint_path(tier, "fno_cartesian", seed)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    manifest = Path(ck["config"]["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()
    return manifest


def torch_sync(device, torch) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def prepare_case(
    case: str,
    tier: str,
    seed: int,
    split: str,
    scenario: str,
    requested_time: float,
    device,
    torch,
    inf,
    MorinoFrameDataset,
):
    ckpt = checkpoint_path(tier, case, seed)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    ck, cfg, kind, rep, model, scale = inf._load_model(ckpt, device)

    manifest = Path(cfg["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()

    ds = MorinoFrameDataset(
        str(manifest),
        split,
        rep,
        scale,
        frame_stride=1,
        include_t0=True,
        raster_dx=float(cfg["data"].get("raster_dx", 10.0)),
        cache_static_items=int(cfg["data"].get("cache_static_items", 12)),
    )

    batch, actual_time, _, _ = inf._find_frame(ds, scenario, requested_time)

    if abs(actual_time - requested_time) > 1.0e-6:
        raise RuntimeError(
            f"{case}: requested t={requested_time:g}, got t={actual_time:g}"
        )

    torch_sync(device, torch)

    return {
        "model": model,
        "kind": kind,
        "batch": batch,
        "parameters": int(ck.get("parameter_count", -1)),
        "representation": rep,
        "actual_time": float(actual_time),
    }


def time_inference(
    prepared: dict,
    device,
    warmup: int,
    repeats: int,
    np,
    torch,
    inf,
):
    model = prepared["model"]
    kind = prepared["kind"]
    batch = prepared["batch"]

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            _ = inf._forward(model, kind, batch, device)
            torch_sync(device, torch)

    times = []
    with torch.no_grad():
        for _ in range(repeats):
            torch_sync(device, torch)
            t0 = time.perf_counter()
            pred = inf._forward(model, kind, batch, device)
            torch_sync(device, torch)
            times.append(time.perf_counter() - t0)
            del pred

    a = np.asarray(times, dtype=float)
    return {
        "single_s": float(a[0]),
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
        "std_s": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
    }


def run_inference_benchmark(args):
    import numpy as np
    import torch
    import infer_same_sample as inf
    from ml.data import MorinoFrameDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = manifest_from_checkpoint(args.tier, args.seed, torch)

    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else manifest.parent.resolve()
    )
    os.environ["MORINO_DATASET_ROOT"] = str(dataset_root)

    row = read_scenario(manifest, args.scenario)

    print("\n" + "=" * 78)
    print("PHASE 2/2 - SIX NEURAL-OPERATOR INFERENCES")
    print("=" * 78)
    print(f"[DEVICE]  {device}")
    print(f"scenario  : {row['scenario_id']}")
    print(f"profile   : {row['profile_id']}")
    print(f"split     : {row['split']}")
    print(f"mu_scale  : {float(row['mu_scale']):.12g}")
    print(f"xi        : {float(row['xi']):.12g}")
    print(f"V0 [m3]   : {float(row['volume_m3']):.12g}")
    print(f"time      : {args.time:g} s")
    print(f"tier      : {args.tier}")
    print(
        f"timing    : inference only "
        f"(warmup={args.inference_warmup}, repeats={args.repeats})"
    )

    results = []

    print("\n=== SIX INFERENCE TIMES ===")
    for case, short in CASES.items():
        prepared = prepare_case(
            case=case,
            tier=args.tier,
            seed=args.seed,
            split=args.split,
            scenario=args.scenario,
            requested_time=args.time,
            device=device,
            torch=torch,
            inf=inf,
            MorinoFrameDataset=MorinoFrameDataset,
        )

        timing = time_inference(
            prepared=prepared,
            device=device,
            warmup=args.inference_warmup,
            repeats=args.repeats,
            np=np,
            torch=torch,
            inf=inf,
        )

        representative_s = (
            timing["single_s"]
            if args.repeats == 1
            else timing["median_s"]
        )

        rec = {
            "case": case,
            "model": short,
            "representation": prepared["representation"],
            "parameters": prepared["parameters"],
            "target_time_s": prepared["actual_time"],
            "inference_s": float(representative_s),
            **{f"timing_{k}": v for k, v in timing.items()},
        }
        results.append(rec)

        print(f"{short:6s} : {representative_s:.6f} s")

        del prepared
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_s = sum(r["inference_s"] for r in results)
    print(f"\n6 model inferences total : {total_s:.6f} s")

    scenario_info = {
        "scenario_id": row["scenario_id"],
        "profile_id": int(row["profile_id"]),
        "split": row["split"],
        "mu_scale": float(row["mu_scale"]),
        "xi": float(row["xi"]),
        "volume_m3": float(row["volume_m3"]),
        "target_time_s": float(args.time),
        "device": str(device),
    }

    return results, float(total_s), scenario_info


def write_final_report(
    args,
    solver_result: dict,
    inference_results: list[dict],
    inference_total_s: float,
    scenario_info: dict,
) -> None:
    solver_s = float(solver_result["solver_wall_s"])

    for r in inference_results:
        r["speedup_vs_solver"] = (
            solver_s / r["inference_s"]
            if r["inference_s"] > 0.0
            else None
        )

    print("\n" + "=" * 78)
    print("FINAL BENCHMARK - ALL TIMES IN SECONDS")
    print("=" * 78)
    print(f"{'PyDebrisFlow2D':16s} : {solver_s:.6f} s")
    for r in inference_results:
        print(
            f"{r['model']:16s} : {r['inference_s']:.6f} s"
            f"    speed-up = {r['speedup_vs_solver']:.2f}x"
        )
    print("-" * 78)
    print(f"{'6 NN total':16s} : {inference_total_s:.6f} s")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = [{
        "method": "PyDebrisFlow2D",
        "time_s": solver_s,
        "speedup_vs_solver": 1.0,
        "timing_type": "warm steady-state solver 0->target time; JIT excluded",
    }]
    for r in inference_results:
        rows.append({
            "method": r["model"],
            "time_s": r["inference_s"],
            "speedup_vs_solver": r["speedup_vs_solver"],
            "timing_type": "single target-time model inference; warm-up excluded",
        })

    csv_path = out / "benchmark_p01_s0048_all_times.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out / "benchmark_p01_s0048_all_times.json"
    json_path.write_text(
        json.dumps(
            {
                "scenario": scenario_info,
                "solver": solver_result,
                "inference": {
                    "tier": args.tier,
                    "seed": args.seed,
                    "warmup_forwards": args.inference_warmup,
                    "timed_repeats": args.repeats,
                    "results": inference_results,
                    "total_six_inferences_s": inference_total_s,
                    "timing_definition": (
                        "Timer starts immediately before model forward and stops "
                        "after CUDA synchronization. Model loading, HDF5/data "
                        "preparation and warm-up are excluded."
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[SAVED] {csv_path}")
    print(f"[SAVED] {json_path}")


def build_parser():
    ap = argparse.ArgumentParser(
        description=(
            "All-in-one p01_s0048 benchmark: warm PyDebrisFlow2D 0->1000 s "
            "plus six neural-operator inference times."
        )
    )

    ap.add_argument(
        "--pydebrisflow-root",
        type=Path,
        required=True,
        help="Personal PyDebrisFlow2D repository containing pydebrisflow/.",
    )
    ap.add_argument(
        "--dem",
        type=Path,
        required=True,
        help="Morino DEM ZIP.",
    )

    ap.add_argument("--scenario", default="p01_s0048")
    ap.add_argument("--split", default="joint_ood_test")
    ap.add_argument("--time", type=float, default=1000.0)

    ap.add_argument(
        "--tier",
        default="deep200",
        choices=["deep", "deep200"],
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--inference-warmup",
        type=int,
        default=1,
        help="Untimed warm-up forwards per neural model.",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Timed forwards per model. If >1, median is reported.",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset folder containing runs/. Usually not needed.",
    )

    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs_ml" / "morino_dataset.yaml",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "output"
            / "dataset"
            / "manifest.csv"
        ),
    )
    ap.add_argument(
        "--geometry",
        type=Path,
        default=PROJECT_ROOT / "output" / "geometry_morino_6profiles.json",
    )
    ap.add_argument(
        "--output-dt",
        type=float,
        default=1000.0,
        help="Solver output interval in seconds. Default=1000.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "output" / "p01_s0048_full_benchmark",
    )

    ap.add_argument("--_solver-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--solver-result",
        type=Path,
        default=PROJECT_ROOT / "output" / "p01_s0048_full_benchmark" / "_solver_result.json",
        help=argparse.SUPPRESS,
    )

    return ap


def main():
    args = build_parser().parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.inference_warmup < 0:
        raise ValueError("--inference-warmup must be >= 0")

    if args._solver_child:
        raise SystemExit(solver_child(args))

    args.out.mkdir(parents=True, exist_ok=True)

    solver_result = run_solver_subprocess(args)

    inference_results, inference_total_s, scenario_info = (
        run_inference_benchmark(args)
    )

    write_final_report(
        args=args,
        solver_result=solver_result,
        inference_results=inference_results,
        inference_total_s=inference_total_s,
        scenario_info=scenario_info,
    )


if __name__ == "__main__":
    main()

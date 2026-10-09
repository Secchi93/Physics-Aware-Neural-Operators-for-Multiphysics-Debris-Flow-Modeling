from __future__ import annotations



"""

Paper-quality inference for the six trained FNO/GNO/GINO models.



The script can run either one model (``--case``) or all six on the same

Morino scenario and requested physical time.



Comparison cases

----------------

FNO-C  : Cartesian solver field -> FNO

GNO-C  : same Cartesian field represented as points -> pure GNO

GINO-C : same Cartesian field represented as points -> GINO

FNO-A  : adaptive solver field -> conservative raster -> FNO

GNO-A  : native adaptive cells -> pure GNO

GINO-A : native adaptive cells -> GINO



Evaluation policy

-----------------

* Cartesian models are evaluated against the Cartesian native solver truth.

* GNO-A and GINO-A are evaluated directly on the native adaptive cells.

* FNO-A follows the SAME primary policy used by ``evaluation.py``:

  its raster prediction is area-overlap back-projected to the original

  adaptive cells and compared with the pre-rasterization adaptive truth.

* Adaptive native fields are rasterized only for visualization.

* For FNO-A, raster-space metrics are retained only as explicitly labelled

  secondary diagnostics.



Defaults are tuned for the final paper figure requested here: tier=deep200,\nFNO-C, split=joint_ood_test, auto-selected scenario, requested time=1000 s.

"""



import argparse

import csv

import json

from collections import OrderedDict

from pathlib import Path

import sys

import time

import os



import h5py

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import LightSource, ListedColormap, Normalize, TwoSlopeNorm

from matplotlib.patches import Patch

import numpy as np

import torch

from torch.utils.data import DataLoader, Subset



PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(0, str(PROJECT_ROOT))



from ml.data import MorinoFrameDataset

from ml.metrics import frame_metrics, primitive_from_target

from ml.models import build_model

from ml.rasterize import (

    apply_raster_map,

    apply_raster_to_source_map,

    build_raster_map,

)





CASES = OrderedDict([

    ("fno_cartesian", {"short": "FNO-C", "truth": "Cartesian"}),

    ("gno_cartesian", {"short": "GNO-C", "truth": "Cartesian"}),

    ("gino_cartesian", {"short": "GINO-C", "truth": "Cartesian"}),

    ("fno_adaptive", {"short": "FNO-A", "truth": "Adaptive"}),

    ("gno_adaptive", {"short": "GNO-A", "truth": "Adaptive"}),

    ("gino_adaptive", {"short": "GINO-A", "truth": "Adaptive"}),

])



def _sanitize_state_dict(state_dict):

    """Remove NeuralOperator's literal _metadata key without touching weights."""

    native_metadata = getattr(state_dict, "_metadata", None)

    cleaned = state_dict.copy()

    cleaned.pop("_metadata", None)

    if native_metadata is not None:

        try:

            cleaned._metadata = native_metadata

        except Exception:

            pass

    return cleaned





WET_COLORS = ListedColormap([

    (0, 0, 0, 0),

    (0.13, 0.62, 0.39, .84),  # TRUE and PRED

    (1.00, 0.54, 0.08, .87),  # false positive

    (0.67, 0.25, 0.65, .92),  # false negative

])





def _resolve_path(raw: str, manifest: Path) -> Path:

    """Resolve manifest data paths, including an external dataset root.



    The dataset loader already understands MORINO_DATASET_ROOT; this helper must

    use the same rule so plotting/topography does not fall back to stale absolute

    Windows paths stored in older manifests.

    """

    raw = str(raw or "").strip()

    p = Path(raw)



    # 1) Path is already valid on this machine.

    if p.is_file():

        return p.resolve()



    normalized = raw.replace("\\", "/")



    # 2) Normal relative path next to the manifest/dataset.

    if not p.is_absolute():

        q = manifest.parent.joinpath(*normalized.split('/'))

        if q.is_file():

            return q.resolve()



    # 3) External dataset root explicitly supplied through --dataset-root.

    external_root = os.environ.get("MORINO_DATASET_ROOT", "").strip()

    if external_root:

        root = Path(external_root).expanduser().resolve()



        # Legacy manifests can contain an absolute path ending in runs/... .

        # Preserve only that portable tail and attach it to the supplied root.

        low = normalized.lower()

        token = '/runs/'

        if token in low:

            tail = normalized[low.index(token) + 1:]  # starts with runs/...

            q = root.joinpath(*tail.split('/'))

            if q.is_file():

                return q.resolve()



        # Newer manifests may already store runs/... as a relative path.

        if normalized.lower().startswith('runs/'):

            q = root.joinpath(*normalized.split('/'))

            if q.is_file():

                return q.resolve()



        # Also allow the user to pass the parent study directory rather than the

        # dataset directory itself.

        dataset_name = manifest.parent.name

        q = root / dataset_name

        if normalized.lower().startswith('runs/'):

            q = q.joinpath(*normalized.split('/'))

            if q.is_file():

                return q.resolve()

        elif token in low:

            tail = normalized[low.index(token) + 1:]

            q = q.joinpath(*tail.split('/'))

            if q.is_file():

                return q.resolve()



    # Return original candidate so the caller prints a meaningful missing path.

    return p





def _select_scenario(manifest: Path, split: str, requested: str | None, require_both: bool) -> str:
    """Select a scenario actually available in the requested split.

    Selection is deterministic (manifest order) and requires actual HDF5 files
    for both representations when comparing all six models.
    """
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if row.get("split") == split]
    if not rows:
        with manifest.open(newline="", encoding="utf-8-sig") as f:
            available = sorted({row.get("split", "") for row in csv.DictReader(f)})
        raise ValueError(f"No scenarios in split={split!r}. Available splits: {available}")

    def present(row):
        keys = ("cartesian_path", "adaptive_path") if require_both else ("cartesian_path", "adaptive_path")
        # The representation-specific preflight in run_case checks each selected model.
        from ml.data import resolve_manifest_data_path
        return all(Path(resolve_manifest_data_path(row.get(k, ""), str(manifest))).is_file() for k in keys)

    if requested:
        matches = [row for row in rows if row.get("scenario_id") == requested]
        if not matches:
            sample = [row.get("scenario_id") for row in rows[:8]]
            raise ValueError(f"Scenario {requested!r} absent from split {split!r}. Examples: {sample}")
        if require_both and not any(present(row) for row in matches):
            raise FileNotFoundError(f"Scenario {requested!r} lacks available Cartesian/adaptive HDF5 inputs")
        return requested

    for row in rows:
        if not require_both or present(row):
            selected = row["scenario_id"]
            print(f"[SCENARIO AUTO] {selected} (split={split})")
            return selected
    raise FileNotFoundError(f"Split {split!r} has no scenario with both Cartesian and adaptive HDF5 files")


def _manifest_row(manifest: Path, split: str, scenario: str) -> dict:

    with manifest.open(newline="", encoding="utf-8-sig") as f:

        for row in csv.DictReader(f):

            if row.get("split") == split and row.get("scenario_id") == scenario:

                return row

    raise RuntimeError(

        f"Scenario {scenario!r} not found in split {split!r} in {manifest}"

    )





def _coords_from_batch(batch: dict, rep: str) -> np.ndarray:

    if rep in ("adaptive", "cartesian_points"):

        return batch["coords_m"][0].cpu().numpy()

    x = batch["x_coord"][0].cpu().numpy()

    y = batch["y_coord"][0].cpu().numpy()

    Y, X = np.meshgrid(y, x, indexing="ij")

    return np.stack((X, Y), axis=-1)





def _forward(model, kind: str, batch: dict, device: torch.device) -> torch.Tensor:

    x = batch["x"].to(device)

    if kind == "fno":

        return model(x)

    coords = batch["coords"].to(device)

    area = batch["area"].to(device)

    pid = int(batch["profile_id"][0])

    # PureGNO returns [B,N,C]; evaluator convention is [B,C,N].

    return model(x=x, coords=coords, area=area, profile_id=pid).permute(0, 2, 1)





def _find_frame(ds: MorinoFrameDataset, scenario: str, requested_time: float):

    row_indices = [

        i for i, row in enumerate(ds.rows)

        if row.get("scenario_id") == scenario

    ]

    if not row_indices:

        raise RuntimeError(

            f"Scenario {scenario!r} is absent from dataset split={ds.split!r}, "

            f"representation={ds.representation!r}"

        )

    # One manifest row per scenario in the study design.

    ir = row_indices[0]

    row = ds.rows[ir]

    path_key = (

        "cartesian_path"

        if ds.representation in ("cartesian", "cartesian_points")

        else "adaptive_path"

    )

    path = Path(row[path_key])

    if not path.is_file() and not path.is_absolute():

        path = Path(ds.root) / path

    with h5py.File(path, "r") as h5:

        times = np.asarray(h5["times"][:], dtype=float)

    it = int(np.argmin(np.abs(times - float(requested_time))))

    frame_idx = None

    for k, ref in enumerate(ds.frames):

        if ref.row_index == ir and ref.time_index == it:

            frame_idx = k

            break

    if frame_idx is None:

        raise RuntimeError(

            f"Nearest time index {it} was not exposed by the dataset "

            f"(frame_stride may exclude it)."

        )

    loader = DataLoader(

        Subset(ds, [frame_idx]),

        batch_size=1,

        shuffle=False,

        num_workers=0,

    )

    return next(iter(loader)), float(times[it]), row, frame_idx





def _load_model(checkpoint: Path, device: torch.device):

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)

    kind = str(ck["kind"]).lower()

    cfg = ck["config"]

    rep = str(ck.get("representation", cfg["data"]["representation"]))

    model = build_model(kind, 13, 8, cfg["model"])

    model.load_state_dict(_sanitize_state_dict(ck["model"]))

    model = model.to(device).eval()

    scale = np.asarray(ck["target_scale"], dtype=np.float32)

    return ck, cfg, kind, rep, model, scale





def _physical_states(kind: str, pred_n_t: torch.Tensor, batch: dict, scale: np.ndarray):

    pred_n = pred_n_t[0].detach().cpu().numpy()

    if kind in ("gno", "gino"):

        target_n = batch["y"][0].cpu().numpy().T

    else:

        target_n = batch["y"][0].cpu().numpy()

    sc = scale.reshape((8,) + ((1,) * (pred_n.ndim - 1)))

    return pred_n * sc, target_n * sc





def _initial_state(batch: dict, rep: str, scale: np.ndarray) -> np.ndarray:

    xin = batch["x"][0].cpu().numpy()

    if rep in ("adaptive", "cartesian_points"):

        return np.moveaxis(xin[..., :7], -1, 0) * scale[:7, None]

    return xin[:7] * scale[:7, None, None]





def _reference_policy(rep: str):

    if rep == "adaptive_rasterized":

        return (

            "adaptive_native_pre_rasterization",

            "adaptive_raster",

            "area_overlap_raster_to_native",

        )

    if rep == "adaptive":

        return ("adaptive_native", "adaptive_native", "none")

    return ("cartesian_native", "cartesian_native", "none")





def _native_adaptive_frame(ds: MorinoFrameDataset, dataset_index: int):

    """Load the original adaptive state for the selected dataset frame.



    This mirrors the primary FNO-A evaluation path in ``evaluation.py``.

    """

    ref = ds.frames[dataset_index]

    row = ds.rows[ref.row_index]

    pid = int(row["profile_id"])

    path = row["adaptive_path"]



    U0, z0, _zrel, _sx, _sy, area, _coords, _t_end, coords_m = ds._static_adapt(

        path, pid

    )

    with h5py.File(path, "r") as h5:

        time_s = float(h5["times"][ref.time_index])

        U = h5["U"][ref.time_index].astype(np.float32)

        zt = h5["zb"][ref.time_index].astype(np.float32)



    target = np.concatenate([U, (zt - z0)[None]], axis=0).astype(np.float32)

    return {

        "U0": U0.astype(np.float32, copy=False),

        "target": target,

        "area": area.astype(np.float32, copy=False),

        "coords_m": coords_m.astype(np.float32, copy=False),

        "time_s": time_s,

        "profile_id": pid,

        "rmap": ds._rmap(pid),

    }





def _jsonable(value):

    """Convert numpy/scalar containers to objects accepted by json.dump."""

    if isinstance(value, dict):

        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):

        return [_jsonable(v) for v in value]

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return value.item()

    return value





def _read_topography(

    manifest: Path,

    manifest_row: dict,

    rep: str,

    profile_id: int,

    raster_dx: float,

):

    if rep in ("cartesian", "cartesian_points"):

        p = _resolve_path(manifest_row["cartesian_path"], manifest)

        if not p.is_file():

            raise FileNotFoundError(f"Missing Cartesian HDF5: {p}")

        with h5py.File(p, "r") as h5:

            z0 = h5["zb"][0].astype(np.float32)

            x = h5["x"][:].astype(np.float32)

            y = h5["y"][:].astype(np.float32)

        return z0, x, y, None



    p = _resolve_path(manifest_row["adaptive_path"], manifest)

    if not p.is_file():

        raise FileNotFoundError(f"Missing adaptive HDF5: {p}")

    with h5py.File(p, "r") as h5:

        z0_native = h5["zb"][0].astype(np.float32)

    mesh = manifest.parent / "profiles" / f"profile_{profile_id:02d}_adaptive_mesh.npz"

    if not mesh.is_file():

        raise FileNotFoundError(f"Missing adaptive mesh: {mesh}")

    rm = build_raster_map(str(mesh), target_dx=float(raster_dx))

    z0r = apply_raster_map(rm, z0_native)

    return z0r, rm.x, rm.y, rm





def _plot_fields(

    rep: str,

    h_true: np.ndarray,

    h_pred: np.ndarray,

    z0_plot: np.ndarray,

    x: np.ndarray,

    y: np.ndarray,

    rmap,

):

    if rep == "adaptive":

        # Native adaptive inference/evaluation is untouched. Rasterization is

        # performed here only to make a common paper-quality map.

        ht = apply_raster_map(rmap, np.asarray(h_true))

        hp = apply_raster_map(rmap, np.asarray(h_pred))

        return ht, hp, z0_plot, x, y



    if rep == "cartesian_points":

        ny, nx = len(y), len(x)

        return (

            np.asarray(h_true).reshape(ny, nx),

            np.asarray(h_pred).reshape(ny, nx),

            z0_plot,

            x,

            y,

        )



    # Cartesian FNO and adaptive-rasterized FNO are already rasters.

    return np.asarray(h_true), np.asarray(h_pred), z0_plot, x, y





def _orient(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:

    q = np.asarray(a)

    if len(y) > 1 and y[1] < y[0]:

        q = q[::-1]

    if len(x) > 1 and x[1] < x[0]:

        q = q[:, ::-1]

    return q





def _extent(x: np.ndarray, y: np.ndarray):

    dx = float(abs(x[1] - x[0])) if len(x) > 1 else 1.0

    dy = float(abs(y[1] - y[0])) if len(y) > 1 else 1.0

    return [

        float(min(x) - dx / 2), float(max(x) + dx / 2),

        float(min(y) - dy / 2), float(max(y) + dy / 2),

    ]





def _hillshade(z: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Return the original paper-style grayscale hillshade with inverted relief.

    The plotting style is the same as the previous inference figures
    (Matplotlib LightSource + ``cmap="gray"``).  Only the elevation field is
    inverted before shading, so low and high topography are swapped while the
    grayscale/hillshade appearance is preserved.
    """
    zz = _orient(np.asarray(z, dtype=float), x, y)

    finite = np.isfinite(zz)
    if not np.any(finite):
        return np.zeros_like(zz, dtype=float)

    # Invert the DEM while preserving its original numerical range.
    # This is equivalent to reflecting elevations about the mid-range:
    #     z_inv = z_min + z_max - z
    zmin = float(np.nanmin(zz[finite]))
    zmax = float(np.nanmax(zz[finite]))
    zz_inv = (zmin + zmax) - zz

    # Fill any non-finite cells only for the shading calculation.
    if not np.all(finite):
        fill = float(np.nanmedian(zz_inv[finite]))
        zz_inv = np.where(finite, zz_inv, fill)

    dx = float(abs(x[1] - x[0])) if len(x) > 1 else 1.0
    dy = float(abs(y[1] - y[0])) if len(y) > 1 else 1.0

    return LightSource(azdeg=315, altdeg=45).hillshade(
        zz_inv,
        vert_exag=1.0,
        dx=dx,
        dy=dy,
    )



def _base(ax, shade, extent):
    # Paper-style grayscale: preserve the old hillshade, but compress the
    # black/white extremes toward mid-gray for a softer background.
    shade_gray = np.clip(0.20 + 0.65 * shade, 0.0, 1.0)

    ax.imshow(
        shade_gray, origin="lower", extent=extent, cmap="gray",
        vmin=0.0, vmax=1.0, interpolation="nearest", alpha=0.90,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(axis="both", labelsize=8)


def _draw_2x2(result: dict, out_png: Path, hmax: float, emax: float, hwet: float, dpi: int):

    x, y = result["x"], result["y"]

    ht, hp = result["h_true_plot"], result["h_pred_plot"]

    shade = _hillshade(result["z0_plot"], x, y)

    extent = _extent(x, y)

    wet_t = ht >= hwet

    wet_p = hp >= hwet



    cmap_h = plt.get_cmap("turbo").copy()

    cmap_h.set_bad((0, 0, 0, 0))

    cmap_e = plt.get_cmap("RdBu_r").copy()

    cmap_e.set_bad((0, 0, 0, 0))



    fig, axes = plt.subplots(2, 2, figsize=(13.4, 10.0), constrained_layout=True)

    for ax in axes.flat:

        _base(ax, shade, extent)



    im_h = None

    for ax, field, wet, title in (

        (axes[0, 0], ht, wet_t, f"TRUE — PyDebrisFlow2D ({result['truth_family']})"),

        (axes[0, 1], hp, wet_p, f"PRED — {result['short']} {result['tier']} physics-aware"),

    ):

        arr = np.ma.masked_where(~_orient(wet, x, y), _orient(field, x, y))

        im_h = ax.imshow(

            arr, origin="lower", extent=extent, cmap=cmap_h,

            vmin=0.0, vmax=hmax, alpha=0.93, interpolation="nearest",

        )

        ax.set_title(title, fontsize=12, weight="semibold")



    err = hp - ht

    err_mask = wet_t | wet_p

    im_e = axes[1, 0].imshow(

        np.ma.masked_where(~_orient(err_mask, x, y), _orient(err, x, y)),

        origin="lower", extent=extent, cmap=cmap_e,

        norm=TwoSlopeNorm(vmin=-emax, vcenter=0.0, vmax=emax),

        alpha=0.92, interpolation="nearest",

    )

    axes[1, 0].set_title("Errore di spessore: PRED − TRUE", fontsize=12, weight="semibold")



    overlap = np.zeros(ht.shape, dtype=np.uint8)

    overlap[wet_t & wet_p] = 1

    overlap[~wet_t & wet_p] = 2

    overlap[wet_t & ~wet_p] = 3

    ov = np.ma.masked_where(_orient(overlap == 0, x, y), _orient(overlap, x, y))

    axes[1, 1].imshow(

        ov, origin="lower", extent=extent, cmap=WET_COLORS,

        norm=Normalize(vmin=0, vmax=3), interpolation="nearest",

    )

    axes[1, 1].set_title("Footprint: intersezione / errori", fontsize=12, weight="semibold")

    axes[1, 1].legend(

        handles=[

            Patch(facecolor=WET_COLORS(1 / 3), label="TRUE ∩ PRED"),

            Patch(facecolor=WET_COLORS(2 / 3), label="Solo PRED (FP)"),

            Patch(facecolor=WET_COLORS(1.0), label="Solo TRUE (FN)"),

        ],

        loc="lower left", fontsize=8, framealpha=.91,

    )



    fig.colorbar(

        im_h, ax=[axes[0, 0], axes[0, 1]], orientation="horizontal",

        fraction=.044, pad=.03, label="Spessore h [m]",

    )

    fig.colorbar(

        im_e, ax=axes[1, 0], orientation="horizontal",

        fraction=.055, pad=.05, label="Errore h [m]",

    )



    m = result["metrics"]

    fig.suptitle(

        f"{result['scenario']} | {result['split']} | t={result['time_s']:.1f} s | "

        f"{result['short']} {result['tier']}\n"

        f"IoU={m['wet_iou']:.3f} | R²(h)={m['h_r2']:.3f} | "

        f"RMSE(h)={m['h_rmse']:.3f} m | "

        f"runout TRUE/PRED={m['runout_true_m']:.0f}/{m['runout_pred_m']:.0f} m",

        fontsize=13,

    )

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=dpi, facecolor="white")

    plt.close(fig)





def _draw_overview(results: OrderedDict, out_png: Path, hmax: float, hwet: float, dpi: int):

    fc = results["fno_cartesian"]

    gc = results["gno_cartesian"]

    gic = results["gino_cartesian"]

    fa = results["fno_adaptive"]

    ga = results["gno_adaptive"]

    gia = results["gino_adaptive"]



    fig, ax = plt.subplots(2, 4, figsize=(20.0, 8.8), constrained_layout=True)

    cmap_h = plt.get_cmap("turbo").copy()

    cmap_h.set_bad((0, 0, 0, 0))



    panels = [

        (ax[0, 0], fc, fc["h_true_plot"], "TRUE — Cartesian"),

        (ax[0, 1], fc, fc["h_pred_plot"], "FNO-C"),

        (ax[0, 2], gc, gc["h_pred_plot"], "GNO-C"),

        (ax[0, 3], gic, gic["h_pred_plot"], "GINO-C"),

        (ax[1, 0], fa, fa["h_true_plot"], "TRUE — Adaptive"),

        (ax[1, 1], fa, fa["h_pred_plot"], "FNO-A"),

        (ax[1, 2], ga, ga["h_pred_plot"], "GNO-A"),

        (ax[1, 3], gia, gia["h_pred_plot"], "GINO-A"),

    ]

    im = None

    for a, r, field, title in panels:

        x, y = r["x"], r["y"]

        extent = _extent(x, y)

        shade = _hillshade(r["z0_plot"], x, y)

        _base(a, shade, extent)

        wet = field >= hwet

        arr = np.ma.masked_where(~_orient(wet, x, y), _orient(field, x, y))

        im = a.imshow(

            arr, origin="lower", extent=extent, cmap=cmap_h,

            vmin=0.0, vmax=hmax, alpha=.93, interpolation="nearest",

        )

        if title.startswith("TRUE"):

            a.set_title(title, fontsize=12, weight="semibold")

        else:

            m = r["metrics"]

            a.set_title(

                f"{title} — {r['tier']}\n"

                f"IoU={m['wet_iou']:.3f} | R²={m['h_r2']:.3f} | RMSE={m['h_rmse']:.3f} m",

                fontsize=10.5, weight="semibold",

            )



    fig.colorbar(

        im, ax=ax.ravel().tolist(), orientation="horizontal",

        fraction=.038, pad=.04, label="Spessore h [m]",

    )

    sample = fc

    fig.suptitle(

        f"Confronto sullo stesso campione — {sample['scenario']} | "

        f"{sample['split']} | t≈{sample['requested_time']:.1f} s",

        fontsize=15,

    )

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_png, dpi=dpi, facecolor="white")

    plt.close(fig)





def _case_checkpoint(tier: str, case: str, seed: int) -> Path:

    return (

        PROJECT_ROOT / "output" / "models" / tier

        / f"{case}_seed{seed}" / "best.pt"

    )





def run_case(

    case: str,

    meta: dict,

    tier: str,

    seed: int,

    split: str,

    scenario: str,

    requested_time: float,

    device: torch.device,

):

    checkpoint = _case_checkpoint(tier, case, seed)

    if not checkpoint.is_file():

        raise FileNotFoundError(

            f"Missing checkpoint for {case}: {checkpoint}\n"

            f"Train it first with run.py --stage train"

        )



    ck, cfg, kind, rep, model, scale = _load_model(checkpoint, device)



    # Checkpoints retain their original training configuration, including
    # obsolete study/ paths. The publication package has a stable data layout.
    dataset_root = Path(os.environ.get("MORINO_DATASET_ROOT") or (PROJECT_ROOT / "output" / "dataset")).expanduser().resolve()
    manifest = dataset_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}. Place the Zenodo dataset under "
            "output/dataset/ or specify --dataset-root."
        )



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

    batch, actual_time, _ds_row, dataset_index = _find_frame(

        ds, scenario, requested_time

    )

    pid = int(batch["profile_id"][0])



    # Model-only forward timing, matching evaluation.py.

    if device.type == "cuda":

        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():

        pred_n_t = _forward(model, kind, batch, device)

    if device.type == "cuda":

        torch.cuda.synchronize()

    inference_s = time.perf_counter() - t0



    # Network-output-space states.

    pred_out, target_out = _physical_states(kind, pred_n_t, batch, scale)

    area_out = batch["area"][0].cpu().numpy()

    coords_out = _coords_from_batch(batch, rep)

    U0_out = _initial_state(batch, rep, scale)



    # Primary evaluation state. FNO-A is back-projected to native adaptive cells.

    pred = pred_out

    target = target_out

    area = area_out

    coords_m = coords_out

    U0 = U0_out

    eval_rep = rep

    postprocess_s = 0.0

    secondary_raster_metrics = None



    reference_space, network_output_space, prediction_postprocess = _reference_policy(rep)



    if rep == "adaptive_rasterized":

        native = _native_adaptive_frame(ds, dataset_index)

        if abs(native["time_s"] - actual_time) > 1e-3:

            raise RuntimeError(

                f"time mismatch for {scenario}: "

                f"raster={actual_time}, native={native['time_s']}"

            )



        tp = time.perf_counter()

        pred = apply_raster_to_source_map(native["rmap"], pred_out)

        postprocess_s = time.perf_counter() - tp



        target = native["target"]

        area = native["area"]

        coords_m = native["coords_m"]

        U0 = native["U0"]

        actual_time = native["time_s"]

        eval_rep = "adaptive"



        if pred.shape != target.shape:

            raise RuntimeError(

                f"native FNO-A shape mismatch: pred {pred.shape}, truth {target.shape}"

            )



    ev = cfg.get("evaluation", {})

    lc = cfg.get("loss", {})

    hwet = float(ev.get("h_wet_m", lc.get("h_wet_m", 1e-3)))

    hvel = float(ev.get("h_velocity_m", lc.get("h_velocity_m", 1e-2)))

    relfloor = float(ev.get("relative_volume_floor_fraction", 1e-3))



    metrics = frame_metrics(

        pred,

        target,

        area,

        coords_m,

        U0,

        h_wet_m=hwet,

        h_velocity_m=hvel,

        relative_volume_floor_fraction=relfloor,

    )

    metrics["runout_abs_error_m"] = abs(metrics["runout_error_m"])



    # Keep FNO-A raster-space metrics only as a labelled secondary diagnostic.

    if rep == "adaptive_rasterized":

        secondary_raster_metrics = frame_metrics(

            pred_out,

            target_out,

            area_out,

            coords_out,

            U0_out,

            h_wet_m=hwet,

            h_velocity_m=hvel,

            relative_volume_floor_fraction=relfloor,

        )

        secondary_raster_metrics["runout_abs_error_m"] = abs(

            secondary_raster_metrics["runout_error_m"]

        )



    # Primitive depth fields from the PRIMARY evaluation state.

    hp, *_ = primitive_from_target(pred, h_wet_m=hwet, h_velocity_m=hvel)

    ht, *_ = primitive_from_target(target, h_wet_m=hwet, h_velocity_m=hvel)



    # Common paper visualization. Adaptive primary fields are rasterized only here.

    manifest_row = _manifest_row(manifest, split, scenario)

    z0_plot, x, y, rmap = _read_topography(

        manifest,

        manifest_row,

        rep,

        pid,

        float(cfg["data"].get("raster_dx", 10.0)),

    )

    ht_plot, hp_plot, z0_plot, x, y = _plot_fields(

        eval_rep,

        ht,

        hp,

        z0_plot,

        x,

        y,

        rmap,

    )



    return {

        "case": case,

        "short": meta["short"],

        "truth_family": meta["truth"],

        "tier": tier,

        "kind": kind,

        "representation": rep,

        "evaluation_representation": eval_rep,

        "reference_space": reference_space,

        "network_output_space": network_output_space,

        "prediction_postprocess": prediction_postprocess,

        "parameters": int(ck.get("parameter_count", -1)),

        "checkpoint": str(checkpoint.resolve()),

        "split": split,

        "scenario": scenario,

        "profile_id": pid,

        "requested_time": float(requested_time),

        "time_s": float(actual_time),

        "inference_s": float(inference_s),

        "postprocess_s": float(postprocess_s),

        "inference_plus_postprocess_s": float(inference_s + postprocess_s),

        "h_wet_m": hwet,

        "metrics": metrics,

        "secondary_raster_metrics": secondary_raster_metrics,

        # Primary evaluation states.

        "pred_state": pred,

        "true_state": target,

        # Raw network-output-space states (different from primary only for FNO-A).

        "pred_output_state": pred_out,

        "true_output_state": target_out,

        # Plotting fields.

        "h_pred_plot": hp_plot,

        "h_true_plot": ht_plot,

        "z0_plot": z0_plot,

        "x": np.asarray(x),

        "y": np.asarray(y),

    }





def main():

    ap = argparse.ArgumentParser(

        description=(

            "Run one or all FNO/GNO/GINO Cartesian/adaptive models on the same "

            "scenario and physical time."

        )

    )

    ap.add_argument("--tier", choices=["shallow", "deep", "deep200", "deep200_wpos4", "deep200_wpos16"], default="deep200")

    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument(

        "--case",

        choices=list(CASES.keys()),

        default=None,

        help=("Run only one model. Omit --case to run all six FNO/GNO/GINO "

              "Cartesian/adaptive models on the same sample."),

    )

    ap.add_argument("--split", default="joint_ood_test")

    ap.add_argument("--scenario", default=None, help="Scenario ID in --split; omitted selects first available scenario deterministically.")

    ap.add_argument("--times", type=float, nargs="+", default=[1000.0])

    ap.add_argument("--hmax", type=float, default=6.0)

    ap.add_argument("--emax", type=float, default=2.5)

    ap.add_argument("--dpi", type=int, default=190)

    ap.add_argument("--out", type=Path, default=None)

    ap.add_argument(

        "--dataset-root",

        type=Path,

        default=None,

        help=(

            "Directory containing the external Morino dataset (the folder that "

            "contains runs/ and profiles/). This is exported as "

            "MORINO_DATASET_ROOT so both ml.data and the plotting code resolve "

            "legacy manifest paths consistently."

        ),

    )

    args = ap.parse_args()



    if args.dataset_root is not None:

        dataset_root = args.dataset_root.expanduser().resolve()

        if not dataset_root.is_dir():

            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        os.environ["MORINO_DATASET_ROOT"] = str(dataset_root)

        print(f"[DATASET ROOT] {dataset_root}")



    dataset_root = Path(os.environ.get("MORINO_DATASET_ROOT") or (PROJECT_ROOT / "output" / "dataset")).expanduser().resolve()
    manifest = dataset_root / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    args.scenario = _select_scenario(manifest, args.split, args.scenario, require_both=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_cases = (

        OrderedDict([(args.case, CASES[args.case])])

        if args.case is not None

        else CASES

    )



    if args.out is None:

        selection_tag = args.case if args.case is not None else "all_models"

        out_root = (

            PROJECT_ROOT

            / "output"

            / "models"

            / args.tier

            / ("same_sample_inference_200" if args.tier == "deep200" else "same_sample_inference")

            / f"{args.split}_{args.scenario}"

            / selection_tag

        )

    else:

        out_root = args.out



    out_root = out_root.resolve()

    out_root.mkdir(parents=True, exist_ok=True)



    all_summary_rows = []



    print(f"[DEVICE] {device}")

    print(

        f"[SAMPLE] tier={args.tier} split={args.split} "

        f"scenario={args.scenario} requested_times={args.times} "

        f"cases={list(selected_cases)}"

    )



    for requested_time in args.times:

        results = OrderedDict()

        print("\n" + "=" * 96)

        print(f"[TIME] requested {requested_time:g} s")

        print("=" * 96)



        for case, meta in selected_cases.items():

            result = run_case(

                case,

                meta,

                args.tier,

                args.seed,

                args.split,

                args.scenario,

                requested_time,

                device,

            )

            results[case] = result

            met = result["metrics"]

            print(

                f"[{result['short']}] actual_t={result['time_s']:.3f}s "

                f"p={result['parameters']} inf={result['inference_s']:.4f}s "

                f"post={result['postprocess_s']:.4f}s "

                f"RMSE={met['h_rmse']:.4f}m R2={met['h_r2']:.4f} "

                f"IoU={met['wet_iou']:.4f} "

                f"runout_abs={met['runout_abs_error_m']:.1f}m "

                f"reference={result['reference_space']}"

            )



        # Same visualization threshold for every model in this invocation.

        hwet_plot = min(r["h_wet_m"] for r in results.values())

        tag = f"t{requested_time:08.2f}s".replace(".", "p")

        tdir = out_root / tag

        tdir.mkdir(parents=True, exist_ok=True)



        for case, r in results.items():

            png = tdir / f"{case}_true_pred_error_footprint.png"

            _draw_2x2(r, png, args.hmax, args.emax, hwet_plot, args.dpi)



            # Persist both the PRIMARY evaluation state and the raw network output.

            np.savez_compressed(

                tdir / f"{case}_inference.npz",

                pred_state=r["pred_state"],

                true_state=r["true_state"],

                pred_output_state=r["pred_output_state"],

                true_output_state=r["true_output_state"],

                h_pred_plot=r["h_pred_plot"],

                h_true_plot=r["h_true_plot"],

                z0_plot=r["z0_plot"],

                x=r["x"],

                y=r["y"],

                requested_time=np.float64(r["requested_time"]),

                actual_time=np.float64(r["time_s"]),

                profile_id=np.int32(r["profile_id"]),

            )



            met = r["metrics"]

            all_summary_rows.append({

                "tier": args.tier,

                "case": case,

                "model": r["short"],

                "representation": r["representation"],

                "reference_space": r["reference_space"],

                "prediction_postprocess": r["prediction_postprocess"],

                "parameters": r["parameters"],

                "split": args.split,

                "scenario": args.scenario,

                "requested_time_s": requested_time,

                "actual_time_s": r["time_s"],

                "inference_s": r["inference_s"],

                "postprocess_s": r["postprocess_s"],

                "inference_plus_postprocess_s": r["inference_plus_postprocess_s"],

                "h_mae": met["h_mae"],

                "h_rmse": met["h_rmse"],

                "h_r2": met["h_r2"],

                "wet_iou": met["wet_iou"],

                "runout_true_m": met["runout_true_m"],

                "runout_pred_m": met["runout_pred_m"],

                "runout_abs_error_m": met["runout_abs_error_m"],

                "fluid_volume_rel_error": met.get("fluid_volume_rel_error", np.nan),

            })



        # Overview requires all six models. Never call it in single-model mode.

        if set(results.keys()) == set(CASES.keys()):

            _draw_overview(

                results,

                tdir / "all_models_same_sample.png",

                args.hmax,

                hwet_plot,

                args.dpi,

            )



        serializable = {}

        for case, r in results.items():

            serializable[case] = {

                k: v

                for k, v in r.items()

                if k not in {

                    "pred_state",

                    "true_state",

                    "pred_output_state",

                    "true_output_state",

                    "h_pred_plot",

                    "h_true_plot",

                    "z0_plot",

                    "x",

                    "y",

                }

            }

        with (tdir / "metrics.json").open("w", encoding="utf-8") as f:

            json.dump(_jsonable(serializable), f, indent=2)



    csv_path = out_root / "same_sample_metrics.csv"

    if all_summary_rows:

        with csv_path.open("w", newline="", encoding="utf-8") as f:

            writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))

            writer.writeheader()

            writer.writerows(all_summary_rows)



    print("\n[SAVE]", out_root)

    print("[SAVE]", csv_path)

    print(

        "\nNOTE: primary metrics follow evaluation.py. "

        "GNO-A and GINO-A are evaluated on native adaptive cells and rasterized "

        "only for visualization. FNO-A is area-overlap back-projected to the "

        "original adaptive cells for primary metrics and primary plots; its raw "

        "raster output is retained in the NPZ and as labelled secondary metrics."

    )





if __name__ == "__main__":

    main()

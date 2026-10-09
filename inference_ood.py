from __future__ import annotations

"""
Same-profile IID/OOD inference comparison for the Morino neural-operator benchmark.

This script is intentionally SEPARATE from ``inference.py`` and reuses
its exact model loading, inference, FNO-A back-projection, metrics and plotting
logic.  It automatically selects, for one profile, one representative scenario
from each of these five categories:

    IID | mu-OOD | xi-OOD | V0-OOD | Joint-OOD

It then runs the requested trained models at the same physical time(s), writes
CSV/JSON summaries, the usual per-model diagnostic figures, and two comparison
figures:

1) ``paper_fno_gino_same_profile_ood.png``
   Cartesian TRUE / FNO-C / Adaptive TRUE / GINO-A across all five categories.

2) ``all_models_same_profile_ood.png``
   Cartesian TRUE / FNO-C / GNO-C / GINO-C / Adaptive TRUE / FNO-A / GNO-A /
   GINO-A across all five categories.

The default IID ranges match the manuscript:
    s_mu : [0.8, 3.2]
    xi   : [180, 420] m s^-2
    V0   : [20,000, 80,000] m^3

Put this file next to ``inference.py`` and run, for example:

    python inference_ood.py --tier deep200 --profile-id 1 --times 1000

If the dataset is external:

    python inference_ood.py --tier deep200 --profile-id 1 --times 1000 \
        --dataset-root /path/to/morino_dataset

Selection is deterministic.  By default the script picks a representative
scenario near the centre of each category rather than simply the first row.
Use ``--selection first`` if you prefer the lexicographically first scenario.
"""

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# -----------------------------------------------------------------------------
# Load the existing inference implementation without modifying it.
# -----------------------------------------------------------------------------

def _load_base_module():
    try:
        return importlib.import_module("infer_same_sample")
    except ModuleNotFoundError:
        here = Path(__file__).resolve().parent
        candidates = sorted(
            p for p in here.glob("infer_same_sample*.py")
            if p.resolve() != Path(__file__).resolve()
        )
        if not candidates:
            raise ModuleNotFoundError(
                "Could not import inference.py. Put this script in the same "
                "directory as inference.py."
            )
        src = candidates[0]
        spec = importlib.util.spec_from_file_location("infer_same_sample", src)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load {src}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["infer_same_sample"] = mod
        spec.loader.exec_module(mod)
        return mod


base = _load_base_module()
CASES = base.CASES
PROJECT_ROOT = base.PROJECT_ROOT


# -----------------------------------------------------------------------------
# Manifest parameter helpers.
# -----------------------------------------------------------------------------

ALIASES = {
    "s_mu": (
        "s_mu", "mu_scale", "friction_scale", "friction_multiplier",
        "mu_multiplier", "mu_factor", "smu",
    ),
    "xi": (
        "xi", "voellmy_xi", "voellmy_xi_m_s2", "xi_m_s2", "xi_ms2",
    ),
    "v0": (
        "V0", "v0", "volume", "volume_m3", "initial_volume",
        "initial_volume_m3", "release_volume", "release_volume_m3",
    ),
}


def _first_nonempty(row: dict, names: Iterable[str]):
    lower = {str(k).lower(): k for k in row.keys()}
    for name in names:
        key = lower.get(str(name).lower())
        if key is None:
            continue
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        return raw
    return None


def _as_float(value, label: str, row: dict) -> float:
    if value is None:
        available = ", ".join(row.keys())
        raise KeyError(
            f"Could not identify manifest field for {label}. Available columns: {available}"
        )
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Manifest value for {label} is not numeric: {value!r}") from exc


def _params(row: dict) -> dict:
    return {
        "s_mu": _as_float(_first_nonempty(row, ALIASES["s_mu"]), "s_mu", row),
        "xi": _as_float(_first_nonempty(row, ALIASES["xi"]), "xi", row),
        "v0": _as_float(_first_nonempty(row, ALIASES["v0"]), "V0", row),
    }


def _profile_id(row: dict) -> int:
    raw = _first_nonempty(row, ("profile_id", "profile", "profile_idx", "profile_number"))
    if raw is None:
        sid = str(row.get("scenario_id", ""))
        if sid.lower().startswith("p") and "_" in sid:
            try:
                return int(sid[1:sid.index("_")])
            except Exception:
                pass
        raise KeyError("Could not identify profile_id in manifest row")
    return int(float(raw))


def _scenario_id(row: dict) -> str:
    sid = _first_nonempty(row, ("scenario_id", "scenario", "case_id"))
    if sid is None:
        raise KeyError("Could not identify scenario_id in manifest")
    return str(sid)


def _outside(v: float, lo: float, hi: float, eps: float = 1e-12) -> bool:
    return v < lo - eps or v > hi + eps


def _ood_flags(p: dict, ranges: dict) -> dict:
    return {
        "mu": _outside(p["s_mu"], *ranges["s_mu"]),
        "xi": _outside(p["xi"], *ranges["xi"]),
        "v0": _outside(p["v0"], *ranges["v0"]),
    }


def _canonical_split(raw: str) -> str:
    s = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "iid": "iid_test",
        "test_iid": "iid_test",
        "single_ood": "single_ood_test",
        "ood_single": "single_ood_test",
        "joint_ood": "joint_ood_test",
        "ood_joint": "joint_ood_test",
    }
    return aliases.get(s, s)


def _category(row: dict, ranges: dict) -> str | None:
    """Return iid, mu_ood, xi_ood, v0_ood, joint_ood, or None."""
    split = _canonical_split(row.get("split", ""))
    p = _params(row)
    flags = _ood_flags(p, ranges)
    n = sum(bool(v) for v in flags.values())

    # Prefer physics-derived classification; use split as a consistency guard.
    if split == "iid_test":
        return "iid" if n == 0 else None

    if split == "single_ood_test":
        if n != 1:
            return None
        if flags["mu"]:
            return "mu_ood"
        if flags["xi"]:
            return "xi_ood"
        return "v0_ood"

    if split == "joint_ood_test":
        # Manuscript design says all 3 are outside for joint OOD.
        return "joint_ood" if n == 3 else None

    # Also understand more explicit split names if a future manifest uses them.
    if "joint" in split and "ood" in split and n >= 2:
        return "joint_ood"
    if "single" in split and "ood" in split and n == 1:
        if flags["mu"]:
            return "mu_ood"
        if flags["xi"]:
            return "xi_ood"
        return "v0_ood"
    if "iid" in split and n == 0:
        return "iid"
    return None


CATEGORY_ORDER = ["iid", "mu_ood", "xi_ood", "v0_ood", "joint_ood"]
CATEGORY_LABEL = {
    "iid": "IID",
    "mu_ood": r"$s_\mu$-OOD",
    "xi_ood": r"$\xi$-OOD",
    "v0_ood": r"$V_0$-OOD",
    "joint_ood": "Joint-OOD",
}


def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _normalised_distance_to_interval_centre(value: float, lo: float, hi: float) -> float:
    span = max(hi - lo, 1e-12)
    return abs(value - 0.5 * (lo + hi)) / span


def _representative_score(category: str, p: dict, ranges: dict) -> float:
    """Smaller = more representative/central within the requested category."""
    dmu = _normalised_distance_to_interval_centre(p["s_mu"], *ranges["s_mu"])
    dxi = _normalised_distance_to_interval_centre(p["xi"], *ranges["xi"])
    dv0 = _normalised_distance_to_interval_centre(p["v0"], *ranges["v0"])

    # Keep non-OOD parameters near the training-domain centre.  For the OOD
    # parameter(s), prefer a moderate extrapolation rather than the most extreme
    # available case.  Distance 0.75 means roughly one-quarter of a training
    # span beyond one edge and tends to select visually representative cases.
    target_ood = 0.75
    if category == "iid":
        return dmu**2 + dxi**2 + dv0**2
    if category == "mu_ood":
        return (dmu - target_ood) ** 2 + dxi**2 + dv0**2
    if category == "xi_ood":
        return dmu**2 + (dxi - target_ood) ** 2 + dv0**2
    if category == "v0_ood":
        return dmu**2 + dxi**2 + (dv0 - target_ood) ** 2
    return (dmu - target_ood) ** 2 + (dxi - target_ood) ** 2 + (dv0 - target_ood) ** 2


def select_scenarios(
    rows: list[dict],
    profile_id: int,
    ranges: dict,
    selection: str,
) -> OrderedDict:
    buckets: dict[str, list[dict]] = {k: [] for k in CATEGORY_ORDER}

    for row in rows:
        try:
            if _profile_id(row) != profile_id:
                continue
            cat = _category(row, ranges)
        except (KeyError, ValueError):
            continue
        if cat in buckets:
            buckets[cat].append(row)

    missing = [k for k in CATEGORY_ORDER if not buckets[k]]
    if missing:
        details = {k: len(v) for k, v in buckets.items()}
        raise RuntimeError(
            f"Profile {profile_id} does not contain all requested categories. "
            f"Missing={missing}; counts={details}. Check the IID ranges or manifest fields."
        )

    out = OrderedDict()
    for cat in CATEGORY_ORDER:
        candidates = buckets[cat]
        if selection == "first":
            row = sorted(candidates, key=_scenario_id)[0]
        else:
            row = min(
                candidates,
                key=lambda r: (_representative_score(cat, _params(r), ranges), _scenario_id(r)),
            )
        out[cat] = row
    return out


# -----------------------------------------------------------------------------
# Checkpoint / manifest discovery.
# -----------------------------------------------------------------------------

def _manifest_from_checkpoint(tier: str, seed: int, preferred_case: str) -> Path:
    ckpt = base._case_checkpoint(tier, preferred_case, seed)
    if not ckpt.is_file():
        # Try any available case to discover the common manifest.
        for case in CASES:
            q = base._case_checkpoint(tier, case, seed)
            if q.is_file():
                ckpt = q
                break
        else:
            raise FileNotFoundError(
                f"No checkpoints found for tier={tier}, seed={seed}. "
                f"Expected under {PROJECT_ROOT / 'output' / 'models' / tier}"
            )

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    manifest = Path(cfg["data"]["manifest"])
    if not manifest.is_absolute():
        manifest = (PROJECT_ROOT / manifest).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    return manifest


# -----------------------------------------------------------------------------
# Plotting.
# -----------------------------------------------------------------------------

def _plot_panel(ax, result: dict, field: np.ndarray, title: str, hmax: float, hwet: float):
    x = result["x"]
    y = result["y"]
    extent = base._extent(x, y)
    shade = base._hillshade(result["z0_plot"], x, y)
    base._base(ax, shade, extent)

    wet = np.asarray(field) >= hwet
    arr = np.ma.masked_where(~base._orient(wet, x, y), base._orient(field, x, y))
    cmap_h = plt.get_cmap("turbo").copy()
    cmap_h.set_bad((0, 0, 0, 0))
    im = ax.imshow(
        arr,
        origin="lower",
        extent=extent,
        cmap=cmap_h,
        vmin=0.0,
        vmax=hmax,
        alpha=0.93,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9.2, weight="semibold")
    return im


def _metric_title(result: dict, short: str) -> str:
    m = result["metrics"]
    return (
        f"{short}\n"
        f"RMSE={m['h_rmse']:.3f} m | IoU={m['wet_iou']:.3f}\n"
        f"runout err.={m['runout_abs_error_m']:.0f} m"
    )


def draw_paper_comparison(all_results: OrderedDict, out_png: Path, hmax: float, dpi: int):
    """4 x 5: Cartesian TRUE/FNO-C and Adaptive TRUE/GINO-A."""
    required = {"fno_cartesian", "gino_adaptive"}
    for cat, d in all_results.items():
        if not required.issubset(d):
            return

    ncol = len(CATEGORY_ORDER)
    fig, axes = plt.subplots(4, ncol, figsize=(4.3 * ncol, 14.3), constrained_layout=True)
    im = None

    for j, cat in enumerate(CATEGORY_ORDER):
        fc = all_results[cat]["fno_cartesian"]
        ga = all_results[cat]["gino_adaptive"]
        hwet = min(fc["h_wet_m"], ga["h_wet_m"])

        p = all_results[cat]["_params"]
        scenario = all_results[cat]["_scenario"]
        axes[0, j].text(
            0.5, 1.17,
            f"{CATEGORY_LABEL[cat]}  |  {scenario}\n"
            f"$s_\\mu$={p['s_mu']:.3g}, $\\xi$={p['xi']:.3g}, $V_0$={p['v0']:.0f} m$^3$",
            transform=axes[0, j].transAxes,
            ha="center", va="bottom", fontsize=10.5, weight="bold",
        )

        im = _plot_panel(axes[0, j], fc, fc["h_true_plot"], "TRUE — Cartesian", hmax, hwet)
        _plot_panel(axes[1, j], fc, fc["h_pred_plot"], _metric_title(fc, "FNO-C"), hmax, hwet)
        _plot_panel(axes[2, j], ga, ga["h_true_plot"], "TRUE — Adaptive", hmax, hwet)
        _plot_panel(axes[3, j], ga, ga["h_pred_plot"], _metric_title(ga, "GINO-A"), hmax, hwet)

    # Keep labels only on the outer panels to make the 5-column figure cleaner.
    for i in range(4):
        for j in range(ncol):
            if j > 0:
                axes[i, j].set_ylabel("")
            if i < 3:
                axes[i, j].set_xlabel("")

    if im is not None:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.018,
            pad=0.025,
            label="Flow thickness h [m]",
        )

    first = all_results[CATEGORY_ORDER[0]]["fno_cartesian"]
    fig.suptitle(
        f"Same-profile parameter generalization — profile {first['profile_id']} — "
        f"t≈{first['requested_time']:.0f} s",
        fontsize=15, weight="semibold",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def draw_all_models_comparison(all_results: OrderedDict, out_png: Path, hmax: float, dpi: int):
    """8 x 5 full architecture comparison with branch-specific truths."""
    required = set(CASES)
    for cat, d in all_results.items():
        if not required.issubset(d):
            return

    row_specs = [
        ("truth_c", "TRUE — Cartesian"),
        ("fno_cartesian", "FNO-C"),
        ("gno_cartesian", "GNO-C"),
        ("gino_cartesian", "GINO-C"),
        ("truth_a", "TRUE — Adaptive"),
        ("fno_adaptive", "FNO-A"),
        ("gno_adaptive", "GNO-A"),
        ("gino_adaptive", "GINO-A"),
    ]

    nrow, ncol = len(row_specs), len(CATEGORY_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.15 * ncol, 3.15 * nrow), constrained_layout=True)
    im = None

    for j, cat in enumerate(CATEGORY_ORDER):
        d = all_results[cat]
        fc = d["fno_cartesian"]
        fa = d["fno_adaptive"]
        hwet = min(r["h_wet_m"] for k, r in d.items() if k in CASES)
        p = d["_params"]
        scenario = d["_scenario"]

        axes[0, j].text(
            0.5, 1.20,
            f"{CATEGORY_LABEL[cat]} | {scenario}\n"
            f"$s_\\mu$={p['s_mu']:.3g}, $\\xi$={p['xi']:.3g}, $V_0$={p['v0']:.0f}",
            transform=axes[0, j].transAxes,
            ha="center", va="bottom", fontsize=9.8, weight="bold",
        )

        for i, (key, label) in enumerate(row_specs):
            if key == "truth_c":
                r, field, title = fc, fc["h_true_plot"], label
            elif key == "truth_a":
                r, field, title = fa, fa["h_true_plot"], label
            else:
                r = d[key]
                field = r["h_pred_plot"]
                title = _metric_title(r, label)
            im = _plot_panel(axes[i, j], r, field, title, hmax, hwet)

    for i in range(nrow):
        for j in range(ncol):
            if j > 0:
                axes[i, j].set_ylabel("")
            if i < nrow - 1:
                axes[i, j].set_xlabel("")

    if im is not None:
        fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.012,
            pad=0.018,
            label="Flow thickness h [m]",
        )

    first = all_results[CATEGORY_ORDER[0]]["fno_cartesian"]
    fig.suptitle(
        f"FNO / GNO / GINO — same profile {first['profile_id']} — "
        f"IID to joint-OOD — t≈{first['requested_time']:.0f} s",
        fontsize=15, weight="semibold",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run same-profile IID, single-parameter OOD and joint-OOD inference "
            "using the exact inference/evaluation logic in inference.py."
        )
    )
    ap.add_argument("--tier", choices=["shallow", "deep", "deep200"], default="deep200")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--profile-id", type=int, default=1)
    ap.add_argument("--times", type=float, nargs="+", default=[1000.0])
    ap.add_argument("--hmax", type=float, default=6.0)
    ap.add_argument("--emax", type=float, default=2.5)
    ap.add_argument("--dpi", type=int, default=190)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument(
        "--selection",
        choices=["representative", "first"],
        default="representative",
        help="How to pick one scenario per category for the selected profile.",
    )
    ap.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASES.keys()),
        default=list(CASES.keys()),
        help="Models to run. Default: all six.",
    )

    # Manuscript IID domain. Expose it in CLI so the script remains reproducible
    # if the benchmark ranges are changed later.
    ap.add_argument("--mu-iid", type=float, nargs=2, default=[0.8, 3.2], metavar=("LO", "HI"))
    ap.add_argument("--xi-iid", type=float, nargs=2, default=[180.0, 420.0], metavar=("LO", "HI"))
    ap.add_argument("--v0-iid", type=float, nargs=2, default=[20000.0, 80000.0], metavar=("LO", "HI"))
    args = ap.parse_args()

    if args.dataset_root is not None:
        root = args.dataset_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        os.environ["MORINO_DATASET_ROOT"] = str(root)
        print(f"[DATASET ROOT] {root}")

    ranges = {
        "s_mu": tuple(map(float, args.mu_iid)),
        "xi": tuple(map(float, args.xi_iid)),
        "v0": tuple(map(float, args.v0_iid)),
    }

    manifest = _manifest_from_checkpoint(args.tier, args.seed, args.cases[0])
    rows = _read_manifest(manifest)
    selected = select_scenarios(rows, args.profile_id, ranges, args.selection)

    print(f"[MANIFEST] {manifest}")
    print(f"[PROFILE] {args.profile_id}")
    print("[SELECTED SCENARIOS]")
    for cat, row in selected.items():
        p = _params(row)
        print(
            f"  {CATEGORY_LABEL[cat]:>10s}: split={row.get('split')} "
            f"scenario={_scenario_id(row)}  s_mu={p['s_mu']:.6g} "
            f"xi={p['xi']:.6g} V0={p['v0']:.6g}"
        )

    if args.out is None:
        out_root = (
            PROJECT_ROOT / "output" / "models" / args.tier /
            "same_profile_ood" / f"profile_{args.profile_id:02d}"
        )
    else:
        out_root = args.out
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    with (out_root / "selected_scenarios.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                cat: {
                    "split": row.get("split"),
                    "scenario": _scenario_id(row),
                    "profile_id": _profile_id(row),
                    **_params(row),
                }
                for cat, row in selected.items()
            },
            f,
            indent=2,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    summary_rows = []

    for requested_time in args.times:
        print("\n" + "=" * 110)
        print(f"[TIME] requested {requested_time:g} s")
        print("=" * 110)

        all_results = OrderedDict()

        for cat, row in selected.items():
            split = str(row.get("split"))
            scenario = _scenario_id(row)
            p = _params(row)
            cat_results = OrderedDict()
            cat_results["_scenario"] = scenario
            cat_results["_split"] = split
            cat_results["_params"] = p

            print(f"\n[{CATEGORY_LABEL[cat]}] {split} / {scenario}")

            for case in args.cases:
                meta = CASES[case]
                r = base.run_case(
                    case,
                    meta,
                    args.tier,
                    args.seed,
                    split,
                    scenario,
                    requested_time,
                    device,
                )
                cat_results[case] = r
                m = r["metrics"]
                print(
                    f"  [{r['short']}] t={r['time_s']:.3f}s "
                    f"RMSE={m['h_rmse']:.4f} m R2={m['h_r2']:.4f} "
                    f"IoU={m['wet_iou']:.4f} runout_abs={m['runout_abs_error_m']:.1f} m"
                )

                summary_rows.append({
                    "tier": args.tier,
                    "profile_id": args.profile_id,
                    "category": cat,
                    "category_label": CATEGORY_LABEL[cat].replace("$", ""),
                    "split": split,
                    "scenario": scenario,
                    "s_mu": p["s_mu"],
                    "xi": p["xi"],
                    "V0_m3": p["v0"],
                    "model_case": case,
                    "model": r["short"],
                    "representation": r["representation"],
                    "reference_space": r["reference_space"],
                    "requested_time_s": requested_time,
                    "actual_time_s": r["time_s"],
                    "h_mae_m": m["h_mae"],
                    "h_rmse_m": m["h_rmse"],
                    "h_r2": m["h_r2"],
                    "wet_iou": m["wet_iou"],
                    "runout_true_m": m["runout_true_m"],
                    "runout_pred_m": m["runout_pred_m"],
                    "runout_abs_error_m": m["runout_abs_error_m"],
                    "fluid_volume_rel_error": m.get("fluid_volume_rel_error", np.nan),
                    "inference_s": r["inference_s"],
                    "postprocess_s": r["postprocess_s"],
                    "inference_plus_postprocess_s": r["inference_plus_postprocess_s"],
                })

            all_results[cat] = cat_results

        tag = f"t{requested_time:08.2f}s".replace(".", "p")
        tdir = out_root / tag
        tdir.mkdir(parents=True, exist_ok=True)

        # Persist normal per-model diagnostic figures/NPZ files as well.
        for cat in CATEGORY_ORDER:
            d = all_results[cat]
            scenario = d["_scenario"]
            cat_dir = tdir / f"{cat}_{scenario}"
            cat_dir.mkdir(parents=True, exist_ok=True)
            model_results = [d[k] for k in args.cases if k in d]
            if not model_results:
                continue
            hwet_plot = min(r["h_wet_m"] for r in model_results)

            for case in args.cases:
                if case not in d:
                    continue
                r = d[case]
                base._draw_2x2(
                    r,
                    cat_dir / f"{case}_true_pred_error_footprint.png",
                    args.hmax,
                    args.emax,
                    hwet_plot,
                    args.dpi,
                )
                np.savez_compressed(
                    cat_dir / f"{case}_inference.npz",
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

        # Main figures. They are emitted only when the required models were run.
        draw_paper_comparison(
            all_results,
            tdir / "paper_fno_gino_same_profile_ood.png",
            args.hmax,
            args.dpi,
        )
        draw_all_models_comparison(
            all_results,
            tdir / "all_models_same_profile_ood.png",
            args.hmax,
            args.dpi,
        )

    csv_path = out_root / "same_profile_ood_metrics.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\n[SAVE]", out_root)
    print("[SAVE]", csv_path)
    print("[SAVE]", out_root / "selected_scenarios.json")
    print(
        "\nNOTE: every category uses the same profile_id, but necessarily a different "
        "scenario_id. Primary metrics remain exactly those of inference.py / "
        "evaluation.py, including native adaptive evaluation and FNO-A "
        "area-overlap back-projection."
    )


if __name__ == "__main__":
    main()

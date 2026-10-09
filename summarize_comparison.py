from __future__ import annotations

"""Create accuracy and compute-cost tables for the six-model benchmark."""

import argparse
import csv
import json
from pathlib import Path

CASES = (
    "fno_cartesian", "gno_cartesian", "gino_cartesian",
    "fno_adaptive", "gno_adaptive", "gino_adaptive",
)
SPLITS = ("validation", "iid_test", "single_ood_test", "joint_ood_test")


def metric(block, name):
    x = block.get(name)
    if isinstance(x, dict):
        return x.get("mean")
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[SAVE]", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["shallow", "deep", "deep200"], required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    root = Path("output") / "models" / args.tier
    rows = []
    resource_rows = []
    missing = []

    for case in CASES:
        run = root / f"{case}_seed{args.seed}"
        sp = run / "evaluation" / "summary.json"
        tp = run / "training_summary.json"
        if not sp.is_file():
            missing.append(str(sp))
            continue
        s = json.loads(sp.read_text(encoding="utf-8"))
        t = json.loads(tp.read_text(encoding="utf-8")) if tp.is_file() else {}
        split_timings = {}

        for split in SPLITS:
            block = s.get("frame_metrics_post_t0", {}).get(split, {})
            traj = s.get("trajectory_metrics", {}).get(split, {})
            timing = s.get("timing", {}).get(split, {})
            split_timings[split] = timing.get("mean_inference_s_per_frame")
            rows.append({
                "tier": args.tier,
                "case": case,
                "kind": s.get("kind"),
                "representation": s.get("representation"),
                "reference_space": s.get("reference_space"),
                "prediction_postprocess": s.get("prediction_postprocess"),
                "parameters": s.get("parameters", t.get("parameters")),
                "split": split,
                "best_val": t.get("best_val"),
                "best_val_space": (
                    "adaptive_rasterized_training_loss" if s.get("representation") == "adaptive_rasterized"
                    else "adaptive_native_training_loss" if s.get("representation") == "adaptive"
                    else "cartesian_native_training_loss"
                ),
                "training_wall_s": t.get("training_wall_s"),
                "training_wall_min": t.get("training_wall_min", (t.get("training_wall_s") or 0) / 60.0 if t.get("training_wall_s") is not None else None),
                "training_wall_h": t.get("training_wall_h", (t.get("training_wall_s") or 0) / 3600.0 if t.get("training_wall_s") is not None else None),
                "mean_epoch_s": t.get("mean_epoch_s"),
                "train_frames_per_s": t.get("train_frames_per_s"),
                "peak_cuda_mib": t.get("peak_cuda_mib"),
                "h_mae": metric(block, "h_mae"),
                "h_rmse": metric(block, "h_rmse"),
                "h_r2": metric(block, "h_r2"),
                "wet_iou": metric(block, "wet_iou"),
                "wet_iou_10mm": metric(block, "wet_iou_10mm"),
                "runout_abs_error_m": metric(block, "runout_abs_error_m"),
                "fluid_volume_rel_error": metric(block, "fluid_volume_rel_error"),
                "speed_mae_union_10mm": metric(block, "speed_mae"),
                "speed_rmse_union_10mm": metric(block, "speed_rmse"),
                "speed_bothwet_mae_50mm": metric(block, "speed_bothwet_mae_50mm"),
                "speed_bothwet_rmse_50mm": metric(block, "speed_bothwet_rmse_50mm"),
                "velocity_bothwet_coverage_50mm": metric(block, "velocity_bothwet_coverage_50mm"),
                "arrival_time_mae_s": metric(traj, "arrival_time_mae_s"),
                "arrival_time_bias_s": metric(traj, "arrival_time_bias_s"),
                "mean_inference_s_per_frame": timing.get("mean_inference_s_per_frame"),
            })

        resource_rows.append({
            "tier": args.tier,
            "case": case,
            "kind": s.get("kind"),
            "representation": s.get("representation"),
            "parameters": s.get("parameters", t.get("parameters")),
            "epochs_completed": t.get("epochs_completed"),
            "samples_per_epoch": t.get("samples_per_epoch"),
            "effective_batch": t.get("effective_batch"),
            "training_wall_s": t.get("training_wall_s"),
            "training_wall_min": t.get("training_wall_min", (t.get("training_wall_s") or 0) / 60.0 if t.get("training_wall_s") is not None else None),
            "training_wall_h": t.get("training_wall_h", (t.get("training_wall_s") or 0) / 3600.0 if t.get("training_wall_s") is not None else None),
            "mean_epoch_s": t.get("mean_epoch_s"),
            "mean_train_s_per_epoch": t.get("mean_train_s_per_epoch"),
            "mean_validation_s_per_epoch": t.get("mean_validation_s_per_epoch"),
            "train_frames_per_s": t.get("train_frames_per_s"),
            "peak_cuda_mib": t.get("peak_cuda_mib"),
            "validation_inference_ms_per_frame": (split_timings.get("validation") * 1000.0) if split_timings.get("validation") is not None else None,
            "iid_inference_ms_per_frame": (split_timings.get("iid_test") * 1000.0) if split_timings.get("iid_test") is not None else None,
            "single_ood_inference_ms_per_frame": (split_timings.get("single_ood_test") * 1000.0) if split_timings.get("single_ood_test") is not None else None,
            "joint_ood_inference_ms_per_frame": (split_timings.get("joint_ood_test") * 1000.0) if split_timings.get("joint_ood_test") is not None else None,
            "gpu": t.get("gpu"),
        })

    out = root / f"comparison_{args.tier}_seed{args.seed}.csv"
    resource_out = root / f"compute_cost_{args.tier}_seed{args.seed}.csv"
    write_csv(out, rows)
    write_csv(resource_out, resource_rows)

    if rows:
        print("\nJOINT-OOD compact view\n")
        by_case_resource = {r["case"]: r for r in resource_rows}
        for r in rows:
            if r["split"] != "joint_ood_test":
                continue
            def fmt(x, n=4):
                return "nan" if x is None else f"{x:.{n}f}"
            rr = by_case_resource.get(r["case"], {})
            print(
                f"{r['case']:18s} p={int(r['parameters'] or -1):7d} "
                f"RMSE={fmt(r['h_rmse'])} R2={fmt(r['h_r2'])} "
                f"IoU={fmt(r['wet_iou'])} runout={fmt(r['runout_abs_error_m'],1)}m "
                f"speedMAE5cm={fmt(r.get('speed_bothwet_mae_50mm'),3)}m/s "
                f"arrivalMAE={fmt(r.get('arrival_time_mae_s'),1)}s "
                f"arrivalBias={fmt(r.get('arrival_time_bias_s'),1)}s "
                f"train={fmt(rr.get('training_wall_min'),1)}min "
                f"infer={fmt(rr.get('joint_ood_inference_ms_per_frame'),2)}ms/frame"
            )

    if resource_rows:
        print("\nCOMPUTE-COST view\n")
        for r in resource_rows:
            def fmt(x, n=2):
                return "nan" if x is None else f"{x:.{n}f}"
            print(
                f"{r['case']:18s} train={fmt(r.get('training_wall_min'),1):>8s} min "
                f"epoch={fmt(r.get('mean_epoch_s'),1):>7s} s "
                f"peak={fmt(r.get('peak_cuda_mib'),0):>6s} MiB "
                f"jointOOD={fmt(r.get('joint_ood_inference_ms_per_frame'),2):>7s} ms/frame"
            )

    if missing:
        print("\n[MISSING] Evaluation summaries not found:")
        for p in missing:
            print(" -", p)


if __name__ == "__main__":
    main()

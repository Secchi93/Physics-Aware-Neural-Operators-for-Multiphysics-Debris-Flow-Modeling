from __future__ import annotations

"""Static audit of the six-model comparison protocol.

This script intentionally does not import PyTorch/NeuralOperator. It verifies
that configuration drift cannot silently make one model receive a different
training budget, split, loss, optimizer or evaluation threshold.
"""

import argparse
from pathlib import Path
import yaml

CASES = (
    "fno_cartesian", "gno_cartesian", "gino_cartesian",
    "fno_adaptive", "gno_adaptive", "gino_adaptive",
)
FAMILY_PAIRS = {
    "fno": ("fno_cartesian", "fno_adaptive"),
    "gno": ("gno_cartesian", "gno_adaptive"),
    "gino": ("gino_cartesian", "gino_adaptive"),
}
EXPECTED_REP = {
    "fno_cartesian": "cartesian",
    "gno_cartesian": "cartesian_points",
    "gino_cartesian": "cartesian_points",
    "fno_adaptive": "adaptive_rasterized",
    "gno_adaptive": "adaptive",
    "gino_adaptive": "adaptive",
}
COMMON_TRAIN_KEYS = (
    "epochs", "samples_per_epoch", "lr", "min_lr", "weight_decay",
    "grad_clip", "validation_max_batches", "num_workers",
    "validation_num_workers", "prefetch_factor",
)
COMMON_DATA_KEYS = (
    "manifest", "target_scales", "frame_stride", "validation_frame_stride",
    "raster_dx", "cache_static_items",
)


def load_tier(tier: str) -> dict[str, dict]:
    root = Path("configs_ml") / "comparison" / tier
    expected_files = {f"{case}.yaml" for case in CASES}
    found_files = {p.name for p in root.glob("*.yaml")}
    if found_files != expected_files:
        missing = sorted(expected_files - found_files)
        unexpected = sorted(found_files - expected_files)
        raise AssertionError(
            f"Tier {tier!r} must contain exactly the six active configs. "
            f"Missing={missing}, unexpected={unexpected}"
        )
    out = {}
    for case in CASES:
        path = root / f"{case}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required config: {path}")
        out[case] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return out


def assert_equal(label: str, values: dict[str, object]) -> None:
    serial = {k: repr(v) for k, v in values.items()}
    if len(set(serial.values())) != 1:
        pretty = "\n".join(f"  {k}: {v}" for k, v in values.items())
        raise AssertionError(f"Protocol mismatch for {label}:\n{pretty}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["shallow", "deep", "deep200", "deep200_v020"], required=True)
    args = ap.parse_args()
    cfgs = load_tier(args.tier)

    assert_equal("seed", {c: cfgs[c].get("seed") for c in CASES})
    for key in COMMON_DATA_KEYS:
        assert_equal(f"data.{key}", {c: cfgs[c]["data"].get(key) for c in CASES})
    for key in COMMON_TRAIN_KEYS:
        assert_equal(f"training.{key}", {c: cfgs[c]["training"].get(key) for c in CASES})
    assert_equal("loss", {c: cfgs[c]["loss"] for c in CASES})
    assert_equal("evaluation", {c: cfgs[c]["evaluation"] for c in CASES})

    effective = {
        c: int(cfgs[c]["training"]["batch_size"])
        * int(cfgs[c]["training"]["gradient_accumulation"])
        for c in CASES
    }
    assert_equal("effective_batch", effective)

    for case, rep in EXPECTED_REP.items():
        got = str(cfgs[case]["data"]["representation"])
        if got != rep:
            raise AssertionError(f"{case}: expected representation={rep!r}, got {got!r}")

    for family, (cart, adapt) in FAMILY_PAIRS.items():
        a = cfgs[cart]["model"]
        b = cfgs[adapt]["model"]
        if a != b:
            raise AssertionError(
                f"{family.upper()} architecture differs between Cartesian and adaptive configs.\n"
                f"Cartesian: {a}\nAdaptive: {b}"
            )
        if a.get("kind") != family:
            raise AssertionError(f"{cart}: model.kind must be {family!r}")

    gno = cfgs["gno_cartesian"]["model"]
    if gno.get("transform_type") != "nonlinear":
        raise AssertionError("GNO benchmark must use transform_type='nonlinear'")
    gino = cfgs["gino_cartesian"]["model"]
    if gino.get("in_gno_transform_type") != "nonlinear" or gino.get("out_gno_transform_type") != "nonlinear":
        raise AssertionError("GINO benchmark must use nonlinear input and output GNO transforms")

    # Fair geometric sampling budget: GNO's local integral and both geometric
    # transforms inside GINO may have different radii because the operators map
    # different spaces, but each query is capped to the same nearest-K budget.
    k_budget = {
        "GNO": int(gno.get("max_neighbors", -1)),
        "GINO-in": int(gino.get("in_gno_max_neighbors", -1)),
        "GINO-out": int(gino.get("out_gno_max_neighbors", -1)),
    }
    assert_equal("geometric nearest-K budget", k_budget)
    if next(iter(k_budget.values())) <= 0:
        raise AssertionError(f"Invalid geometric K budget: {k_budget}")

    print(f"[PROTOCOL PASS] tier={args.tier}")
    print(f"  cases: {len(CASES)}")
    print(f"  seed: {cfgs[CASES[0]]['seed']}")
    print(f"  epochs: {cfgs[CASES[0]]['training']['epochs']}")
    print(f"  samples/epoch: {cfgs[CASES[0]]['training']['samples_per_epoch']}")
    print(f"  validation frames/epoch: {cfgs[CASES[0]]['training']['validation_max_batches']} (validation batch=1)")
    print(f"  effective batch: {effective[CASES[0]]}")
    print("  common loss/optimizer/scheduler inputs: verified")
    print("  Cartesian/adaptive architecture identity within each family: verified")
    print("  GNO/GINO geometric transforms: nonlinear")
    print(f"  common geometric neighbor cap: K={k_budget['GNO']} (GNO, GINO-in, GINO-out)")


if __name__ == "__main__":
    main()

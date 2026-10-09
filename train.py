from __future__ import annotations

"""Unified trainer for the controlled six-model neural-operator study.

The same trainer, loss, normalization, splits and optimization logic are used
for all six workflows:

    FNO-C  : Cartesian solver -> regular raster -> FNO
    FNO-A  : adaptive solver -> conservative 10-m raster -> FNO
    GNO-C  : Cartesian solver -> exact points -> pure GNO
    GNO-A  : adaptive solver -> native irregular cells -> pure GNO
    GINO-C : Cartesian solver -> exact points -> GINO
    GINO-A : adaptive solver -> native irregular cells -> GINO

The point-model batches are profile-bucketed so GNO and GINO see one shared
geometry per minibatch. FNO uses shape-bucketed raster batches.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from ml.data import MorinoFrameDataset
from ml.losses import LossConfig, operator_loss
from ml.models import build_model, count_parameters
from ml.normalization import load_scales
from ml.training_preflight import check_training_inputs

LOSS_VERSION = "physics_aware_velocity_v11_from_scratch"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def device_for_run() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("cpu")


def validate_representation(kind: str, rep: str) -> None:
    if kind == "fno" and rep not in ("cartesian", "adaptive_rasterized"):
        raise ValueError("FNO requires data.representation: cartesian or adaptive_rasterized")
    if kind in ("gno", "gino") and rep not in ("cartesian_points", "adaptive"):
        raise ValueError(f"{kind.upper()} requires data.representation: cartesian_points or adaptive")
    if kind not in ("fno", "gno", "gino"):
        raise ValueError(f"Unsupported model kind: {kind!r}")


class LeanView(Dataset):
    """Keep only tensors needed by training and geometry bucketing."""

    def __init__(self, base: MorinoFrameDataset, kind: str):
        self.base = base
        self.kind = kind

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if self.kind == "fno":
            return {k: item[k] for k in ("x", "y", "area")}
        return {k: item[k] for k in ("x", "y", "area", "coords", "profile_id")}


class ShapeBucketBatchSampler(Sampler):
    """Replacement sampling with identical raster shape inside each FNO batch."""

    def __init__(self, dataset, batch_size: int, num_samples: int, generator):
        if num_samples % batch_size:
            raise ValueError("samples_per_epoch must be divisible by batch_size")
        source = dataset.base if isinstance(dataset, LeanView) else dataset
        groups = defaultdict(list)
        for index, ref in enumerate(source.frames):
            groups[tuple(source.row_spatial_shapes[ref.row_index])].append(index)
        if not groups:
            raise ValueError("Empty FNO shape groups")
        self.shapes = sorted(groups)
        self.indices = [torch.tensor(groups[s], dtype=torch.long) for s in self.shapes]
        self.weights = torch.tensor([len(v) for v in self.indices], dtype=torch.double)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_samples) // int(batch_size)
        self.generator = generator

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        shape_draws = torch.multinomial(
            self.weights, self.num_batches, replacement=True, generator=self.generator
        ).tolist()
        for sid in shape_draws:
            pool = self.indices[sid]
            pos = torch.randint(len(pool), (self.batch_size,), generator=self.generator)
            yield pool[pos].tolist()


class ProfileBucketBatchSampler(Sampler):
    """Replacement sampling with a fixed point geometry inside each GNO/GINO batch."""

    def __init__(self, dataset, batch_size: int, num_samples: int, generator):
        if num_samples % batch_size:
            raise ValueError("samples_per_epoch must be divisible by batch_size")
        source = dataset.base if isinstance(dataset, LeanView) else dataset
        groups = defaultdict(list)
        for index, ref in enumerate(source.frames):
            pid = int(source.rows[ref.row_index]["profile_id"])
            groups[pid].append(index)
        if not groups:
            raise ValueError("Empty GNO profile groups")
        self.pids = sorted(groups)
        self.indices = [torch.tensor(groups[p], dtype=torch.long) for p in self.pids]
        self.weights = torch.tensor([len(v) for v in self.indices], dtype=torch.double)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_samples) // int(batch_size)
        self.generator = generator

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        group_draws = torch.multinomial(
            self.weights, self.num_batches, replacement=True, generator=self.generator
        ).tolist()
        for gid in group_draws:
            pool = self.indices[gid]
            pos = torch.randint(len(pool), (self.batch_size,), generator=self.generator)
            yield pool[pos].tolist()


class DeviceAverages:
    def __init__(self):
        self.values = {}
        self.count = 0

    def add(self, parts):
        for key, value in parts.items():
            self.values[key] = self.values[key] + value if key in self.values else value.clone()
        self.count += 1

    def result(self):
        if not self.count:
            return {}
        keys = sorted(self.values)
        vals = torch.stack([self.values[k] / self.count for k in keys]).cpu().tolist()
        return dict(zip(keys, map(float, vals)))


def loss_config(cfg: dict) -> LossConfig:
    lc = cfg.get("loss", {})
    return LossConfig(
        wet_weight=float(lc.get("wet_weight", 8.0)),
        lambda_early_front=float(lc.get("lambda_early_front", 0.0)),
        early_front_hard_fraction=float(lc.get("early_front_hard_fraction", 0.02)),
        h_wet_m=float(lc.get("h_wet_m", 1e-3)),
        wet_temperature_m=float(lc.get("wet_temperature_m", 2e-4)),
        max_support_class_weight=float(lc.get("max_support_class_weight", 50.0)),
        lambda_mass=float(lc.get("lambda_mass", 0.05)),
        lambda_wet=float(lc.get("lambda_wet", 0.20)),
        lambda_dry_depth=float(lc.get("lambda_dry_depth", 0.05)),
        lambda_dry_momentum=float(lc.get("lambda_dry_momentum", 0.02)),
        lambda_admissible=float(lc.get("lambda_admissible", 0.05)),
        dry_depth_tolerance_m=float(lc.get("dry_depth_tolerance_m", 2.5e-4)),
        lambda_velocity=float(lc.get("lambda_velocity", 0.05)),
        lambda_momentum_consistency=float(lc.get("lambda_momentum_consistency", 0.02)),
        lambda_admissible_l1=float(lc.get("lambda_admissible_l1", 0.02)),
        velocity_min_depth_m=float(lc.get("velocity_min_depth_m", 0.05)),
        velocity_scale_ms=float(lc.get("velocity_scale_ms", 2.0)),
        velocity_huber_delta_ms=float(lc.get("velocity_huber_delta_ms", 1.0)),
        rho_f_kg_m3=float(lc.get("rho_f_kg_m3", 1000.0)),
        rho_s_kg_m3=float(lc.get("rho_s_kg_m3", 2650.0)),
    )


def forward_batch(model, kind: str, batch: dict, device: torch.device):
    x = batch["x"].to(device, non_blocking=True)
    if kind == "fno":
        pred = model(x)
        target = batch["y"].to(device, non_blocking=True)
        area = batch["area"].to(device, non_blocking=True)
        return pred, target, area

    target = batch["y"].to(device, non_blocking=True).permute(0, 2, 1)
    area = batch["area"].to(device, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    profile_id = int(batch["profile_id"][0])
    pred = model(x=x, coords=coords, area=area, profile_id=profile_id).permute(0, 2, 1)
    return pred, target, area


def validate(model, kind, loader, device, scales, lc, max_batches: int, progress_every: int = 0):
    model.eval()
    avg = DeviceAverages()
    t0 = time.perf_counter()
    with torch.no_grad():
        for ib, batch in enumerate(loader, 1):
            if ib > max_batches:
                break
            pred, target, area = forward_batch(model, kind, batch, device)
            _, parts = operator_loss(pred, target, area, scales, lc, return_tensor_parts=True)
            avg.add(parts)
            if progress_every and (ib == 1 or ib % progress_every == 0 or ib == max_batches):
                elapsed = time.perf_counter() - t0
                remaining = elapsed / max(ib, 1) * max(max_batches - ib, 0)
                print(f"[VAL] batch={ib}/{max_batches} elapsed={elapsed:.1f}s remaining~{remaining:.1f}s")
    out = avg.result()
    return float(out.get("total", float("inf"))), out


def make_loaders(cfg: dict, kind: str, rep: str, scale_np: np.ndarray, seed: int, run_fraction: float, device):
    dc = cfg["data"]
    tr = cfg["training"]
    manifest = os.path.abspath(dc["manifest"])
    common = dict(
        target_scale=scale_np,
        raster_dx=float(dc.get("raster_dx", 10.0)),
        cache_static_items=int(dc.get("cache_static_items", 12)),
    )
    train_base = MorinoFrameDataset(
        manifest, "train", rep,
        frame_stride=int(dc.get("frame_stride", 2)),
        run_fraction=run_fraction,
        seed=seed,
        **common,
    )
    val_base = MorinoFrameDataset(
        manifest, "validation", rep,
        frame_stride=int(dc.get("validation_frame_stride", 2)),
        **common,
    )
    train_ds = LeanView(train_base, kind)
    val_ds = LeanView(val_base, kind)

    batch_size = int(tr.get("batch_size", 1))
    samples = int(tr.get("samples_per_epoch", 256))
    gen = torch.Generator().manual_seed(seed)
    if kind == "fno":
        sampler = ShapeBucketBatchSampler(train_ds, batch_size, samples, gen)
    else:
        sampler = ProfileBucketBatchSampler(train_ds, batch_size, samples, gen)

    nw = int(tr.get("num_workers", 0))
    vw = int(tr.get("validation_num_workers", 0))
    train_opts = {"num_workers": nw, "pin_memory": device.type == "cuda"}
    val_opts = {"num_workers": vw, "pin_memory": device.type == "cuda"}
    if nw > 0:
        train_opts.update(persistent_workers=True, prefetch_factor=int(tr.get("prefetch_factor", 2)))
    if vw > 0:
        val_opts.update(persistent_workers=True, prefetch_factor=int(tr.get("prefetch_factor", 2)))

    train_loader = DataLoader(train_ds, batch_sampler=sampler, **train_opts)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **val_opts)
    return train_loader, val_loader, sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-fraction", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--inspect-only", action="store_true", help="Build model and print parameter count only")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--benchmark-batches", type=int, default=None)
    ap.add_argument("--overwrite-run", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--stop-after-epoch", type=int, default=None)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--validation-progress-every", type=int, default=10)
    ap.add_argument("--profile-first-batch", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 1234) if args.seed is None else args.seed)
    # The persisted resolved config must describe the actual run, including a
    # command-line seed override used by repeated-seed experiments.
    cfg["seed"] = seed
    seed_all(seed)
    device = device_for_run()

    mc = cfg["model"]
    kind = str(mc.get("kind", "fno")).lower()
    rep = str(cfg["data"]["representation"]).lower()
    validate_representation(kind, rep)

    manifest = os.path.abspath(cfg["data"]["manifest"])
    scales_path = os.path.abspath(cfg["data"]["target_scales"])
    check_training_inputs(
        manifest, rep, scales_path,
        full_run=args.run_fraction == 1.0,
        seed=seed,
        run_fraction=args.run_fraction,
    )
    if args.preflight_only:
        return

    model = build_model(kind, 13, 8, mc).to(device)
    nparams = count_parameters(model)
    tier = str(cfg.get("experiment", {}).get("tier", "unspecified"))
    label = str(cfg.get("experiment", {}).get("label", Path(args.config).stem))
    print(f"[MODEL] label={label} tier={tier} kind={kind} parameters={nparams:,} device={device}")
    if kind == "fno":
        print(
            f"[FNO] layers={mc.get('n_layers')} hidden={mc.get('hidden_channels')} "
            f"modes={mc.get('n_modes')}"
        )
    elif kind == "gno":
        print(
            f"[GNO] layers={mc.get('n_layers')} hidden={mc.get('hidden_channels')} "
            f"radius={mc.get('radius')} max_neighbors={mc.get('max_neighbors')} "
            f"transform={mc.get('transform_type', 'nonlinear')} "
            f"global_context={mc.get('global_context', True)}"
        )
        if getattr(model, "fast_segment_reduce_installed", False):
            print("[GNO] CSR reduction: torch.segment_reduce")
    else:
        print(
            f"[GINO] latent={mc.get('latent_resolution')} fno_layers={mc.get('fno_n_layers')} "
            f"hidden={mc.get('fno_hidden_channels')} modes={mc.get('fno_n_modes')} "
            f"in_transform={mc.get('in_gno_transform_type', 'nonlinear')} "
            f"out_transform={mc.get('out_gno_transform_type', 'nonlinear')}"
        )
        if getattr(model, "fast_segment_reduce_installed", False):
            print("[GINO] CSR reduction: torch.segment_reduce")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    if args.inspect_only:
        return

    scale_np = np.asarray(load_scales(scales_path), dtype=np.float32)
    scales = torch.tensor(scale_np, dtype=torch.float32, device=device)
    train_loader, val_loader, sampler = make_loaders(
        cfg, kind, rep, scale_np, seed, args.run_fraction, device
    )
    tr = cfg["training"]
    batch_size = int(tr.get("batch_size", 1))
    accumulation = max(1, int(tr.get("gradient_accumulation", 1)))
    effective_batch = batch_size * accumulation
    print(
        f"[DATA] representation={rep} train_batches={len(train_loader)} "
        f"batch={batch_size} accumulation={accumulation} effective_batch={effective_batch}"
    )
    if kind == "fno":
        print(f"[BUCKET] raster_shapes={getattr(sampler, 'shapes', [])}")
    else:
        print(f"[BUCKET] profiles={getattr(sampler, 'pids', [])}")

    lc = loss_config(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tr.get("lr", 5e-4)),
        weight_decay=float(tr.get("weight_decay", 1e-6)),
    )
    epochs = int(tr.get("epochs", 30))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(epochs, 1), eta_min=float(tr.get("min_lr", 1e-6))
    )
    clip = float(tr.get("grad_clip", 1.0))

    if args.smoke_test:
        model.train()
        batch = next(iter(train_loader))
        if args.profile_first_batch:
            print(f"[PROFILE] x={tuple(batch['x'].shape)} area={tuple(batch['area'].shape)}")
            if kind in ("gno", "gino"):
                print(
                    f"[PROFILE] coords={tuple(batch['coords'].shape)} "
                    f"profile={int(batch['profile_id'][0])}"
                )
        pred, target, area = forward_batch(model, kind, batch, device)
        loss, parts = operator_loss(pred, target, area, scales, lc)
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke-test loss is not finite")
        loss.backward()
        val, val_parts = validate(model, kind, val_loader, device, scales, lc, 1, 0)
        print("[SMOKE PASS] finite forward/backward and validation")
        print(json.dumps({"parameters": nparams, "train_parts": parts, "validation_parts": val_parts, "validation_loss": val}, indent=2))
        return

    if args.benchmark_batches is not None:
        model.train()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        n = 0
        for n, batch in enumerate(train_loader, 1):
            if args.profile_first_batch and n == 1:
                print(f"[PROFILE] x={tuple(batch['x'].shape)} area={tuple(batch['area'].shape)}")
                if kind in ("gno", "gino"):
                    print(
                        f"[PROFILE] coords={tuple(batch['coords'].shape)} "
                        f"profile={int(batch['profile_id'][0])}"
                    )
            pred, target, area = forward_batch(model, kind, batch, device)
            loss, _ = operator_loss(pred, target, area, scales, lc)
            (loss / accumulation).backward()
            if n % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if n >= args.benchmark_batches:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None
        print(
            f"[BENCHMARK] batches={n} total_s={elapsed:.3f} "
            f"mean_batch_s={elapsed/max(n,1):.4f} peak_cuda_mib={peak}"
        )
        return

    start_epoch = 1
    best = float("inf")
    history = []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if ck.get("kind") != kind:
            raise ValueError("Checkpoint kind does not match config")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sch.load_state_dict(ck["scheduler"])
        start_epoch = int(ck["epoch"]) + 1
        best = float(ck.get("best_val", float("inf")))
        out = os.path.dirname(os.path.abspath(args.resume))
        hp = os.path.join(out, "history.json")
        if os.path.isfile(hp):
            with open(hp, "r", encoding="utf-8") as f:
                history = json.load(f)
        print(f"[RESUME] next_epoch={start_epoch} best={best:.6g}")
    else:
        base_out = args.out or cfg.get("output", {}).get("out_dir", f"output/models/{label}")
        out = os.path.abspath(base_out) + f"_seed{seed}"
        if os.path.isfile(os.path.join(out, "best.pt")) and not args.overwrite_run:
            raise FileExistsError(f"Existing run: {out}. Use --overwrite-run or --resume.")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "resolved_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    wall0 = time.perf_counter()
    last_epoch = min(epochs, args.stop_after_epoch) if args.stop_after_epoch else epochs
    for epoch in range(start_epoch, last_epoch + 1):
        model.train()
        avg = DeviceAverages()
        e0 = time.perf_counter()
        data_wait = 0.0
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        iterator = iter(train_loader)
        ib = 0
        while True:
            w0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_wait += time.perf_counter() - w0
            ib += 1
            if args.profile_first_batch and epoch == start_epoch and ib == 1:
                print(f"[PROFILE] x={tuple(batch['x'].shape)} area={tuple(batch['area'].shape)}")
                if kind in ("gno", "gino"):
                    print(
                        f"[PROFILE] coords={tuple(batch['coords'].shape)} "
                        f"profile={int(batch['profile_id'][0])}"
                    )
            pred, target, area = forward_batch(model, kind, batch, device)
            loss, parts = operator_loss(
                pred, target, area, scales, lc, return_tensor_parts=True
            )
            (loss / accumulation).backward()
            if ib % accumulation == 0 or ib == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
            avg.add(parts)
            if args.progress_every and (ib == 1 or ib % args.progress_every == 0 or ib == len(train_loader)):
                elapsed = time.perf_counter() - e0
                remaining = elapsed / max(ib, 1) * max(len(train_loader) - ib, 0)
                print(
                    f"[TRAIN] epoch={epoch}/{epochs} batch={ib}/{len(train_loader)} "
                    f"elapsed={elapsed:.1f}s remaining~{remaining:.1f}s"
                )
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - e0
        sch.step()

        v0 = time.perf_counter()
        val, val_parts = validate(
            model, kind, val_loader, device, scales, lc,
            int(tr.get("validation_max_batches", 30)),
            args.validation_progress_every,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        validation_s = time.perf_counter() - v0
        train_parts = avg.result()
        rec = {
            "epoch": epoch,
            "train_loss": train_parts.get("total", float("nan")),
            "val_loss": val,
            "train_parts": train_parts,
            "val_parts": val_parts,
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_s": time.perf_counter() - e0,
            "train_s": train_s,
            "data_wait_s": data_wait,
            "validation_s": validation_s,
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "effective_batch": effective_batch,
        }
        if device.type == "cuda":
            rec["peak_cuda_mib"] = float(torch.cuda.max_memory_allocated() / 2**20)
        history.append(rec)
        print(json.dumps(rec))

        ck = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sch.state_dict(),
            "epoch": epoch,
            "best_val": min(best, val),
            "kind": kind,
            "representation": rep,
            "target_scale": scales.detach().cpu(),
            "seed": seed,
            "config": cfg,
            "parameter_count": nparams,
            "loss_version": LOSS_VERSION,
            "experiment_tier": tier,
            "experiment_label": label,
        }
        torch.save(ck, os.path.join(out, "last.pt"))
        if val < best:
            best = val
            torch.save(ck, os.path.join(out, "best.pt"))
        with open(os.path.join(out, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    invocation_wall_s = time.perf_counter() - wall0
    epoch_times = [float(r.get("epoch_s", 0.0)) for r in history if r.get("epoch_s") is not None]
    train_times = [float(r.get("train_s", 0.0)) for r in history if r.get("train_s") is not None]
    validation_times = [float(r.get("validation_s", 0.0)) for r in history if r.get("validation_s") is not None]
    peak_values = [float(r.get("peak_cuda_mib", 0.0)) for r in history if r.get("peak_cuda_mib") is not None]
    samples_per_epoch = int(tr.get("samples_per_epoch", 256))
    completed_total = max((int(r.get("epoch", 0)) for r in history), default=0)
    recorded_wall_s = float(sum(epoch_times))
    total_train_s = float(sum(train_times))
    total_validation_s = float(sum(validation_times))
    frames_seen = int(completed_total * samples_per_epoch)

    summary = {
        "kind": kind,
        "representation": rep,
        "tier": tier,
        "label": label,
        "seed": seed,
        "best_val": best,
        # Total recorded epoch time is resume-safe because history.json contains
        # the previous epochs. The per-invocation wall clock is also retained.
        "training_wall_s": recorded_wall_s,
        "training_wall_s_this_invocation": invocation_wall_s,
        "training_wall_min": recorded_wall_s / 60.0,
        "training_wall_h": recorded_wall_s / 3600.0,
        "total_train_forward_backward_s": total_train_s,
        "total_validation_s": total_validation_s,
        "mean_epoch_s": (recorded_wall_s / len(epoch_times)) if epoch_times else None,
        "mean_train_s_per_epoch": (total_train_s / len(train_times)) if train_times else None,
        "mean_validation_s_per_epoch": (total_validation_s / len(validation_times)) if validation_times else None,
        "frames_seen": frames_seen,
        "train_frames_per_s": (frames_seen / total_train_s) if total_train_s > 0 else None,
        "peak_cuda_mib": max(peak_values) if peak_values else None,
        "parameters": nparams,
        "epochs_requested": epochs,
        "epochs_completed": completed_total,
        "samples_per_epoch": samples_per_epoch,
        "effective_batch": effective_batch,
        "loss_version": LOSS_VERSION,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    with open(os.path.join(out, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if summary["training_wall_s"] is not None:
        print(
            f"[TRAINING-TIME] total={summary['training_wall_min']:.2f} min "
            f"({summary['training_wall_h']:.3f} h) mean_epoch={summary['mean_epoch_s']:.2f}s "
            f"throughput={summary['train_frames_per_s']:.2f} frames/s"
        )
    try:
        from ml.training_plot import plot_history
        print("[PLOT]", plot_history(os.path.join(out, "history.json")))
    except Exception as exc:
        print("[PLOT] skipped:", exc)


if __name__ == "__main__":
    main()

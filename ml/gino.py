from __future__ import annotations

"""Reproducible wrapper around NeuralOperator's official 2-D GINO.

The wrapper keeps the official GINO architecture (input GNO -> latent FNO ->
output GNO) but replaces only the radius-neighbour search used by the two GNO
stages with a deterministic radius + nearest-K search.  This gives the GNO and
GINO workflows the same explicit local geometric sampling budget without
changing GINO's native latent-grid/Fourier mechanism.

Call signature::

    model(x=x, coords=coords, area=area, profile_id=profile_id) -> [B, N, C]

``area`` is accepted for trainer/evaluator symmetry; the official GINO operator
does not use cell areas internally. Physical cell areas remain used by the
common area-weighted loss and evaluation metrics.
"""

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from .pure_gno import install_fast_segment_reduce


class CappedRadiusNeighborSearch(nn.Module):
    """Deterministic cKDTree radius search capped at the K nearest neighbours.

    The returned dictionary follows NeuralOperator's CSR neighbour-search
    contract: ``neighbors_index`` stores source-point indices and
    ``neighbors_row_splits`` stores cumulative counts for every query point.

    Graphs are cached in RAM and on disk. ``set_active_key`` lets GINOWrapper
    identify a fixed profile geometry without hashing/copying all coordinates
    on every training step.
    """

    def __init__(
        self,
        max_neighbors: int,
        cache_dir: str,
        tag: str,
        return_norm: bool = False,
    ):
        super().__init__()
        self.max_neighbors = max(1, int(max_neighbors))
        self.cache_dir = Path(cache_dir)
        self.tag = str(tag)
        self.return_norm = bool(return_norm)
        self.active_key = "unknown"
        self._ram_cache: Dict[Tuple[str, float, str], Dict[str, torch.Tensor]] = {}
        self._geom_hash: Dict[Tuple[str, int, int], str] = {}

    def set_active_key(self, key: str) -> None:
        self.active_key = str(key)

    @staticmethod
    def _hash_pair(data: torch.Tensor, queries: torch.Tensor) -> str:
        h = sha1()
        for t in (data, queries):
            a = t.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
            h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
            h.update(a.tobytes())
        return h.hexdigest()[:16]

    def _disk_path(self, radius: float, geom_hash: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        r = f"{float(radius):.6f}".replace(".", "p")
        safe = self.active_key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"gino_{self.tag}_{safe}_{geom_hash}_r{r}_k{self.max_neighbors}.npz"

    def _build_numpy(self, data_np: np.ndarray, query_np: np.ndarray, radius: float):
        from scipy.spatial import cKDTree

        n_data = int(data_np.shape[0])
        n_query = int(query_np.shape[0])
        tree = cKDTree(np.asarray(data_np, dtype=np.float64))
        dist, idx = tree.query(
            np.asarray(query_np, dtype=np.float64),
            k=self.max_neighbors,
            distance_upper_bound=float(radius),
            workers=-1,
        )
        if self.max_neighbors == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        valid = np.isfinite(dist) & (idx < n_data)
        lengths = valid.sum(axis=1, dtype=np.int64)
        neighbors_index = idx[valid].astype(np.int64, copy=False)
        row_splits = np.empty(n_query + 1, dtype=np.int64)
        row_splits[0] = 0
        np.cumsum(lengths, out=row_splits[1:])
        neighbors_norm = dist[valid].astype(np.float32, copy=False) if self.return_norm else None
        return neighbors_index, row_splits, neighbors_norm

    def forward(self, data: torch.Tensor, queries: torch.Tensor, radius: float):
        if data.ndim != 2 or queries.ndim != 2:
            raise ValueError(
                f"CappedRadiusNeighborSearch expects [N,D] tensors, got data={tuple(data.shape)}, "
                f"queries={tuple(queries.shape)}"
            )
        key0 = (self.active_key, int(data.shape[0]), int(queries.shape[0]))
        geom_hash = self._geom_hash.get(key0)
        if geom_hash is None:
            geom_hash = self._hash_pair(data, queries)
            self._geom_hash[key0] = geom_hash
        devkey = str(data.device)
        key = (geom_hash, float(radius), devkey)
        got = self._ram_cache.get(key)
        if got is not None:
            return got

        path = self._disk_path(radius, geom_hash)
        if path.is_file():
            with np.load(path) as d:
                ni = d["neighbors_index"].astype(np.int64, copy=False)
                rs = d["neighbors_row_splits"].astype(np.int64, copy=False)
                nnorm = d["neighbors_norm"].astype(np.float32, copy=False) if "neighbors_norm" in d else None
            source = "disk"
        else:
            data_np = data.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
            query_np = queries.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
            ni, rs, nnorm = self._build_numpy(data_np, query_np, radius)
            if nnorm is None:
                np.savez_compressed(path, neighbors_index=ni, neighbors_row_splits=rs)
            else:
                np.savez_compressed(
                    path,
                    neighbors_index=ni,
                    neighbors_row_splits=rs,
                    neighbors_norm=nnorm,
                )
            source = "built"

        graph = {
            "neighbors_index": torch.from_numpy(ni).to(data.device, non_blocking=True),
            "neighbors_row_splits": torch.from_numpy(rs).to(data.device, non_blocking=True),
        }
        if nnorm is not None:
            graph["neighbors_norm"] = torch.from_numpy(nnorm).to(data.device, non_blocking=True)
        self._ram_cache[key] = graph
        mean_degree = float(len(ni) / max(len(rs) - 1, 1))
        print(
            f"[GINO-GRAPH] stage={self.tag} profile={self.active_key} "
            f"data={data.shape[0]} queries={queries.shape[0]} edges={len(ni)} "
            f"mean_degree={mean_degree:.1f} k_cap={self.max_neighbors} source={source}"
        )
        return graph


@dataclass
class GINOConfig:
    latent_resolution: Tuple[int, int] = (48, 48)
    fno_n_modes: Tuple[int, int] = (12, 12)
    fno_hidden_channels: int = 18
    fno_n_layers: int = 4
    fno_lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    in_gno_radius: float = 0.055
    out_gno_radius: float = 0.055
    in_gno_max_neighbors: int = 16
    out_gno_max_neighbors: int = 16
    gno_graph_cache_dir: str = "output/gino_graph_cache_k16"
    in_gno_transform_type: str = "nonlinear"
    out_gno_transform_type: str = "nonlinear"
    in_gno_channel_mlp_hidden_layers: Tuple[int, ...] = (64, 64)
    out_gno_channel_mlp_hidden_layers: Tuple[int, ...] = (64, 64)
    gno_pos_embed_type: str | None = None
    gno_embed_channels: int = 16
    gno_use_torch_scatter: bool = False
    gno_use_open3d: bool = False
    fast_segment_reduce: bool = True


_ALLOWED_TRANSFORMS = {
    "linear_kernelonly",
    "linear",
    "nonlinear_kernelonly",
    "nonlinear",
}


class GINOWrapper(nn.Module):
    """Official NeuralOperator GINO with shared latent grid and capped GNO graphs."""

    def __init__(self, in_channels: int, out_channels: int, cfg: GINOConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.in_gno_transform_type not in _ALLOWED_TRANSFORMS:
            raise ValueError(f"Unsupported input GNO transform: {cfg.in_gno_transform_type!r}")
        if cfg.out_gno_transform_type not in _ALLOWED_TRANSFORMS:
            raise ValueError(f"Unsupported output GNO transform: {cfg.out_gno_transform_type!r}")

        if cfg.fast_segment_reduce:
            self.fast_segment_reduce_installed = install_fast_segment_reduce()
        else:
            self.fast_segment_reduce_installed = False

        try:
            from neuralop.models import GINO
        except Exception as exc:
            raise RuntimeError(
                "NeuralOperator GINO is required. Install requirements-ml.txt "
                "in the active environment."
            ) from exc

        self.model = GINO(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            gno_coord_dim=2,
            fno_n_modes=tuple(int(v) for v in cfg.fno_n_modes),
            fno_hidden_channels=int(cfg.fno_hidden_channels),
            fno_n_layers=int(cfg.fno_n_layers),
            fno_lifting_channel_ratio=int(cfg.fno_lifting_channel_ratio),
            projection_channel_ratio=int(cfg.projection_channel_ratio),
            in_gno_radius=float(cfg.in_gno_radius),
            out_gno_radius=float(cfg.out_gno_radius),
            in_gno_transform_type=str(cfg.in_gno_transform_type),
            out_gno_transform_type=str(cfg.out_gno_transform_type),
            in_gno_pos_embed_type=cfg.gno_pos_embed_type,
            out_gno_pos_embed_type=cfg.gno_pos_embed_type,
            gno_embed_channels=int(cfg.gno_embed_channels),
            in_gno_channel_mlp_hidden_layers=[int(v) for v in cfg.in_gno_channel_mlp_hidden_layers],
            out_gno_channel_mlp_hidden_layers=[int(v) for v in cfg.out_gno_channel_mlp_hidden_layers],
            gno_use_torch_scatter=bool(cfg.gno_use_torch_scatter),
            gno_use_open3d=bool(cfg.gno_use_open3d),
        )

        # Preserve the official GINO/GNOBlock forward implementation and swap
        # only its neighbour-search module. Both stages therefore retain the
        # configured nonlinear IntegralTransform and the official latent FNO.
        self.model.gno_in.neighbor_search = CappedRadiusNeighborSearch(
            max_neighbors=cfg.in_gno_max_neighbors,
            cache_dir=cfg.gno_graph_cache_dir,
            tag="input",
            return_norm=False,
        )
        self.model.gno_out.neighbor_search = CappedRadiusNeighborSearch(
            max_neighbors=cfg.out_gno_max_neighbors,
            cache_dir=cfg.gno_graph_cache_dir,
            tag="output",
            return_norm=False,
        )

        nx, ny = (int(v) for v in cfg.latent_resolution)
        if nx < 2 or ny < 2:
            raise ValueError("latent_resolution must contain values >= 2")
        gx = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
        gy = torch.linspace(0.0, 1.0, ny, dtype=torch.float32)
        xx, yy = torch.meshgrid(gx, gy, indexing="ij")
        latent = torch.stack((xx, yy), dim=-1).unsqueeze(0)
        self.register_buffer("latent_queries", latent, persistent=False)

    @staticmethod
    def _shared_geometry(coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim == 2:
            return coords.unsqueeze(0)
        if coords.ndim != 3:
            raise ValueError(f"coords must be [N,2] or [B,N,2], got {tuple(coords.shape)}")
        return coords[:1]

    @staticmethod
    def _profile_key(profile_id) -> str:
        if profile_id is None:
            return "unknown"
        if torch.is_tensor(profile_id):
            if profile_id.numel() == 0:
                return "unknown"
            return str(int(profile_id.reshape(-1)[0].detach().cpu()))
        if isinstance(profile_id, (list, tuple)):
            return str(int(profile_id[0])) if profile_id else "unknown"
        return str(int(profile_id))

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        area: torch.Tensor | None = None,
        profile_id=None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"GINO x must be [B,N,C], got {tuple(x.shape)}")
        geom = self._shared_geometry(coords).to(device=x.device, dtype=x.dtype)
        latent = self.latent_queries.to(device=x.device, dtype=x.dtype)
        pkey = self._profile_key(profile_id)
        self.model.gno_in.neighbor_search.set_active_key(pkey)
        self.model.gno_out.neighbor_search.set_active_key(pkey)
        return self.model(
            input_geom=geom,
            latent_queries=latent,
            output_queries=geom,
            x=x,
        )


def config_from_mapping(mc: dict) -> GINOConfig:
    pos = mc.get("gno_pos_embed_type", None)
    if isinstance(pos, str) and pos.lower() in {"none", "null", ""}:
        pos = None
    return GINOConfig(
        latent_resolution=tuple(int(v) for v in mc.get("latent_resolution", [48, 48])),
        fno_n_modes=tuple(int(v) for v in mc.get("fno_n_modes", [12, 12])),
        fno_hidden_channels=int(mc.get("fno_hidden_channels", 18)),
        fno_n_layers=int(mc.get("fno_n_layers", 4)),
        fno_lifting_channel_ratio=int(mc.get("fno_lifting_channel_ratio", 2)),
        projection_channel_ratio=int(mc.get("projection_channel_ratio", 2)),
        in_gno_radius=float(mc.get("in_gno_radius", 0.055)),
        out_gno_radius=float(mc.get("out_gno_radius", 0.055)),
        in_gno_max_neighbors=int(mc.get("in_gno_max_neighbors", 16)),
        out_gno_max_neighbors=int(mc.get("out_gno_max_neighbors", 16)),
        gno_graph_cache_dir=str(mc.get("gno_graph_cache_dir", "output/gino_graph_cache_k16")),
        in_gno_transform_type=str(mc.get("in_gno_transform_type", "nonlinear")),
        out_gno_transform_type=str(mc.get("out_gno_transform_type", "nonlinear")),
        in_gno_channel_mlp_hidden_layers=tuple(
            int(v) for v in mc.get("in_gno_channel_mlp_hidden_layers", [64, 64])
        ),
        out_gno_channel_mlp_hidden_layers=tuple(
            int(v) for v in mc.get("out_gno_channel_mlp_hidden_layers", [64, 64])
        ),
        gno_pos_embed_type=pos,
        gno_embed_channels=int(mc.get("gno_embed_channels", 16)),
        gno_use_torch_scatter=bool(mc.get("gno_use_torch_scatter", False)),
        gno_use_open3d=bool(mc.get("gno_use_open3d", False)),
        fast_segment_reduce=bool(mc.get("fast_segment_reduce", True)),
    )


def build_gino(in_channels: int, out_channels: int, model_cfg: dict) -> GINOWrapper:
    return GINOWrapper(in_channels, out_channels, config_from_mapping(model_cfg))

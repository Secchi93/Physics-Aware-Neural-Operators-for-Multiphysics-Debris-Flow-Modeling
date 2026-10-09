from __future__ import annotations

"""Pure residual Graph Neural Operator for Morino point-cloud data.

This module intentionally contains NO Fourier layers and NO regular latent grid.
It stacks Graph Neural Operator (GNO) kernel-integral blocks directly on the
native point set. The same architecture therefore works on:

* Cartesian solver fields exposed as points; and
* native adaptive / irregular finite-volume cells.

The GNO kernel integral itself is provided by ``neuralop.layers.GNOBlock``.
Neighbourhoods are built once with SciPy cKDTree, cached by geometry, and fed
directly to the NeuralOperator IntegralTransform. Cell areas are used as
quadrature weights after normalization to unit total measure.
"""

from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def install_fast_segment_reduce() -> bool:
    """Replace NeuralOperator's Python CSR loop with torch.segment_reduce.

    NeuralOperator 2.0 falls back to a Python loop when torch-scatter is not
    installed. PyTorch >= 2.10 provides a native segmented reduction.  The
    replacement preserves the same ``sum``/``mean`` semantics and supports
    both unbatched [E,C] and batched [B,E,C] edge features.
    """
    if not hasattr(torch, "segment_reduce"):
        return False

    seg_mod = importlib.import_module("neuralop.layers.segment_csr")
    int_mod = importlib.import_module("neuralop.layers.integral_transform")

    def _fast_segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduction: str, use_scatter=True):
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean'")
        offsets = indptr[0] if indptr.ndim == 2 else indptr
        offsets = offsets.to(device=src.device, dtype=torch.long)
        if src.ndim == 2:
            return torch.segment_reduce(src, reduce=reduction, offsets=offsets, axis=0)
        if src.ndim == 3:
            b, e, c = src.shape
            # torch.segment_reduce is most robust here with the edge axis first.
            flat = src.permute(1, 0, 2).reshape(e, b * c)
            out = torch.segment_reduce(flat, reduce=reduction, offsets=offsets, axis=0)
            return out.reshape(out.shape[0], b, c).permute(1, 0, 2).contiguous()
        raise ValueError(f"Unsupported segment source shape: {tuple(src.shape)}")

    seg_mod.segment_csr = _fast_segment_csr
    # integral_transform imports the function directly, so patch that binding too.
    int_mod.segment_csr = _fast_segment_csr
    return True


def _import_gno_block():
    """Import GNOBlock robustly across NeuralOperator 2.x packaging layouts.

    In NeuralOperator 2.0.0 the class exists in
    ``neuralop.layers.gno_block`` but is not necessarily re-exported from
    ``neuralop.layers.__init__`` in every PyPI build.
    """
    first_exc = None
    try:
        from neuralop.layers.gno_block import GNOBlock
        return GNOBlock
    except Exception as exc:
        first_exc = exc

    try:
        from neuralop.layers import GNOBlock
        return GNOBlock
    except Exception as exc:
        raise RuntimeError(
            "Could not import NeuralOperator GNOBlock. "
            "Expected neuralop.layers.gno_block.GNOBlock in NeuralOperator 2.x. "
            "Check with: python -c \"import neuralop; print(neuralop.__version__); "
            "from neuralop.layers.gno_block import GNOBlock; print(GNOBlock)\""
        ) from (first_exc or exc)


@dataclass
class PureGNOConfig:
    hidden_channels: int = 32
    n_layers: int = 4
    radius: float = 0.020
    kernel_mlp_hidden: Tuple[int, ...] = (64, 64)
    projection_ratio: int = 2
    ffn_expansion: int = 2
    global_context: bool = True
    pos_embedding_type: Optional[str] = None
    pos_embedding_channels: int = 8
    max_neighbors: int = 64
    graph_cache_dir: str = "output/gno_graph_cache_v1"
    fast_segment_reduce: bool = True
    include_coords_in_lifting: bool = True
    transform_type: str = "nonlinear"


class _PointFFN(nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        mid = max(channels, int(channels) * int(expansion))
        self.net = nn.Sequential(
            nn.Linear(channels, mid),
            nn.GELU(),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        return self.net(x)


class PureGNO(nn.Module):
    """Residual GNO acting directly on a fixed point geometry per batch.

    Each operator layer is

        z <- z + K_r[z] + G[z]
        z <- z + FFN(z)

    where K_r is a NeuralOperator GNO kernel integral over a radius graph and
    G is an optional global quadrature-pooled context branch.  G contains no
    Fourier transform and no latent raster; it is a cheap full-domain graph
    integral term that prevents a four-layer local GNO from having an
    artificially tiny receptive field in this direct-time debris-flow task.
    """

    def __init__(self, in_channels: int = 13, out_channels: int = 8, cfg: PureGNOConfig | None = None):
        super().__init__()
        self.cfg = cfg or PureGNOConfig()
        if self.cfg.transform_type not in {"linear_kernelonly", "linear", "nonlinear_kernelonly", "nonlinear"}:
            raise ValueError(f"Unsupported GNO transform_type: {self.cfg.transform_type!r}")
        if self.cfg.fast_segment_reduce:
            self.fast_segment_reduce_installed = install_fast_segment_reduce()
        else:
            self.fast_segment_reduce_installed = False

        GNOBlock = _import_gno_block()
        h = int(self.cfg.hidden_channels)
        lift_in = int(in_channels) + (2 if self.cfg.include_coords_in_lifting else 0)
        self.lifting = nn.Sequential(
            nn.Linear(lift_in, 2 * h),
            nn.GELU(),
            nn.Linear(2 * h, h),
        )
        self.blocks = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        self.global_proj = nn.ModuleList()

        kernel_hidden = [int(v) for v in self.cfg.kernel_mlp_hidden]
        for _ in range(int(self.cfg.n_layers)):
            self.blocks.append(
                GNOBlock(
                    in_channels=h,
                    out_channels=h,
                    coord_dim=2,
                    radius=float(self.cfg.radius),
                    transform_type=str(self.cfg.transform_type),
                    reduction="mean",
                    weighting_fn=None,
                    pos_embedding_type=self.cfg.pos_embedding_type,
                    pos_embedding_channels=int(self.cfg.pos_embedding_channels),
                    channel_mlp_layers=kernel_hidden,
                    use_torch_scatter_reduce=False,
                    use_open3d_neighbor_search=False,
                )
            )
            self.norm1.append(nn.LayerNorm(h))
            self.norm2.append(nn.LayerNorm(h))
            self.ffn.append(_PointFFN(h, int(self.cfg.ffn_expansion)))
            self.global_proj.append(nn.Linear(h, h, bias=False))

        proj = max(h, int(self.cfg.projection_ratio) * h)
        self.projection = nn.Sequential(
            nn.Linear(h, proj),
            nn.GELU(),
            nn.Linear(proj, int(out_channels)),
        )

        # GPU-resident CSR maps. Keyed by (profile, geometry hash, device).
        self._graph_cache: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        self._profile_hashes: Dict[str, str] = {}

    @staticmethod
    def _geometry_hash(coords: torch.Tensor) -> str:
        a = coords.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        return sha1(a.tobytes()).hexdigest()[:16]

    def _disk_graph_path(self, profile_key: str, geom_hash: str) -> Path:
        root = Path(self.cfg.graph_cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        r = f"{float(self.cfg.radius):.6f}".replace(".", "p")
        return root / f"gno_{profile_key}_n{geom_hash}_r{r}_k{int(self.cfg.max_neighbors)}.npz"

    def _build_graph_numpy(self, coords_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.spatial import cKDTree

        n = int(coords_np.shape[0])
        k = max(1, int(self.cfg.max_neighbors))
        tree = cKDTree(np.asarray(coords_np, dtype=np.float64))
        # k-nearest within the fixed radius. With the supplied r=0.02 and k=64
        # the Morino meshes are not truncated in normal operation, while the cap
        # guards against accidental graph explosions.
        dist, idx = tree.query(
            coords_np,
            k=k,
            distance_upper_bound=float(self.cfg.radius),
            workers=-1,
        )
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        valid = np.isfinite(dist) & (idx < n)
        lengths = valid.sum(axis=1, dtype=np.int64)
        if np.any(lengths == 0):
            raise RuntimeError("GNO graph contains a query with no neighbours; self-neighbour should always be present")
        neighbors_index = idx[valid].astype(np.int64, copy=False)
        row_splits = np.empty(n + 1, dtype=np.int64)
        row_splits[0] = 0
        np.cumsum(lengths, out=row_splits[1:])
        return neighbors_index, row_splits

    def _get_graph(self, coords: torch.Tensor, profile_id) -> Dict[str, torch.Tensor]:
        if coords.ndim != 2 or coords.shape[-1] != 2:
            raise ValueError(f"coords must be [N,2], got {tuple(coords.shape)}")
        profile_key = str(int(profile_id)) if profile_id is not None else "unknown"
        geom_hash = self._profile_hashes.get(profile_key)
        if geom_hash is None:
            geom_hash = self._geometry_hash(coords)
            self._profile_hashes[profile_key] = geom_hash
        devkey = str(coords.device)
        key = (profile_key, geom_hash, devkey)
        got = self._graph_cache.get(key)
        if got is not None:
            return got

        path = self._disk_graph_path(profile_key, geom_hash)
        if path.is_file():
            with np.load(path) as d:
                ni = d["neighbors_index"].astype(np.int64, copy=False)
                rs = d["neighbors_row_splits"].astype(np.int64, copy=False)
            source = "disk"
        else:
            c = coords.detach().to("cpu", dtype=torch.float32).numpy()
            ni, rs = self._build_graph_numpy(c)
            np.savez_compressed(path, neighbors_index=ni, neighbors_row_splits=rs)
            source = "built"

        graph = {
            "neighbors_index": torch.from_numpy(ni).to(coords.device, non_blocking=True),
            "neighbors_row_splits": torch.from_numpy(rs).to(coords.device, non_blocking=True),
        }
        self._graph_cache[key] = graph
        mean_degree = float(len(ni) / max(len(rs) - 1, 1))
        print(
            f"[GNO-GRAPH] profile={profile_key} N={coords.shape[0]} edges={len(ni)} "
            f"mean_degree={mean_degree:.1f} source={source}"
        )
        return graph

    @staticmethod
    def _node_area(area: torch.Tensor) -> torch.Tensor:
        # Geometry is shared within a profile-bucketed batch.
        return (area[0] if area.ndim == 2 else area).to(dtype=torch.float32)

    @staticmethod
    def _global_quadrature_weights(area: torch.Tensor) -> torch.Tensor:
        a = PureGNO._node_area(area)
        return a / a.sum().clamp_min(1e-12)

    @staticmethod
    def _local_edge_quadrature_weights(area: torch.Tensor, graph: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Area-weighted LOCAL mean.  IntegralTransform's actual implementation
        # multiplies supplied weights edge-by-edge, so construct one weight for
        # every representative edge and normalize each query neighbourhood to 1.
        cached = graph.get("local_quadrature_weights")
        if cached is not None:
            return cached
        a = PureGNO._node_area(area).to(graph["neighbors_index"].device)
        ni = graph["neighbors_index"]
        rs = graph["neighbors_row_splits"]
        ew = a[ni]
        local_sum = torch.segment_reduce(ew, reduce="sum", offsets=rs, axis=0).clamp_min(1e-12)
        lengths = rs[1:] - rs[:-1]
        denom = torch.repeat_interleave(local_sum, lengths)
        ew = (ew / denom).contiguous()
        graph["local_quadrature_weights"] = ew
        return ew

    @staticmethod
    def _same_geometry(coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim == 2:
            return coords
        if coords.ndim != 3:
            raise ValueError(f"coords must be [N,2] or [B,N,2], got {tuple(coords.shape)}")
        # Profile-bucketed training guarantees this. Avoid a full allclose every
        # step; using the first geometry follows NeuralOperator's shared-geometry batching contract.
        return coords[0]

    def _integral(self, block, z, coords, graph, edge_weights):
        if block.pos_embedding is not None:
            y = block.pos_embedding(coords)
            x = y
        else:
            y = coords
            x = coords
        return block.integral_transform(y=y, x=x, neighbors=graph, f_y=z, weights=edge_weights)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, area: torch.Tensor, profile_id=None) -> torch.Tensor:
        """Return [B,N,8] normalized conservative targets."""
        if x.ndim != 3:
            raise ValueError(f"x must be [B,N,C], got {tuple(x.shape)}")
        c = self._same_geometry(coords).to(device=x.device, dtype=x.dtype)
        if self.cfg.include_coords_in_lifting:
            cb = c.unsqueeze(0).expand(x.shape[0], -1, -1)
            z = self.lifting(torch.cat((x, cb), dim=-1))
        else:
            z = self.lifting(x)

        graph = self._get_graph(c, profile_id)
        global_weights = self._global_quadrature_weights(area).to(device=x.device, dtype=x.dtype)
        edge_weights = self._local_edge_quadrature_weights(area, graph).to(device=x.device, dtype=x.dtype)

        for i, block in enumerate(self.blocks):
            zn = self.norm1[i](z)
            local = self._integral(block, zn, c, graph, edge_weights)
            if self.cfg.global_context:
                pooled = torch.sum(zn * global_weights.view(1, -1, 1), dim=1, keepdim=True)
                global_term = self.global_proj[i](pooled).expand(-1, z.shape[1], -1)
                z = z + local + global_term
            else:
                z = z + local
            z = z + self.ffn[i](self.norm2[i](z))

        return self.projection(z)


def config_from_mapping(mc: dict) -> PureGNOConfig:
    return PureGNOConfig(
        hidden_channels=int(mc.get("hidden_channels", 32)),
        n_layers=int(mc.get("n_layers", 4)),
        radius=float(mc.get("radius", 0.020)),
        kernel_mlp_hidden=tuple(int(v) for v in mc.get("kernel_mlp_hidden", [64, 64])),
        projection_ratio=int(mc.get("projection_ratio", 2)),
        ffn_expansion=int(mc.get("ffn_expansion", 2)),
        global_context=bool(mc.get("global_context", True)),
        pos_embedding_type=mc.get("pos_embedding_type", None),
        pos_embedding_channels=int(mc.get("pos_embedding_channels", 8)),
        max_neighbors=int(mc.get("max_neighbors", 64)),
        graph_cache_dir=str(mc.get("graph_cache_dir", "output/gno_graph_cache_v1")),
        fast_segment_reduce=bool(mc.get("fast_segment_reduce", True)),
        include_coords_in_lifting=bool(mc.get("include_coords_in_lifting", True)),
        transform_type=str(mc.get("transform_type", "nonlinear")),
    )


def build_pure_gno(in_channels: int, out_channels: int, mc: dict) -> PureGNO:
    return PureGNO(in_channels, out_channels, config_from_mapping(mc))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

from __future__ import annotations

"""Model builders for the controlled six-workflow neural-operator benchmark.

Active operator families:

* FNO  — Fourier Neural Operator on regular rasters;
* GNO  — pure point/cell Graph Neural Operator;
* GINO — Geometry-Informed Neural Operator (GNO -> latent FNO -> GNO).
"""

from dataclasses import dataclass
from typing import Tuple

from .pure_gno import build_pure_gno
from .gino import build_gino


@dataclass
class FNOConfig:
    n_modes: Tuple[int, int] = (8, 8)
    hidden_channels: int = 18
    n_layers: int = 2
    lifting_channel_ratio: int = 2
    projection_channel_ratio: int = 2
    domain_padding: float = 0.05


def build_fno(in_channels: int = 13, out_channels: int = 8, cfg: FNOConfig | None = None):
    try:
        from neuralop.models import FNO
    except Exception as exc:
        raise RuntimeError(
            "NeuralOperator is required. Install requirements-ml.txt in the active environment."
        ) from exc
    c = cfg or FNOConfig()
    return FNO(
        n_modes=tuple(c.n_modes),
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        hidden_channels=int(c.hidden_channels),
        n_layers=int(c.n_layers),
        lifting_channel_ratio=int(c.lifting_channel_ratio),
        projection_channel_ratio=int(c.projection_channel_ratio),
        domain_padding=float(c.domain_padding),
        positional_embedding="grid",
    )


def build_model(kind: str, in_channels: int, out_channels: int, model_cfg: dict):
    kind = str(kind).lower()
    if kind == "fno":
        return build_fno(
            in_channels,
            out_channels,
            FNOConfig(
                n_modes=tuple(int(v) for v in model_cfg.get("n_modes", [8, 8])),
                hidden_channels=int(model_cfg.get("hidden_channels", 18)),
                n_layers=int(model_cfg.get("n_layers", 2)),
                lifting_channel_ratio=int(model_cfg.get("lifting_channel_ratio", 2)),
                projection_channel_ratio=int(model_cfg.get("projection_channel_ratio", 2)),
                domain_padding=float(model_cfg.get("domain_padding", 0.05)),
            ),
        )
    if kind == "gno":
        return build_pure_gno(in_channels, out_channels, model_cfg)
    if kind == "gino":
        return build_gino(in_channels, out_channels, model_cfg)
    raise ValueError(f"Unsupported model kind: {kind!r}. Use 'fno', 'gno' or 'gino'.")


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

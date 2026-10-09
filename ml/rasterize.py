from __future__ import annotations

"""Conservative rectangle-to-raster remapping for the adaptive-mesh baseline."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy import sparse


@dataclass
class RasterMap:
    W: sparse.csr_matrix  # overlap / target_area; forward target averages use coverage normalization
    x: np.ndarray
    y: np.ndarray
    shape: Tuple[int,int]
    target_area: float
    coverage: np.ndarray
    source_area: np.ndarray


def build_raster_map(mesh_npz: str, target_dx: float = 10.0) -> RasterMap:
    with np.load(mesh_npz) as m:
        x0=m["x0"].astype(float); x1=m["x1"].astype(float); y0=m["y0"].astype(float); y1=m["y1"].astype(float)
        source_area=(m["area"].astype(float) if "area" in m.files else (x1-x0)*(y1-y0))
    xmin=float(np.min(x0)); xmax=float(np.max(x1)); ymin=float(np.min(y0)); ymax=float(np.max(y1)); d=float(target_dx)
    nx=max(1,int(round((xmax-xmin)/d))); ny=max(1,int(round((ymax-ymin)/d)))
    # If bounds are not exact multiples, expand by at most one cell.
    if xmin+nx*d < xmax-1e-9: nx+=1
    if ymin+ny*d < ymax-1e-9: ny+=1
    x=xmin+(np.arange(nx)+0.5)*d; y=ymin+(np.arange(ny)+0.5)*d
    rows=[]; cols=[]; vals=[]; area_t=d*d
    for s in range(x0.size):
        i0=max(0,int(np.floor((x0[s]-xmin)/d))); i1=min(nx-1,int(np.floor((x1[s]-xmin-1e-12)/d)))
        j0=max(0,int(np.floor((y0[s]-ymin)/d))); j1=min(ny-1,int(np.floor((y1[s]-ymin-1e-12)/d)))
        for j in range(j0,j1+1):
            ty0=ymin+j*d; ty1=ty0+d; oy=max(0.0,min(y1[s],ty1)-max(y0[s],ty0))
            if oy<=0: continue
            for i in range(i0,i1+1):
                tx0=xmin+i*d; tx1=tx0+d; ox=max(0.0,min(x1[s],tx1)-max(x0[s],tx0))
                if ox<=0: continue
                rows.append(j*nx+i); cols.append(s); vals.append((ox*oy)/area_t)
    W=sparse.csr_matrix((vals,(rows,cols)),shape=(ny*nx,x0.size),dtype=np.float64)
    coverage=np.asarray(W.sum(axis=1)).ravel()
    return RasterMap(
        W=W, x=x.astype(np.float32), y=y.astype(np.float32), shape=(ny,nx),
        target_area=area_t, coverage=coverage, source_area=source_area.astype(np.float64),
    )


def apply_raster_map(rm: RasterMap, values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Map source cell averages. values shape (..., Nsrc) -> (..., Ny, Nx)."""
    v=np.asarray(values); lead=v.shape[:-1]; flat=v.reshape((-1,v.shape[-1]))
    out=np.empty((flat.shape[0],rm.W.shape[0]),dtype=np.result_type(v.dtype,np.float32))
    for k in range(flat.shape[0]): out[k]=rm.W.dot(flat[k])
    cov=rm.coverage
    # W already divides by target area. Cells not fully covered are normalized by
    # their covered fraction so scalar fields such as elevation remain averages.
    good=cov>1e-12
    out[:,good]/=cov[good][None,:]
    out[:,~good]=fill_value
    return out.reshape((*lead,*rm.shape))


def apply_raster_to_source_map(rm: RasterMap, values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Area-overlap back projection from a raster to the original adaptive cells.

    Parameters
    ----------
    rm:
        The same :class:`RasterMap` used for adaptive -> raster preprocessing.
    values:
        Raster cell averages with shape ``(..., Ny, Nx)``.

    Returns
    -------
    np.ndarray
        Piecewise-raster field averaged over each original adaptive cell, with
        shape ``(..., Nsrc)``. The operation uses the exact rectangle-overlap
        areas from ``rm.W``. It is not an interpolation and it does not attempt
        to reconstruct information lost during rasterization.

    Notes
    -----
    ``rm.W[t,s] = A_ts / A_t``. Therefore ``rm.W.T * A_t`` contains the
    physical overlap areas ``A_ts``. For source cell ``s`` we compute

        q_s = sum_t A_ts q_t / A_s.

    This preserves constant fields and, for a fully covered mesh, preserves
    the domain integral of the raster prediction when evaluated with native
    adaptive-cell areas.
    """
    v=np.asarray(values)
    if v.ndim < 2 or tuple(v.shape[-2:]) != tuple(rm.shape):
        raise ValueError(f"expected raster trailing shape {rm.shape}, got {v.shape}")
    lead=v.shape[:-2]
    flat=v.reshape((-1, rm.W.shape[0]))
    out=np.empty((flat.shape[0], rm.W.shape[1]), dtype=np.result_type(v.dtype, np.float32))
    # Sparse transpose times target area gives A_ts with shape (Nsrc, Ntgt).
    overlap_t = rm.W.transpose().tocsr() * float(rm.target_area)
    area=np.asarray(rm.source_area, dtype=np.float64)
    good=area > 1.0e-12
    for k in range(flat.shape[0]):
        integ=np.asarray(overlap_t.dot(flat[k])).ravel()
        out[k,good]=integ[good]/area[good]
        out[k,~good]=fill_value
    return out.reshape((*lead, rm.W.shape[1]))

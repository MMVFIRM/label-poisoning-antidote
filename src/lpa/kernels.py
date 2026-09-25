from __future__ import annotations

import numpy as np


def squared_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances with float64 accumulation."""
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.ndim != 2 or bb.ndim != 2:
        raise ValueError("a and b must be 2-D arrays.")
    if aa.shape[1] != bb.shape[1]:
        raise ValueError("a and b must have the same feature dimension.")
    dist = (
        np.sum(aa * aa, axis=1)[:, None]
        + np.sum(bb * bb, axis=1)[None, :]
        - 2.0 * (aa @ bb.T)
    )
    return np.maximum(dist, 0.0)


def rbf(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return np.exp(-float(gamma) * squared_euclidean(a, b))

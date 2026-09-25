from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._checks import as_finite_2d


@dataclass(frozen=True)
class RidgeSufficientStatistics:
    a: np.ndarray
    b: np.ndarray
    n_examples: int


def client_sufficient_statistics(
    phi: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None = None,
) -> RidgeSufficientStatistics:
    """A_i = Phi^T diag(w) Phi and B_i = Phi^T diag(w) Q for one client.

    `weights` are per-row ridge weights (for example `trusted_row_weights`);
    omit them for the unweighted v1.0 student.
    """
    x = as_finite_2d(phi, "phi")
    q = as_finite_2d(targets, "targets")
    if len(x) != len(q):
        raise ValueError("phi and targets must be equal-length 2-D arrays.")
    if weights is None:
        return RidgeSufficientStatistics(a=x.T @ x, b=x.T @ q, n_examples=len(x))
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(x),) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be one positive finite value per row.")
    xw = x * w[:, None]
    return RidgeSufficientStatistics(a=xw.T @ x, b=xw.T @ q, n_examples=len(x))


def aggregate_sufficient_statistics(
    statistics: list[RidgeSufficientStatistics],
    ridge: float,
) -> np.ndarray:
    if not statistics:
        raise ValueError("At least one client statistic is required.")
    if ridge <= 0:
        raise ValueError("ridge must be positive.")
    d = statistics[0].a.shape[0]
    c = statistics[0].b.shape[1]
    normal = ridge * np.eye(d, dtype=np.float64)
    rhs = np.zeros((d, c), dtype=np.float64)
    for stat in statistics:
        if stat.a.shape != (d, d) or stat.b.shape != (d, c):
            raise ValueError("All client statistics must share dimensions.")
        if not (np.isfinite(stat.a).all() and np.isfinite(stat.b).all()):
            raise ValueError("A client statistic contains NaN or infinite values.")
        if not np.allclose(stat.a, stat.a.T, rtol=1e-10, atol=1e-8):
            raise ValueError("A client Gram matrix is not symmetric.")
        normal += stat.a
        rhs += stat.b
    cf = cho_factor(normal, check_finite=False)
    return cho_solve(cf, rhs, check_finite=False)


def centralized_ridge(
    phi: np.ndarray,
    targets: np.ndarray,
    ridge: float,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    return aggregate_sufficient_statistics(
        [client_sufficient_statistics(phi, targets, weights)],
        ridge=ridge,
    )


def federated_equivalence_audit(
    phi: np.ndarray,
    targets: np.ndarray,
    partitions: list[np.ndarray],
    ridge: float,
    weights: np.ndarray | None = None,
) -> float:
    """Return max |W_federated - W_centralized|."""
    x = np.asarray(phi)
    q = np.asarray(targets)
    central = centralized_ridge(x, q, ridge, weights)
    stats = [
        client_sufficient_statistics(x[idx], q[idx], None if weights is None else np.asarray(weights)[idx])
        for idx in partitions
    ]
    federated = aggregate_sufficient_statistics(stats, ridge)
    return float(np.max(np.abs(federated - central)))


def packed_symmetric_payload_bytes(
    student_dim: int,
    n_classes: int,
    dtype_bytes: int = 4,
) -> int:
    if student_dim <= 0 or n_classes <= 0 or dtype_bytes <= 0:
        raise ValueError("All dimensions must be positive.")
    symmetric_a = student_dim * (student_dim + 1) // 2
    b = student_dim * n_classes
    return (symmetric_a + b) * dtype_bytes

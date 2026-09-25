from __future__ import annotations

import numpy as np


def as_finite_2d(values: np.ndarray, name: str, dtype: type = np.float64) -> np.ndarray:
    """Return `values` as a 2-D floating array, rejecting NaN and infinity."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def as_int_1d(values: np.ndarray, name: str) -> np.ndarray:
    """Return `values` as a 1-D int64 array without silently truncating.

    Floating inputs are accepted only when every value is integral; booleans,
    strings, and other non-numeric inputs are rejected.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if arr.dtype == object:
        try:
            arr = arr.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain integers.") from None
    if arr.dtype.kind == "b" or arr.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain integers.")
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all() or not np.array_equal(arr, np.floor(arr)):
            raise ValueError(f"{name} must contain integers; got non-integral values.")
    return arr.astype(np.int64)

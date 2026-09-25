from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FeatureConfig


@dataclass(frozen=True)
class FeatureViews:
    view_a: np.ndarray
    view_b: np.ndarray
    joint: np.ndarray


def row_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D feature matrix.")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def _as_cifar_nchw(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Image data contains NaN or infinite values.")
    if arr.ndim == 2 and arr.shape[1] == 3072:
        return arr.reshape(-1, 3, 32, 32)
    if arr.ndim == 4 and arr.shape[1:] == (3, 32, 32):
        return arr
    if arr.ndim == 4 and arr.shape[1:] == (32, 32, 3):
        return np.transpose(arr, (0, 3, 1, 2))
    raise ValueError(
        "Validated image adapter expects CIFAR-like data as (N,3072), "
        "(N,3,32,32), or (N,32,32,3)."
    )


def hog576(x: np.ndarray, chunk_size: int = 2000, eps: float = 1e-12) -> np.ndarray:
    """Fixed 576-D HOG-style descriptor used by the validated release.

    This intentionally matches the Gate-26 through Gate-32 benchmark harness.
    It is not a call to an external HOG implementation.
    """
    im_all = _as_cifar_nchw(x)
    outputs: list[np.ndarray] = []
    for start in range(0, len(im_all), chunk_size):
        im = im_all[start : start + chunk_size]
        gray = 0.299 * im[:, 0] + 0.587 * im[:, 1] + 0.114 * im[:, 2]
        gx = gray[:, 1:-1, 2:] - gray[:, 1:-1, :-2]
        gy = gray[:, 2:, 1:-1] - gray[:, :-2, 1:-1]
        mag = np.sqrt(gx * gx + gy * gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        bins = np.minimum((9.0 * ang / np.pi).astype(np.int64), 8)

        n = len(im)
        h = np.zeros((n, 8, 8, 9), dtype=np.float32)
        rows = np.arange(30) // 4
        cols = np.arange(30) // 4
        sample = np.arange(n)
        for i in range(30):
            for j in range(30):
                np.add.at(
                    h[:, rows[i], cols[j], :],
                    (sample, bins[:, i, j]),
                    mag[:, i, j],
                )
        h = np.sqrt(h.reshape(n, -1))
        outputs.append(row_normalize(h, eps=eps).astype(np.float32))
    return np.concatenate(outputs, axis=0)


def coarse48(x: np.ndarray) -> np.ndarray:
    """48-D 4x4 spatial RGB block means used by the validated release."""
    im = _as_cifar_nchw(x)
    return im.reshape(-1, 3, 4, 8, 4, 8).mean(axis=(3, 5)).reshape(len(im), -1)


class CIFARFeatureExtractor:
    """Label-independent feature extractor for the validated CIFAR configuration."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.color_mean_: np.ndarray | None = None
        self.color_scale_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.color_mean_ is not None and self.color_scale_ is not None

    def fit(self, x: np.ndarray) -> "CIFARFeatureExtractor":
        c0 = coarse48(x).astype(np.float64)
        self.color_mean_ = c0.mean(axis=0)
        self.color_scale_ = c0.std(axis=0) + self.config.color_floor
        return self

    def transform(self, x: np.ndarray, chunk_size: int = 2000) -> FeatureViews:
        if self.color_mean_ is None or self.color_scale_ is None:
            raise RuntimeError("Feature extractor must be fitted first.")
        h = hog576(x, chunk_size=chunk_size, eps=self.config.normalization_eps)
        c0 = coarse48(x).astype(np.float64)
        c = row_normalize(
            (c0 - self.color_mean_) / self.color_scale_,
            eps=self.config.normalization_eps,
        ).astype(np.float32)
        z = np.concatenate([h, c], axis=1) / np.sqrt(2.0)
        return FeatureViews(view_a=h, view_b=c, joint=z.astype(np.float32))

    def fit_transform(self, x: np.ndarray, chunk_size: int = 2000) -> FeatureViews:
        return self.fit(x).transform(x, chunk_size=chunk_size)

    def state_dict(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Feature extractor is not fitted.")
        return {
            "feature_color_mean": np.asarray(self.color_mean_, dtype=np.float64),
            "feature_color_scale": np.asarray(self.color_scale_, dtype=np.float64),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.color_mean_ = np.asarray(state["feature_color_mean"], dtype=np.float64)
        self.color_scale_ = np.asarray(state["feature_color_scale"], dtype=np.float64)

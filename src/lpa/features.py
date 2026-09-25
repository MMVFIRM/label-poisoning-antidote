from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import FeatureConfig, KMeansFeatureConfig


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


def _as_cifar_nhwc_255(x: np.ndarray) -> np.ndarray:
    """CIFAR-like images as float32 NHWC on the 0-255 scale.

    Integer inputs are taken as 0-255 pixel values; floating inputs as [0, 1].
    """
    raw = np.asarray(x)
    im = _as_cifar_nchw(raw)
    if raw.dtype.kind not in "iu":
        im = im * np.float32(255.0)
    return np.ascontiguousarray(np.transpose(im, (0, 2, 3, 1)), dtype=np.float32)


class KMeansPatchFeatureExtractor:
    """Coates & Ng (2011) single-layer k-means features, as validated in Gate 34.

    6x6 patches with per-patch normalization, ZCA whitening, k-means centroids,
    triangle encoding, and 2x2 quadrant sum-pooling, followed by a square root,
    standardization with training-pool statistics, and row normalization.
    Every fitted quantity is computed from images only; no label is read.
    """

    def __init__(self, config: KMeansFeatureConfig | None = None) -> None:
        self.config = config or KMeansFeatureConfig()
        self.patch_mean_: np.ndarray | None = None
        self.whitening_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return all(
            item is not None
            for item in (
                self.patch_mean_,
                self.whitening_,
                self.centroids_,
                self.feature_mean_,
                self.feature_scale_,
            )
        )

    @property
    def n_features(self) -> int:
        return 4 * self.config.centroids

    def _patch_norm(self, p: np.ndarray) -> np.ndarray:
        p = p - p.mean(-1, keepdims=True)
        return p / np.sqrt(p.var(-1, keepdims=True) + self.config.patch_norm_eps)

    def _fit_dictionary(self, im: np.ndarray) -> None:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        n, size = cfg.patch_samples, cfg.patch_size
        span = 33 - size
        ni = rng.integers(0, len(im), n)
        r = rng.integers(0, span, n)
        c = rng.integers(0, span, n)
        s = np.stack([im[ni, r + i, c + j] for i in range(size) for j in range(size)], 1).reshape(n, -1)
        s = self._patch_norm(s).astype(np.float64)
        mu = s.mean(0)
        ev, v = np.linalg.eigh(np.cov(s - mu, rowvar=False))
        zca = (v @ np.diag(1 / np.sqrt(ev + cfg.zca_eps)) @ v.T).astype(np.float32)
        sw = ((s - mu) @ zca).astype(np.float32)
        if cfg.centroids > n:
            raise ValueError("patch_samples must be at least the number of centroids.")
        cen = sw[rng.choice(n, cfg.centroids, replace=False)].copy()
        for _ in range(cfg.iterations):
            cc = (cen * cen).sum(1)
            assign = np.concatenate(
                [(cc[None] - 2 * sw[a : a + 50000] @ cen.T).argmin(1) for a in range(0, n, 50000)]
            )
            count = np.bincount(assign, minlength=cfg.centroids)
            new = np.zeros_like(cen)
            np.add.at(new, assign, sw)
            empty = count == 0
            new[~empty] /= count[~empty, None]
            new[empty] = sw[rng.choice(n, int(empty.sum()), replace=False)]
            cen = new
        self.patch_mean_ = mu.astype(np.float32)
        self.whitening_ = zca
        self.centroids_ = cen

    def _encode(self, im: np.ndarray) -> np.ndarray:
        assert self.patch_mean_ is not None and self.whitening_ is not None and self.centroids_ is not None
        cfg = self.config
        size, k = cfg.patch_size, len(self.centroids_)
        grid = 33 - size
        half = grid // 2
        mu, zca, cen = self.patch_mean_, self.whitening_, self.centroids_
        cc = (cen * cen).sum(1)
        out = np.empty((len(im), 4 * k), np.float32)

        def job(a: int) -> None:
            block = im[a : a + cfg.chunk_size]
            p = sliding_window_view(block, (size, size), axis=(1, 2)).transpose(0, 1, 2, 4, 5, 3)
            p = self._patch_norm(p.reshape(len(block), grid, grid, -1))
            pw = (p.reshape(-1, p.shape[-1]) - mu) @ zca
            d = np.sqrt(np.maximum((pw * pw).sum(1)[:, None] + cc[None] - 2 * pw @ cen.T, 0))
            f = np.maximum(0, d.mean(1, keepdims=True) - d).reshape(len(block), grid, grid, k)
            quads = (f[:, :half, :half], f[:, :half, half:], f[:, half:, :half], f[:, half:, half:])
            out[a : a + len(block)] = np.concatenate([q.sum((1, 2)) for q in quads], 1)

        with ThreadPoolExecutor(cfg.workers or os.cpu_count() or 1) as pool:
            list(pool.map(job, range(0, len(im), cfg.chunk_size)))
        return out

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        assert self.feature_mean_ is not None and self.feature_scale_ is not None
        a = (np.sqrt(raw.astype(np.float64)) - self.feature_mean_) / self.feature_scale_
        return row_normalize(a)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        im = _as_cifar_nhwc_255(x)
        self._fit_dictionary(im)
        raw = self._encode(im)
        root = np.sqrt(raw.astype(np.float64))
        self.feature_mean_ = root.mean(0)
        self.feature_scale_ = root.std(0) + self.config.standardize_floor
        return self._postprocess(raw)

    def fit(self, x: np.ndarray) -> "KMeansPatchFeatureExtractor":
        self.fit_transform(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("K-means feature extractor must be fitted first.")
        return self._postprocess(self._encode(_as_cifar_nhwc_255(x)))

    def state_dict(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("K-means feature extractor is not fitted.")
        return {
            "kmeans_patch_mean": np.asarray(self.patch_mean_, dtype=np.float32),
            "kmeans_whitening": np.asarray(self.whitening_, dtype=np.float32),
            "kmeans_centroids": np.asarray(self.centroids_, dtype=np.float32),
            "kmeans_feature_mean": np.asarray(self.feature_mean_, dtype=np.float64),
            "kmeans_feature_scale": np.asarray(self.feature_scale_, dtype=np.float64),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.patch_mean_ = np.asarray(state["kmeans_patch_mean"], dtype=np.float32)
        self.whitening_ = np.asarray(state["kmeans_whitening"], dtype=np.float32)
        self.centroids_ = np.asarray(state["kmeans_centroids"], dtype=np.float32)
        self.feature_mean_ = np.asarray(state["kmeans_feature_mean"], dtype=np.float64)
        self.feature_scale_ = np.asarray(state["kmeans_feature_scale"], dtype=np.float64)

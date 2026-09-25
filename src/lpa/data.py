from __future__ import annotations

from pathlib import Path

import numpy as np


def read_cifar_binary(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    raw = np.fromfile(p, dtype=np.uint8)
    if raw.size % 3073:
        raise ValueError(f"Unexpected CIFAR binary size: {p}")
    rec = raw.reshape(-1, 3073)
    x = rec[:, 1:].astype(np.float32) / 255.0
    y = rec[:, 0].astype(np.int64)
    return x, y


def load_cifar10_binary(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_dir)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(1, 6):
        x, y = read_cifar_binary(root / f"data_batch_{i}.bin")
        xs.append(x)
        ys.append(y)
    xt, yt = read_cifar_binary(root / "test_batch.bin")
    return np.concatenate(xs), np.concatenate(ys), xt, yt


def balanced_trusted_indices(
    labels: np.ndarray,
    per_class: int,
    seed: int,
    n_classes: int | None = None,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    if per_class <= 0:
        raise ValueError("per_class must be positive.")
    classes = int(n_classes if n_classes is not None else y.max() + 1)
    rng = np.random.default_rng(seed)
    ids: list[int] = []
    for c in range(classes):
        cids = np.flatnonzero(y == c)
        if len(cids) < per_class:
            raise ValueError(f"Class {c} has fewer than {per_class} examples.")
        ids.extend(rng.choice(cids, per_class, replace=False).tolist())
    return np.sort(np.asarray(ids, dtype=np.int64))

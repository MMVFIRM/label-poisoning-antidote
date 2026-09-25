from __future__ import annotations

import hashlib

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._checks import as_finite_2d
from .config import StudentConfig
from .kernels import rbf


class LandmarkRidgeStudent:
    """Ridge student over [joint features ; RBF similarities to x-only landmarks]."""

    def __init__(self, n_classes: int, config: StudentConfig | None = None) -> None:
        self.n_classes = int(n_classes)
        self.config = config or StudentConfig()
        self.landmark_indices_: np.ndarray | None = None
        self.landmarks_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.landmarks_ is not None and self.weights_ is not None

    def choose_landmarks(self, joint: np.ndarray) -> np.ndarray:
        z = np.asarray(joint)
        if len(z) < self.config.landmark_count:
            raise ValueError(
                f"Need at least {self.config.landmark_count} examples for the "
                "validated landmark count."
            )
        rng = np.random.default_rng(self.config.landmark_seed)
        return np.sort(
            rng.choice(len(z), self.config.landmark_count, replace=False).astype(np.int64)
        )

    def transform(self, joint: np.ndarray) -> np.ndarray:
        if self.landmarks_ is None:
            raise RuntimeError("Landmarks have not been initialized.")
        z = as_finite_2d(joint, "joint")
        if z.shape[1] != self.landmarks_.shape[1]:
            raise ValueError(
                f"Student expects {self.landmarks_.shape[1]} joint features; got {z.shape[1]}."
            )
        k = rbf(z, self.landmarks_, self.config.landmark_gamma)
        return np.concatenate([z, k], axis=1) / np.sqrt(2.0)

    def fit(
        self,
        joint: np.ndarray,
        targets: np.ndarray,
        landmark_indices: np.ndarray | None = None,
    ) -> "LandmarkRidgeStudent":
        z = as_finite_2d(joint, "joint")
        q = as_finite_2d(targets, "targets")
        if len(z) != len(q):
            raise ValueError("joint and targets must have equal length.")
        if q.shape[1] != self.n_classes:
            raise ValueError("targets have the wrong number of classes.")
        if landmark_indices is None:
            landmark_indices = self.choose_landmarks(z)
        ids = np.asarray(landmark_indices, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != self.config.landmark_count:
            raise ValueError("landmark_indices have the wrong shape or count.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("landmark_indices must be unique.")
        if ids.min() < 0 or ids.max() >= len(z):
            raise ValueError("landmark_indices are out of range.")
        self.landmark_indices_ = ids.copy()
        self.landmarks_ = z[ids].copy()

        d = z.shape[1] + len(ids)
        normal = self.config.ridge * np.eye(d, dtype=np.float64)
        rhs = np.zeros((d, self.n_classes), dtype=np.float64)
        for start in range(0, len(z), self.config.chunk_size):
            phi = self.transform(z[start : start + self.config.chunk_size])
            normal += phi.T @ phi
            rhs += phi.T @ q[start : start + self.config.chunk_size]
        cf = cho_factor(normal, check_finite=False)
        self.weights_ = cho_solve(cf, rhs, check_finite=False)
        return self

    def predict_scores(self, joint: np.ndarray, chunk_size: int | None = None) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Student is not fitted.")
        z = np.asarray(joint)
        chunk = int(chunk_size or self.config.chunk_size)
        outputs: list[np.ndarray] = []
        for start in range(0, len(z), chunk):
            outputs.append(self.transform(z[start : start + chunk]) @ self.weights_)
        return np.concatenate(outputs, axis=0)

    def predict(self, joint: np.ndarray, chunk_size: int | None = None) -> np.ndarray:
        return np.argmax(self.predict_scores(joint, chunk_size=chunk_size), axis=1)

    def weight_hash(self) -> str:
        if self.weights_ is None:
            raise RuntimeError("Student is not fitted.")
        return hashlib.sha256(np.ascontiguousarray(self.weights_).view(np.uint8)).hexdigest()

    def state_dict(self) -> dict[str, np.ndarray]:
        if not self.fitted or self.landmark_indices_ is None:
            raise RuntimeError("Student is not fitted.")
        return {
            "student_landmark_indices": np.asarray(self.landmark_indices_, dtype=np.int64),
            "student_landmarks": np.asarray(self.landmarks_, dtype=np.float64),
            "student_weights": np.asarray(self.weights_, dtype=np.float64),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.landmark_indices_ = np.asarray(state["student_landmark_indices"], dtype=np.int64)
        self.landmarks_ = np.asarray(state["student_landmarks"], dtype=np.float64)
        self.weights_ = np.asarray(state["student_weights"], dtype=np.float64)

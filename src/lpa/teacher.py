from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ._checks import as_finite_2d, as_int_1d
from .config import TeacherConfig
from .kernels import rbf


def onehot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    y = as_int_1d(labels, "labels")
    if len(y) and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError("labels contain a class outside [0, n_classes).")
    return np.eye(n_classes, dtype=np.float64)[y]


def softmax_temperature(raw: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    z = np.asarray(raw, dtype=np.float64) / float(temperature)
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


class TrustedKernelTeacher:
    """Two-view kernel-ridge teacher fitted only from trusted labels."""

    def __init__(self, n_classes: int, config: TeacherConfig | None = None) -> None:
        self.n_classes = int(n_classes)
        self.config = config or TeacherConfig()
        self.view_a_anchor_: np.ndarray | None = None
        self.view_b_anchor_: np.ndarray | None = None
        self.alpha_a_: np.ndarray | None = None
        self.alpha_b_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return all(
            item is not None
            for item in (
                self.view_a_anchor_,
                self.view_b_anchor_,
                self.alpha_a_,
                self.alpha_b_,
            )
        )

    def fit(
        self,
        trusted_view_a: np.ndarray,
        trusted_view_b: np.ndarray,
        trusted_labels: np.ndarray,
    ) -> "TrustedKernelTeacher":
        a = as_finite_2d(trusted_view_a, "trusted_view_a")
        b = as_finite_2d(trusted_view_b, "trusted_view_b")
        y = as_int_1d(trusted_labels, "trusted_labels")
        if len(a) != len(b) or len(a) != len(y):
            raise ValueError("Trusted views and trusted labels must have equal length.")
        if len(y) == 0:
            raise ValueError("At least one trusted example is required.")
        target = onehot(y, self.n_classes)
        ka = rbf(a, a, self.config.gamma_view_a)
        kb = rbf(b, b, self.config.gamma_view_b)
        ca = cho_factor(
            ka + self.config.ridge * np.eye(len(y)),
            check_finite=False,
        )
        cb = cho_factor(
            kb + self.config.ridge * np.eye(len(y)),
            check_finite=False,
        )
        self.view_a_anchor_ = a.copy()
        self.view_b_anchor_ = b.copy()
        self.alpha_a_ = cho_solve(ca, target, check_finite=False)
        self.alpha_b_ = cho_solve(cb, target, check_finite=False)
        return self

    def predict_scores(
        self,
        view_a: np.ndarray,
        view_b: np.ndarray,
        chunk_size: int = 2000,
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Teacher is not fitted.")
        a = as_finite_2d(view_a, "view_a")
        b = as_finite_2d(view_b, "view_b")
        if len(a) != len(b):
            raise ValueError("view_a and view_b must have equal length.")
        assert self.view_a_anchor_ is not None and self.view_b_anchor_ is not None
        if a.shape[1] != self.view_a_anchor_.shape[1] or b.shape[1] != self.view_b_anchor_.shape[1]:
            raise ValueError(
                f"Teacher expects views with {self.view_a_anchor_.shape[1]} and "
                f"{self.view_b_anchor_.shape[1]} features; got {a.shape[1]} and {b.shape[1]}."
            )
        outputs: list[np.ndarray] = []
        wa = self.config.view_a_weight
        for start in range(0, len(a), chunk_size):
            sa = rbf(
                a[start : start + chunk_size],
                self.view_a_anchor_,
                self.config.gamma_view_a,
            ) @ self.alpha_a_
            sb = rbf(
                b[start : start + chunk_size],
                self.view_b_anchor_,
                self.config.gamma_view_b,
            ) @ self.alpha_b_
            outputs.append(wa * sa + (1.0 - wa) * sb)
        return np.concatenate(outputs, axis=0)

    def predict_proba(
        self,
        view_a: np.ndarray,
        view_b: np.ndarray,
        chunk_size: int = 2000,
    ) -> np.ndarray:
        return softmax_temperature(
            self.predict_scores(view_a, view_b, chunk_size=chunk_size),
            self.config.temperature,
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Teacher is not fitted.")
        return {
            "teacher_view_a_anchor": np.asarray(self.view_a_anchor_),
            "teacher_view_b_anchor": np.asarray(self.view_b_anchor_),
            "teacher_alpha_a": np.asarray(self.alpha_a_),
            "teacher_alpha_b": np.asarray(self.alpha_b_),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.view_a_anchor_ = np.asarray(state["teacher_view_a_anchor"], dtype=np.float64)
        self.view_b_anchor_ = np.asarray(state["teacher_view_b_anchor"], dtype=np.float64)
        self.alpha_a_ = np.asarray(state["teacher_alpha_a"], dtype=np.float64)
        self.alpha_b_ = np.asarray(state["teacher_alpha_b"], dtype=np.float64)

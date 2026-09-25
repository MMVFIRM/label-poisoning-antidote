from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np

from ._checks import as_finite_2d, as_int_1d
from .config import LinearStudentConfig, StudentConfig
from .student import LandmarkRidgeStudent, LinearRidgeStudent
from .teacher import onehot


@dataclass(frozen=True)
class MutationAuditReport:
    target_max_diff: float
    student_weight_max_diff: float
    target_hash_a: str
    target_hash_b: str
    weight_hash_a: str
    weight_hash_b: str

    @property
    def passed(self) -> bool:
        return self.target_max_diff == 0.0 and self.student_weight_max_diff == 0.0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["passed"] = self.passed
        return data


def _hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def validate_trusted_pairs(
    trusted_indices: np.ndarray,
    trusted_labels: np.ndarray,
    n_examples: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = as_int_1d(trusted_indices, "trusted_indices")
    y = as_int_1d(trusted_labels, "trusted_labels")
    if len(idx) != len(y):
        raise ValueError("trusted_indices and trusted_labels must be equal-length 1-D arrays.")
    if len(idx) == 0:
        raise ValueError("At least one trusted label is required.")
    if len(np.unique(idx)) != len(idx):
        raise ValueError("trusted_indices must be unique.")
    if idx.min() < 0 or idx.max() >= n_examples:
        raise ValueError("trusted_indices are out of range.")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError("trusted_labels contain an invalid class.")
    return idx, y


def build_targets(
    teacher_probabilities: np.ndarray,
    trusted_indices: np.ndarray,
    trusted_labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    p = as_finite_2d(teacher_probabilities, "teacher_probabilities")
    if p.shape[1] != n_classes:
        raise ValueError("teacher_probabilities have the wrong shape.")
    idx, y = validate_trusted_pairs(trusted_indices, trusted_labels, len(p), n_classes)
    q = p.copy()
    q[idx] = onehot(y, n_classes)
    return q


def mutation_invariance_audit(
    teacher_probabilities: np.ndarray,
    joint_features: np.ndarray,
    trusted_indices: np.ndarray,
    observed_labels_a: np.ndarray,
    observed_labels_b: np.ndarray,
    n_classes: int,
    student_config: StudentConfig | LinearStudentConfig | None = None,
) -> MutationAuditReport:
    """Verify that arbitrary changes outside the trusted set cannot alter the model.

    The two observed label arrays may contain arbitrary Python objects on untrusted
    indices. Only the trusted slices are converted to integers. A
    `LinearStudentConfig` (the default) audits the 2.0 linear student; a
    `StudentConfig` audits the v1.0 landmark student.
    """
    idx = as_int_1d(trusted_indices, "trusted_indices")
    a = np.asarray(observed_labels_a, dtype=object)
    b = np.asarray(observed_labels_b, dtype=object)
    if len(a) != len(b) or len(a) != len(teacher_probabilities):
        raise ValueError("Observed labels and teacher probabilities must have equal length.")
    ya = as_int_1d(a[idx], "trusted labels in observed_labels_a")
    yb = as_int_1d(b[idx], "trusted labels in observed_labels_b")
    if not np.array_equal(ya, yb):
        raise ValueError("Trusted labels changed; that is outside the untrusted-label audit.")
    qa = build_targets(teacher_probabilities, idx, ya, n_classes)
    qb = build_targets(teacher_probabilities, idx, yb, n_classes)

    config = student_config or LinearStudentConfig()
    sa: LandmarkRidgeStudent | LinearRidgeStudent
    sb: LandmarkRidgeStudent | LinearRidgeStudent
    if isinstance(config, StudentConfig):
        sa = LandmarkRidgeStudent(n_classes, config).fit(joint_features, qa)
        sb = LandmarkRidgeStudent(n_classes, config).fit(
            joint_features,
            qb,
            landmark_indices=sa.landmark_indices_,
        )
    else:
        sa = LinearRidgeStudent(n_classes, config).fit(joint_features, qa, trusted_indices=idx)
        sb = LinearRidgeStudent(n_classes, config).fit(joint_features, qb, trusted_indices=idx)
    assert sa.weights_ is not None and sb.weights_ is not None
    return MutationAuditReport(
        target_max_diff=float(np.max(np.abs(qa - qb))),
        student_weight_max_diff=float(np.max(np.abs(sa.weights_ - sb.weights_))),
        target_hash_a=_hash_array(qa),
        target_hash_b=_hash_array(qb),
        weight_hash_a=sa.weight_hash(),
        weight_hash_b=sb.weight_hash(),
    )

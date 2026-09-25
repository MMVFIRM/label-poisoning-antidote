from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ._checks import as_int_1d
from ._version import __version__
from .audit import build_targets, validate_trusted_pairs
from .config import LPAConfig
from .features import CIFARFeatureExtractor, KMeansPatchFeatureExtractor
from .student import LandmarkRidgeStudent, LinearRidgeStudent
from .teacher import BlendedKernelTeacher, TrustedKernelTeacher


class LabelPoisoningAntidote:
    """LPA training pipeline.

    Core design rule: the training API does not accept an untrusted label field.
    Only `trusted_indices` and their corresponding `trusted_labels` enter fit().

    The default architecture (`"kmeans"`, LPA 2.0) uses label-free k-means
    patch features, a blended trusted-only kernel teacher, and a linear ridge
    student. `LPAConfig.v1()` selects the frozen v1.0 landmark architecture,
    which is bit-identical to LPA 1.0.0.
    """

    FORMAT_VERSION = 3

    def __init__(self, config: LPAConfig | None = None) -> None:
        self.config = config or LPAConfig()
        self.feature_extractor = CIFARFeatureExtractor(self.config.feature)
        self.kmeans_extractor: KMeansPatchFeatureExtractor | None = None
        self.teacher: TrustedKernelTeacher | BlendedKernelTeacher
        self.student: LandmarkRidgeStudent | LinearRidgeStudent
        if self.config.architecture == "kmeans":
            self.kmeans_extractor = KMeansPatchFeatureExtractor(self.config.kmeans)
            self.teacher = BlendedKernelTeacher(
                self.config.n_classes, self.config.kmeans_teacher, self.config.teacher
            )
            self.student = LinearRidgeStudent(self.config.n_classes, self.config.linear_student)
        else:
            self.teacher = TrustedKernelTeacher(self.config.n_classes, self.config.teacher)
            self.student = LandmarkRidgeStudent(self.config.n_classes, self.config.student)
        self.trusted_indices_: np.ndarray | None = None
        self.trusted_labels_: np.ndarray | None = None
        self.target_hash_: str | None = None

    @property
    def architecture(self) -> str:
        return self.config.architecture

    @property
    def fitted(self) -> bool:
        return self.teacher.fitted and self.student.fitted

    @property
    def _chunk(self) -> int:
        return self.student.config.chunk_size

    @staticmethod
    def _hash_array(a: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

    def fit_views(
        self,
        view_a: np.ndarray,
        view_b: np.ndarray,
        joint: np.ndarray,
        trusted_indices: np.ndarray,
        trusted_labels: np.ndarray,
    ) -> "LabelPoisoningAntidote":
        """Fit from label-independent feature views.

        `view_a` and `view_b` feed the two-view kernel teacher. `joint` is the
        student representation; under the k-means architecture the teacher
        also uses it. For images, `fit_images()` builds all three.
        """
        a = np.asarray(view_a)
        b = np.asarray(view_b)
        z = np.asarray(joint)
        if a.ndim != 2 or b.ndim != 2 or z.ndim != 2:
            raise ValueError("All feature views must be 2-D arrays.")
        if not (len(a) == len(b) == len(z)):
            raise ValueError("All feature views must have equal length.")
        idx, y = validate_trusted_pairs(
            trusted_indices,
            trusted_labels,
            n_examples=len(z),
            n_classes=self.config.n_classes,
        )
        self.trusted_indices_ = idx.copy()
        self.trusted_labels_ = y.copy()
        if isinstance(self.teacher, BlendedKernelTeacher):
            assert isinstance(self.student, LinearRidgeStudent)
            self.teacher.fit(a[idx], b[idx], z[idx], y)
            p = self.teacher.predict_proba(a, b, z, chunk_size=self._chunk)
            q = build_targets(p, idx, y, self.config.n_classes)
            self.target_hash_ = self._hash_array(q)
            self.student.fit(z, q, trusted_indices=idx)
        else:
            assert isinstance(self.student, LandmarkRidgeStudent)
            self.teacher.fit(a[idx], b[idx], y)
            p = self.teacher.predict_proba(a, b, chunk_size=self._chunk)
            q = build_targets(p, idx, y, self.config.n_classes)
            self.target_hash_ = self._hash_array(q)
            self.student.fit(z, q)
        return self

    def image_views(self, x: np.ndarray, fit: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Label-free (view_a, view_b, joint) for CIFAR-like images.

        With `fit=True` the feature statistics (and, for the k-means
        architecture, the patch dictionary) are learned from `x` first.
        """
        chunk = self._chunk
        views = (
            self.feature_extractor.fit_transform(x, chunk_size=chunk)
            if fit
            else self.feature_extractor.transform(x, chunk_size=chunk)
        )
        if self.kmeans_extractor is None:
            return views.view_a, views.view_b, views.joint
        joint = self.kmeans_extractor.fit_transform(x) if fit else self.kmeans_extractor.transform(x)
        return views.view_a, views.view_b, joint

    def _image_joint(self, x: np.ndarray) -> np.ndarray:
        if self.kmeans_extractor is not None:
            if not self.kmeans_extractor.fitted:
                raise RuntimeError("Image feature extractor is not fitted.")
            return self.kmeans_extractor.transform(x)
        if not self.feature_extractor.fitted:
            raise RuntimeError("Image feature extractor is not fitted.")
        return self.feature_extractor.transform(x, chunk_size=self._chunk).joint

    def fit_images(
        self,
        x: np.ndarray,
        trusted_indices: np.ndarray,
        trusted_labels: np.ndarray,
    ) -> "LabelPoisoningAntidote":
        """Fit from CIFAR-like images: float in [0, 1] or integer 0-255 pixels."""
        a, b, z = self.image_views(x, fit=True)
        return self.fit_views(a, b, z, trusted_indices, trusted_labels)

    def predict_views(self, joint: np.ndarray) -> np.ndarray:
        return self.student.predict(joint)

    def predict_scores_views(self, joint: np.ndarray) -> np.ndarray:
        return self.student.predict_scores(joint)

    def predict_images(self, x: np.ndarray) -> np.ndarray:
        return self.student.predict(self._image_joint(x))

    def predict_scores_images(self, x: np.ndarray) -> np.ndarray:
        return self.student.predict_scores(self._image_joint(x))

    def teacher_probabilities_views(
        self,
        view_a: np.ndarray,
        view_b: np.ndarray,
        joint: np.ndarray | None = None,
    ) -> np.ndarray:
        if isinstance(self.teacher, BlendedKernelTeacher):
            if joint is None:
                raise ValueError("The k-means architecture teacher also needs the joint features.")
            return self.teacher.predict_proba(view_a, view_b, joint, chunk_size=self._chunk)
        return self.teacher.predict_proba(view_a, view_b, chunk_size=self._chunk)

    def score_images(self, x: np.ndarray, labels: np.ndarray) -> float:
        y = as_int_1d(labels, "labels")
        return float(np.mean(self.predict_images(x) == y))

    def save(self, directory: str | Path, include_trusted_labels: bool = True) -> Path:
        """Write `model.npz` and `metadata.json` to `directory`.

        Every saved array is recorded in the metadata with a SHA-256 digest that
        `load()` verifies. The saved arrays include trusted-example features
        (teacher anchors) and, for the v1.0 architecture, training-feature
        landmarks; treat the model directory as containing training data. Pass
        `include_trusted_labels=False` to omit trusted indices and labels,
        which are not needed for prediction.
        """
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {}
        if self.feature_extractor.fitted:
            state.update(self.feature_extractor.state_dict())
        if self.kmeans_extractor is not None and self.kmeans_extractor.fitted:
            state.update(self.kmeans_extractor.state_dict())
        state.update(self.teacher.state_dict())
        state.update(self.student.state_dict())
        if include_trusted_labels:
            if self.trusted_indices_ is not None:
                state["trusted_indices"] = self.trusted_indices_
            if self.trusted_labels_ is not None:
                state["trusted_labels"] = self.trusted_labels_
        np.savez_compressed(path / "model.npz", **state)
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "package_version": __version__,
            "architecture": self.config.architecture,
            "config": self.config.to_dict(),
            "target_hash": self.target_hash_,
            "student_weight_hash": self.student.weight_hash(),
            "array_sha256": {name: _canonical_array_hash(arr) for name, arr in sorted(state.items())},
            "validated_threat_model": "untrusted-label-only",
        }
        (path / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def fingerprint(directory: str | Path) -> str:
        """SHA-256 of a saved model's `metadata.json`.

        The metadata records a digest of every saved array, so this single value
        identifies the whole model. Store it separately from the model (for
        example in a deployment manifest) and pass it to `load()` to detect
        tampering; the in-directory hashes alone only detect corruption.
        """
        return hashlib.sha256((Path(directory) / "metadata.json").read_bytes()).hexdigest()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        expected_fingerprint: str | None = None,
    ) -> "LabelPoisoningAntidote":
        path = Path(directory)
        if expected_fingerprint is not None and cls.fingerprint(path) != expected_fingerprint:
            raise ValueError("Model fingerprint does not match the expected value.")
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        format_version = int(metadata["format_version"])
        if format_version not in (1, 2, cls.FORMAT_VERSION):
            raise ValueError(f"Unsupported model format version: {format_version}.")
        model = cls(LPAConfig.from_dict(metadata["config"]))
        with np.load(path / "model.npz", allow_pickle=False) as loaded:
            state = {name: loaded[name] for name in loaded.files}

        if format_version >= 2:
            expected_arrays = metadata["array_sha256"]
            if set(expected_arrays) != set(state):
                raise ValueError("Saved model arrays do not match the metadata manifest.")
            for name, digest in expected_arrays.items():
                if _canonical_array_hash(state[name]) != digest:
                    raise ValueError(f"Saved array {name!r} does not match its recorded hash.")
        else:
            warnings.warn(
                "Loading a format-1 model: only the student weights are integrity-checked. "
                "Re-save the model to upgrade to format 2.",
                stacklevel=2,
            )

        if "feature_color_mean" in state:
            model.feature_extractor.load_state_dict(state)
        if model.kmeans_extractor is not None and "kmeans_centroids" in state:
            model.kmeans_extractor.load_state_dict(state)
        model.teacher.load_state_dict(state)
        model.student.load_state_dict(state)
        model.trusted_indices_ = state.get("trusted_indices")
        model.trusted_labels_ = state.get("trusted_labels")
        model.target_hash_ = metadata.get("target_hash")
        expected = metadata.get("student_weight_hash")
        if expected and model.student.weight_hash() != expected:
            raise ValueError("Saved student-weight hash does not match model contents.")
        return model


def _canonical_array_hash(a: np.ndarray) -> str:
    """Platform-independent digest of an array's dtype, shape, and values."""
    arr = np.ascontiguousarray(a)
    little = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    h = hashlib.sha256()
    h.update(f"{little.dtype.str}|{little.shape}|".encode())
    h.update(little.tobytes())
    return h.hexdigest()

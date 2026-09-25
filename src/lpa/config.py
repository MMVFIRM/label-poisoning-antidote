from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

ARCHITECTURES = ("kmeans", "landmark")


@dataclass(frozen=True)
class FeatureConfig:
    color_floor: float = 0.01
    normalization_eps: float = 1e-12


@dataclass(frozen=True)
class TeacherConfig:
    gamma_view_a: float = 4.0
    gamma_view_b: float = 0.25
    view_a_weight: float = 0.75
    ridge: float = 0.1
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.gamma_view_a <= 0 or self.gamma_view_b <= 0:
            raise ValueError("Teacher kernel gammas must be positive.")
        if not 0.0 <= self.view_a_weight <= 1.0:
            raise ValueError("view_a_weight must be in [0, 1].")
        if self.ridge <= 0:
            raise ValueError("Teacher ridge must be positive.")
        if self.temperature <= 0:
            raise ValueError("Teacher temperature must be positive.")


@dataclass(frozen=True)
class StudentConfig:
    landmark_count: int = 256
    landmark_seed: int = 29002
    landmark_gamma: float = 0.5
    ridge: float = 1.0
    chunk_size: int = 2000

    def __post_init__(self) -> None:
        if self.landmark_count <= 0:
            raise ValueError("landmark_count must be positive.")
        if self.landmark_gamma <= 0:
            raise ValueError("landmark_gamma must be positive.")
        if self.ridge <= 0:
            raise ValueError("Student ridge must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


@dataclass(frozen=True)
class KMeansFeatureConfig:
    """Coates & Ng single-layer k-means patch features (Gate 34). Label-free."""

    centroids: int = 1600
    patch_size: int = 6
    patch_samples: int = 400_000
    iterations: int = 15
    seed: int = 0
    patch_norm_eps: float = 10.0
    zca_eps: float = 0.1
    standardize_floor: float = 1e-3
    chunk_size: int = 100
    workers: int | None = None

    def __post_init__(self) -> None:
        if self.centroids <= 0 or self.patch_samples <= 0 or self.iterations <= 0:
            raise ValueError("centroids, patch_samples, and iterations must be positive.")
        if not 1 <= self.patch_size <= 32:
            raise ValueError("patch_size must be in [1, 32].")
        if self.patch_norm_eps <= 0 or self.zca_eps <= 0 or self.standardize_floor <= 0:
            raise ValueError("patch_norm_eps, zca_eps, and standardize_floor must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.workers is not None and self.workers <= 0:
            raise ValueError("workers must be positive or None.")


@dataclass(frozen=True)
class KMeansTeacherConfig:
    """Blend of a joint-feature RBF kernel teacher with the two-view teacher (Gate 34)."""

    gamma: float = 0.5
    ridge: float = 0.1
    weight: float = 0.5
    temperature: float = 0.05

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.ridge <= 0 or self.temperature <= 0:
            raise ValueError("gamma, ridge, and temperature must be positive.")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("weight must be in [0, 1].")


@dataclass(frozen=True)
class LinearStudentConfig:
    """Linear ridge student on the joint features, with upweighted trusted rows (Gate 34)."""

    ridge: float = 0.03
    trusted_weight: float = 10.0
    chunk_size: int = 2000

    def __post_init__(self) -> None:
        if self.ridge <= 0:
            raise ValueError("Student ridge must be positive.")
        if self.trusted_weight <= 0:
            raise ValueError("trusted_weight must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")


@dataclass(frozen=True)
class LPAConfig:
    """Pipeline configuration.

    `architecture="kmeans"` (the 2.0 default) is the Gate-34 design: k-means
    patch features, a blended trusted-only teacher, and a linear ridge student.
    `architecture="landmark"` is the frozen v1.0 design; use `LPAConfig.v1()`.
    Each architecture reads only its own sub-configurations.
    """

    n_classes: int = 10
    architecture: str = "kmeans"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    kmeans: KMeansFeatureConfig = field(default_factory=KMeansFeatureConfig)
    kmeans_teacher: KMeansTeacherConfig = field(default_factory=KMeansTeacherConfig)
    linear_student: LinearStudentConfig = field(default_factory=LinearStudentConfig)

    def __post_init__(self) -> None:
        if self.n_classes <= 1:
            raise ValueError("n_classes must be at least 2.")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}.")

    @classmethod
    def v1(cls, n_classes: int = 10, **kwargs: Any) -> "LPAConfig":
        """The frozen v1.0 landmark architecture (bit-identical to LPA 1.0.0)."""
        return cls(n_classes=n_classes, architecture="landmark", **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LPAConfig":
        # Configurations saved before 2.0 have no architecture field and are v1.0.
        return cls(
            n_classes=int(data["n_classes"]),
            architecture=str(data.get("architecture", "landmark")),
            feature=FeatureConfig(**data.get("feature", {})),
            teacher=TeacherConfig(**data.get("teacher", {})),
            student=StudentConfig(**data.get("student", {})),
            kmeans=KMeansFeatureConfig(**data.get("kmeans", {})),
            kmeans_teacher=KMeansTeacherConfig(**data.get("kmeans_teacher", {})),
            linear_student=LinearStudentConfig(**data.get("linear_student", {})),
        )

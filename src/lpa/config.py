from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
class LPAConfig:
    n_classes: int = 10
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)

    def __post_init__(self) -> None:
        if self.n_classes <= 1:
            raise ValueError("n_classes must be at least 2.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LPAConfig":
        return cls(
            n_classes=int(data["n_classes"]),
            feature=FeatureConfig(**data.get("feature", {})),
            teacher=TeacherConfig(**data.get("teacher", {})),
            student=StudentConfig(**data.get("student", {})),
        )

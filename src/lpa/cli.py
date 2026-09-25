from __future__ import annotations

import argparse
import inspect
import json

import numpy as np

from . import __version__
from .audit import mutation_invariance_audit
from .config import KMeansTeacherConfig, LinearStudentConfig, LPAConfig, StudentConfig, TeacherConfig
from .federated import federated_equivalence_audit
from .pipeline import LabelPoisoningAntidote
from .student import trusted_row_weights
from .teacher import BlendedKernelTeacher, TrustedKernelTeacher


def _synthetic_problem(seed: int = 1):
    rng = np.random.default_rng(seed)
    n = 320
    classes = 4
    y = np.repeat(np.arange(classes), n // classes)
    rng.shuffle(y)
    centers_a = rng.normal(size=(classes, 12)) * 1.5
    centers_b = rng.normal(size=(classes, 6)) * 1.2
    a = centers_a[y] + 0.7 * rng.normal(size=(n, 12))
    b = centers_b[y] + 0.7 * rng.normal(size=(n, 6))
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    z = np.concatenate([a, b], axis=1) / np.sqrt(2.0)
    trusted = np.concatenate([np.flatnonzero(y == c)[:12] for c in range(classes)]).astype(np.int64)
    config = LPAConfig(
        n_classes=classes,
        teacher=TeacherConfig(gamma_view_a=1.0, gamma_view_b=1.0),
        student=StudentConfig(landmark_count=32, landmark_seed=29002, landmark_gamma=0.5),
        kmeans_teacher=KMeansTeacherConfig(gamma=1.0),
        linear_student=LinearStudentConfig(),
    )
    return a, b, z, y, trusted, config


def cmd_info(_: argparse.Namespace) -> int:
    fit_sig = str(inspect.signature(LabelPoisoningAntidote.fit_views))
    print(json.dumps({
        "name": "Label Poisoning Antidote",
        "version": __version__,
        "training_api": fit_sig,
        "default_architecture": LPAConfig().architecture,
        "architectures": {
            "kmeans": "LPA 2.0 default: k-means patch features, blended teacher, linear ridge student",
            "landmark": "LPA 1.0 frozen design, available as LPAConfig.v1()",
        },
        "untrusted_label_argument_present": False,
        "validated_threat_model": "untrusted-label-only",
    }, indent=2))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    a, b, z, y, trusted, config = _synthetic_problem(args.seed)
    labels_a = y.astype(object)
    labels_b = y.astype(object)
    untrusted = np.ones(len(y), dtype=bool)
    untrusted[trusted] = False
    labels_b[untrusted] = "UNTRUSTED_DO_NOT_READ"

    blended = BlendedKernelTeacher(config.n_classes, config.kmeans_teacher, config.teacher)
    blended.fit(a[trusted], b[trusted], z[trusted], y[trusted])
    v2 = mutation_invariance_audit(
        blended.predict_proba(a, b, z), z, trusted, labels_a, labels_b,
        config.n_classes, config.linear_student,
    )
    two_view = TrustedKernelTeacher(config.n_classes, config.teacher)
    two_view.fit(a[trusted], b[trusted], y[trusted])
    v1 = mutation_invariance_audit(
        two_view.predict_proba(a, b), z, trusted, labels_a, labels_b,
        config.n_classes, config.student,
    )
    passed = v2.passed and v1.passed
    print(json.dumps({"kmeans": v2.to_dict(), "landmark": v1.to_dict(), "passed": passed}, indent=2))
    return 0 if passed else 1


def cmd_federated_audit(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    phi = rng.normal(size=(300, 40))
    q = rng.random(size=(300, 5))
    q /= q.sum(axis=1, keepdims=True)
    order = rng.permutation(len(phi))
    partitions = [part.astype(np.int64) for part in np.array_split(order, 7)]
    diff = federated_equivalence_audit(phi, q, partitions, ridge=1.0)
    weights = trusted_row_weights(len(phi), order[:30], LinearStudentConfig().trusted_weight)
    weighted = federated_equivalence_audit(phi, q, partitions, ridge=LinearStudentConfig().ridge, weights=weights)
    result = {
        "max_weight_diff": diff,
        "max_weight_diff_trusted_weighted": weighted,
        "passed": bool(diff < 1e-10 and weighted < 1e-10),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lpa", description="Label Poisoning Antidote")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="Show the training interface, architectures, and threat model.")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("audit", help="Run a synthetic untrusted-label mutation audit for both architectures.")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("federated-audit", help="Run a synthetic centralized/federated equivalence audit.")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_federated_audit)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

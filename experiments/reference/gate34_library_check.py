#!/usr/bin/env python3
"""
Gate 34 library check: runs the installed `lpa` 2.0 default pipeline on full
CIFAR-10 and compares its student accuracy with the bundled Gate-34 harness
results (benchmarks/results/gate34_results.csv).

This goes through the public API. `image_views(x, fit=True)` followed by
`fit_views(...)` is exactly what `fit_images()` does; the features are
computed once and reused across the 20 trusted-set conditions.

  python gate34_library_check.py --data-dir ./cifar-10-batches-bin
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import lpa
from lpa import LabelPoisoningAntidote
from lpa.data import balanced_trusted_indices, load_cifar10_binary

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=ROOT / "benchmarks/results/gate34_results.csv")
    ap.add_argument("--output", type=Path, default=Path("gate34_library_check.csv"))
    args = ap.parse_args()

    t0 = time.time()
    x, y, xt, yt = load_cifar10_binary(args.data_dir)
    model = LabelPoisoningAntidote()
    assert model.architecture == "kmeans"
    a, b, z = model.image_views(x, fit=True)
    at, bt, zt = model.image_views(xt)
    print(f"lpa {lpa.__version__}: features {z.shape} {zt.shape} in {time.time() - t0:.0f}s", flush=True)

    ref = {(int(r["trusted_per_class"]), int(r["seed"])): r for r in csv.DictReader(args.reference.open())}
    rows = []
    for k, seed in sorted(ref):
        t1 = time.time()
        trusted = balanced_trusted_indices(y, k, seed)
        model.fit_views(a, b, z, trusted, y[trusted])
        acc = float(np.mean(model.predict_views(zt) == yt))
        teacher = float(np.mean(model.teacher_probabilities_views(at, bt, zt).argmax(1) == yt))
        expected = float(ref[(k, seed)]["g34_student_test_accuracy"])
        expected_teacher = float(ref[(k, seed)]["g34_teacher_test_accuracy"])
        row = dict(trusted_per_class=k, seed=seed, library_student_test_accuracy=acc,
                   harness_student_test_accuracy=expected, student_diff=acc - expected,
                   library_teacher_test_accuracy=teacher, harness_teacher_test_accuracy=expected_teacher,
                   teacher_diff=teacher - expected_teacher, seconds=time.time() - t1)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    worst = max(abs(r["student_diff"]) for r in rows)
    print(f"max |library - harness| student accuracy: {worst:.4f} over {len(rows)} conditions; "
          f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize gate34_results.csv (paired over seeds) and cross-check v1.0 columns against Gate 31."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("results", type=Path)
ap.add_argument("--gate31", type=Path, default=Path(__file__).parents[2] / "benchmarks/results/gate31_full_results.csv")
ap.add_argument("--output", type=Path, default=Path("gate34_summary.csv"))
a = ap.parse_args()

rows = list(csv.DictReader(a.results.open()))
g31 = {(r["trusted_per_class"], r["seed"]): r for r in csv.DictReader(a.gate31.open())}
mismatch = [
    (r["trusted_per_class"], r["seed"]) for r in rows
    if (r["trusted_per_class"], r["seed"]) in g31 and (
        abs(float(r["v1_student_test_accuracy"]) - float(g31[(r["trusted_per_class"], r["seed"])]["lpa_student_test_accuracy"])) > 1e-9
        or abs(float(r["v1_teacher_test_accuracy"]) - float(g31[(r["trusted_per_class"], r["seed"])]["teacher_test_accuracy"])) > 1e-9)
]
print(f"v1.0 columns vs bundled Gate 31: {len(rows) - len(mismatch)}/{len(rows)} conditions match exactly")

by = defaultdict(list)
for r in rows:
    by[int(r["trusted_per_class"])].append(r)
cols = ["v1_teacher_test_accuracy", "v1_student_test_accuracy", "g34_trusted_only_linear_test_accuracy",
        "g34_teacher_test_accuracy", "g34_student_test_accuracy"]
out = []
for k in sorted(by):
    rs = by[k]
    m = {c: np.array([float(r[c]) for r in rs]) for c in cols}
    s = m["g34_student_test_accuracy"]
    rec = dict(trusted_per_class=k, trusted_fraction=float(rs[0]["trusted_fraction"]), n_seeds=len(rs))
    rec.update({c.replace("_test_accuracy", "_mean"): float(v.mean()) for c, v in m.items()})
    rec["g34_student_sd"] = float(s.std(ddof=1))
    for ref, name in [("v1_student_test_accuracy", "v1_student"), ("v1_teacher_test_accuracy", "v1_teacher"),
                      ("g34_teacher_test_accuracy", "g34_teacher")]:
        d = s - m[ref]
        rec[f"gain_vs_{name}"] = float(d.mean())
        rec[f"wins_vs_{name}"] = int((d > 0).sum())
        rec[f"paired_t_p_vs_{name}"] = float(stats.ttest_rel(s, m[ref]).pvalue)
    rec["sentinel_target_max_diff"] = max(float(r["sentinel_target_max_diff"]) for r in rs)
    rec["sentinel_student_weight_max_diff"] = max(float(r["sentinel_student_weight_max_diff"]) for r in rs)
    rec["g34_all_label_reference"] = float(rs[0]["g34_all_label_reference_accuracy"])
    out.append(rec)
    print(f"{k:4d}/class  v1 student {rec['v1_student_mean']:.4f}  v1 teacher {rec['v1_teacher_mean']:.4f}  "
          f"g34 trusted-only {rec['g34_trusted_only_linear_mean']:.4f}  g34 teacher {rec['g34_teacher_mean']:.4f}  "
          f"g34 student {rec['g34_student_mean']:.4f}±{rec['g34_student_sd']:.4f}  "
          f"gain vs v1 {rec['gain_vs_v1_student']:+.4f} ({rec['wins_vs_v1_student']}/{len(rs)})  "
          f"vs g34 teacher {rec['gain_vs_g34_teacher']:+.4f} ({rec['wins_vs_g34_teacher']}/{len(rs)}, p={rec['paired_t_p_vs_g34_teacher']:.3g})")
with a.output.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print("wrote", a.output)

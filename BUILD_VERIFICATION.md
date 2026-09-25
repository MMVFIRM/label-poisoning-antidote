# Build verification — v2.0.0

Verification performed on Linux (Python 3.11, 4 CPU cores) on 2026-09-25.

## Source tests and static checks

```text
pytest: 40 passed
ruff check .: all checks passed
mypy: no issues found in 14 source files
```

The same suite runs in CI on Linux, Windows, and macOS for Python 3.10–3.13.

## Invariant audits

```text
lpa audit            kmeans and landmark: target_max_diff = 0.0,
                     student_weight_max_diff = 0.0, passed = true
lpa federated-audit  max |W_federated - W_centralized| = 9.7e-17 (unweighted),
                     1.1e-16 (trusted-weighted), passed = true
```

## Full-scale checks

- Gate 34 harness, full CIFAR-10, 20 conditions: sentinel target and final-weight
  differences 0; weighted federated/centralized difference 2.6e-11.
- Library check: the installed 2.0.0 default pipeline reproduces the Gate-34
  student and teacher test accuracy exactly in 20/20 conditions (1127 s total,
  including 499 s of feature extraction for 60,000 images).

## Equivalence with 1.0.0

- `LPAConfig.v1()` refits to the same student-weight hashes as 1.0.0 on the
  view and image fixtures on the platform that wrote them. CI on other
  platforms checks them to floating-point tolerance, plus identical predicted
  labels (`tests/test_v1_compat.py`).
- Model files written by 1.0.0 (format 2) load and give identical scores.
- The Gate-34 run recomputed the v1.0 student for all 20 Gate-31 conditions; it
  matches `gate31_full_results.csv` exactly. The v1.0 teacher matches in 19/20;
  in the other, one test image differs, from a solver/chunk-order float
  difference on a near-tie.

## Packaging

`python -m build` produced a wheel and sdist; both pass `twine check`. The wheel
was installed into a clean virtual environment, where `lpa info`, `lpa audit`,
and `lpa federated-audit` passed, and the sdist's test suite (including the
1.0.0 model fixtures) passed against the installed wheel.

```text
label_poisoning_antidote-2.0.0-py3-none-any.whl  ba62149b3f121bc3ff6116933a255371e08c411de2715f71185e4274744def7f
label_poisoning_antidote-2.0.0.tar.gz            06e6643017244b23dbe57b491f0dc83669ec169d590926f6f736f8944fa11b05
```

Build hashes depend on file timestamps, so a rebuild will not reproduce them
byte for byte.

# Build verification — v1.0.0

Verification performed on Windows 11 on 2026-09-25.

## Source tests and static checks

```text
pytest: 29 passed   (Python 3.11.9, 3.12.10, 3.13.7)
ruff check .: all checks passed
mypy: no issues found in 14 source files
```

The same suite runs in CI on Linux, Windows, and macOS for Python 3.10–3.13.

## Invariant audits

```text
lpa audit            target_max_diff = 0.0, student_weight_max_diff = 0.0, passed = true
lpa federated-audit  max |W_federated - W_centralized| = 2.8e-17, passed = true
```

## Equivalence with 1.0.0rc1

For identical valid inputs, 1.0.0 and 1.0.0rc1 produce identical `lpa audit`
output and identical student-weight and target hashes on an image-pipeline fit.
The bundled Gate-31/32/33 results, produced with the rc1 architecture, apply
unchanged. They were not recomputed for this release.

## Packaging

`python -m build` produced a wheel and sdist; both pass `twine check`. The wheel
was installed into clean virtual environments (Python 3.11, 3.12, 3.13), and
the CLI audits and the sdist's test suite passed against the installed wheel.

```text
label_poisoning_antidote-1.0.0-py3-none-any.whl  d5eed52a2b900096d40b44d56584c38239bac13d4c01672fcdf47c70472e6957
label_poisoning_antidote-1.0.0.tar.gz            84685d093b2764026f2b0e8e6038ddb487cd68b0214c7f0a448333cf7b0ce78a
```

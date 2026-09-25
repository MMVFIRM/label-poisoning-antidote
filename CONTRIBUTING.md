# Contributing

Changes to either architecture (the 2.0 default or the frozen v1.0 preset)
must preserve two invariants:

1. No untrusted label field may enter target construction, landmark selection,
   feature extraction (including the k-means dictionary), trusted-row weighting,
   or the student objective.
2. Full-participation federated sufficient-statistic aggregation must match the
   centralized ridge solution within numerical tolerance.

Before submitting a change:

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
mypy
lpa audit
lpa federated-audit
```

Architecture changes should be proposed separately from bug fixes so the
validated behavior remains auditable. `LPAConfig.v1()` must stay bit-identical
to LPA 1.0.0; `tests/test_v1_compat.py` checks this against model files written
by 1.0.0.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).

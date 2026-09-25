# Contributing

Changes to the frozen v1.0 core should preserve two invariants:

1. No untrusted label field may enter target construction, landmark selection,
   feature extraction, or the student objective.
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
validated v1.0 behavior remains auditable.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).

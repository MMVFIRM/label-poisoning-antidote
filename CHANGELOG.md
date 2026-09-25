# Changelog

## 1.0.0 — 2026-09-25

Models trained with valid inputs are bit-identical to 1.0.0rc1, so the bundled
Gate-31/32/33 results apply unchanged.

### Added

- MIT license.
- Model format 2: `metadata.json` records a SHA-256 digest of every saved
  array, and `load()` verifies all of them (rc1 only checked student weights).
- `LabelPoisoningAntidote.fingerprint()` and `load(..., expected_fingerprint=)`
  for tamper evidence using an out-of-band fingerprint.
- `save(..., include_trusted_labels=False)` to omit trusted indices and labels.
- Tests for input validation, model-file integrity, the image pipeline
  round trip, and version consistency (8 → 29 tests).
- CI on Linux, Windows, and macOS for Python 3.10–3.13, plus ruff, mypy, and a
  build → `twine check` → clean-install → audit job for the wheel.

### Changed

- Trusted labels and indices must be integers. Non-integral floats, booleans,
  and strings now raise `ValueError` instead of being silently truncated.
- NaN or infinite features, images, targets, and federated statistics now raise
  `ValueError` instead of producing NaN predictions.
- Prediction with the wrong feature width raises a clear `ValueError`.
- Federated aggregation rejects asymmetric client Gram matrices.
- The package version has a single source (`lpa/_version.py`).
- Package metadata: license, author, Python 3.13, `Typing :: Typed`.
- Documentation: model files contain training data; serialization integrity
  and tamper evidence; secure aggregation is the deployer's responsibility.

### Removed

- `LICENSE-NOTICE.md` (replaced by `LICENSE`).
- Build output and caches that were shipped in the rc1 archive.

## 1.0.0rc1 — 2026-09-25

- Froze the open HOG + spatial-RGB trusted teacher architecture.
- Froze direct Gate-26 soft targets; confidence routing and recalibration remain
  diagnostics only.
- Froze the 256 random x-only RBF landmark ridge student.
- Added a training API with no untrusted-label argument.
- Added explicit target and final-student mutation audits.
- Added exact federated ridge sufficient-statistic aggregation.
- Bundled full CIFAR-10 Gate-31 and Gate-32 qualification results.
- Bundled the Gate-33 matched FairMean/FedAvg/q-FFL comparison and external
  reproduction scripts.
- Removed proprietary VIVERE/MACSL components from the release line.

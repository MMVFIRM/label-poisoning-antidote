# Changelog

## 2.0.0 — 2026-09-25

LPA 2.0 makes the Gate-34 design the default. On full CIFAR-10 the default
student reaches 49.6% / 55.9% / 61.4% / 66.4% at 0.5% / 1% / 2% / 5% trusted
labels (1.0.0: 38.6% / 44.1% / 47.6% / 51.5%). It beats its own trusted-only
teacher at every budget, and untrusted-label invariance is unchanged.

### Breaking

- `LabelPoisoningAntidote()` and `LPAConfig()` now use
  `architecture="kmeans"`. New default models differ from 1.0.0 models: they
  have different features, parameters, predictions, and model files. Use
  `LPAConfig.v1()` for the 1.0.0 architecture, which is bit-identical.
- `mutation_invariance_audit(..., student_config=None)` now audits the 2.0
  linear student. Pass a `StudentConfig` to audit the v1.0 landmark student.
- `lpa audit` reports both architectures in one JSON object
  (`{"kmeans": ..., "landmark": ..., "passed": ...}`).
- Model format 3 adds `architecture` to `metadata.json`.

### Added

- `KMeansPatchFeatureExtractor`: label-free Coates & Ng k-means patch features
  (1600 centroids, 6400-D), threaded encoding.
- `BlendedKernelTeacher`: trusted-only blend of a joint-feature RBF kernel ridge
  and the v1.0 two-view teacher.
- `LinearRidgeStudent` with trusted-row weighting, and `trusted_row_weights()`.
- Configs: `KMeansFeatureConfig`, `KMeansTeacherConfig`, `LinearStudentConfig`,
  `LPAConfig.architecture`, `LPAConfig.v1()`.
- `LabelPoisoningAntidote.image_views()`, `predict_scores_images()`, and an
  optional `joint` argument to `teacher_probabilities_views()`.
- Integer (0-255) image input is accepted alongside floats in [0, 1].
- Weighted federated statistics: `client_sufficient_statistics(..., weights=)`,
  and `weights=` on `centralized_ridge` and `federated_equivalence_audit`.
- `lpa federated-audit` also checks the trusted-weighted form.
- Gate-34 full-scale harness, summarizer, and library check in
  `experiments/reference/`, with results in `benchmarks/results/gate34_*`.
- Tests: the 2.0 pipeline, feature determinism, the weighted student,
  federated equivalence, tamper evidence, and 1.0.0 model-file compatibility
  (40 tests).

### Fixed

- mypy errors under recent NumPy type stubs (hashing used ndarray views).
  Hash values are unchanged.

### Compatibility

- Format-2 model files written by 1.0.0 load as the v1.0 architecture and
  predict identically (checked against fixtures written by 1.0.0).

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

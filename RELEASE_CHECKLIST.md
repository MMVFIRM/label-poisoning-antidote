# LPA release checklist

## Completed for 1.0.0rc1

- [x] Frozen core architecture.
- [x] Full CIFAR-10 centralized qualification: 20 runs.
- [x] Full CIFAR-10 federated/non-IID qualification: 15 runs.
- [x] Target mutation invariance at full scale.
- [x] Final-student mutation invariance at full scale.
- [x] Centralized/federated equivalence at full scale.
- [x] CI tests for core invariants.
- [x] Model serialization and integrity hash.
- [x] Threat-model and deployment-boundary documentation.
- [x] Matched external FairMean/FedAvg/q-FFL method-level checkpoint.

## Completed for 1.0.0

- [x] Select an explicit open-source license (MIT).
- [x] Add maintainer/author metadata and repository URLs.
- [x] Integrity-check every saved model array; optional out-of-band fingerprint.
- [x] Reject invalid inputs instead of silently truncating or propagating NaN.
- [x] Cross-platform CI (Linux/Windows/macOS, Python 3.10–3.13), lint, type check.
- [x] Built-wheel verification in CI.
- [x] Secure aggregation: not built in; documented as the deployer's
      responsibility in `docs/DEPLOYMENT.md`.
- [x] Confirm models are bit-identical to 1.0.0rc1 for valid inputs.

## Completed for 2.0.0

- [x] Gate 34 full CIFAR-10 centralized qualification: 20 runs, dev-split model
      selection, a single test-set evaluation.
- [x] Target and final-student mutation invariance at full scale for 2.0.
- [x] Weighted federated/centralized equivalence at full scale.
- [x] Library reproduces the Gate-34 harness through its public API.
- [x] `LPAConfig.v1()` bit-identical to 1.0.0; 1.0.0 model files load unchanged.
- [x] mypy clean under current NumPy stubs.

## Open after 2.0.0

- [ ] Federated partial-participation study (Gate-32 style) for the 2.0 default.
- [ ] Trusted-label contamination stress test for the 2.0 default.
- [ ] External baselines (Gate-33 style) for the 2.0 default.
- [ ] Measure fitting the k-means dictionary on public or trusted-only images
      for federated deployments that cannot pool images.

- [ ] Independent reproduction by a second person. (Gate 31 was reproduced to
      four decimals in a second environment on separately obtained CIFAR-10
      data during Gate 34.)
- [ ] Complete or explicitly waive the paper-faithful external GPU FairMean run
      (`experiments/external/run_official_fairmean_full_optional.sh`). Needs a
      GPU environment; the one-seed sanity script is sufficient to use LPA.
- [ ] Automated PyPI publication (trusted publishing) on tagged releases.
- [ ] Reproducible container image.
- [ ] Additional datasets beyond Fashion-MNIST/CIFAR-10.
- [ ] Production monitoring for partial participation and client skew.

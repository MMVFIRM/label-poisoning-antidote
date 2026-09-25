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

## Open after 1.0.0

- [ ] Independent reproduction by a second environment/person.
- [ ] Complete or explicitly waive the paper-faithful external GPU FairMean run
      (`experiments/external/run_official_fairmean_full_optional.sh`). Needs a
      GPU environment; the one-seed sanity script is sufficient to use LPA.
- [ ] Automated PyPI publication (trusted publishing) on tagged releases.
- [ ] Reproducible container image.
- [ ] Additional datasets beyond Fashion-MNIST/CIFAR-10.
- [ ] Production monitoring for partial participation and client skew.

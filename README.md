# Label Poisoning Antidote (LPA) v1.0

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

LPA is a learning architecture for settings where some labels
are explicitly **untrusted** but a smaller trusted label set is available.

The core design is deliberately simple:

```text
x ──> fixed open features ──> trusted-only teacher ──> q_T(x)
|                                                     |
└────────> x-only random RBF landmarks ───────────────┘
                                                      |
                                                      v
                                                 ridge student

untrusted supplied label y~ ──X──> no training path
```

The strongest validated property is direct invariance to arbitrary replacement
of the designated untrusted label field under the documented label-only threat
model:

```text
M(X, Y_trusted, Y_untrusted) = M(X, Y_trusted, Y'_untrusted)
```

because the core training API never accepts `Y_untrusted`.

This is **not** a claim of universal poisoning immunity. See
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

## Release status

Version: **1.0.0**

The architecture is frozen. Version 1.0.0 produces bit-identical models to the
1.0.0rc1 candidate that generated the bundled Gate-31/32 results; the changes
since then are input validation, model-file integrity, packaging, and CI. See
[`CHANGELOG.md`](CHANGELOG.md).

Independent reproduction of the full CIFAR-10 results by a second party has
not yet happened. Reports, whether they match or not, are welcome as GitHub
issues.

## Frozen architecture

The validated image configuration uses:

- 576-D fixed HOG-style gradient features;
- 48-D 4x4 spatial RGB block means;
- a trusted-only two-view RBF kernel-ridge teacher;
- direct soft teacher targets on untrusted examples;
- hard one-hot targets only on trusted examples;
- 256 fixed-seed, x-only random landmarks;
- `[joint feature ; RBF-to-landmarks]` ridge student;
- optional exact federated aggregation via ridge sufficient statistics.

Core defaults are frozen in `lpa.LPAConfig` and match the Gate-26 through
Gate-32 qualification line.

## Install

```bash
python -m pip install -e ".[test]"
```

Run the two invariant checks:

```bash
lpa audit
lpa federated-audit
```

## Minimal API

```python
import numpy as np
from lpa import LabelPoisoningAntidote

# x_train is CIFAR-like RGB data: (N, 3072), (N, 3, 32, 32), or (N, 32, 32, 3)
# Only trusted labels are passed into fit_images().
trusted_idx = np.array([1, 7, 12, 25])
trusted_y = np.array([0, 4, 2, 1])

model = LabelPoisoningAntidote()
model.fit_images(x_train, trusted_idx, trusted_y)

pred = model.predict_images(x_test)
model.save("lpa_model")
```

Saved models are integrity-checked on load. To also detect deliberate
tampering, record the model fingerprint somewhere the model directory cannot
modify and pass it back at load time:

```python
fp = LabelPoisoningAntidote.fingerprint("lpa_model")  # store out-of-band
model = LabelPoisoningAntidote.load("lpa_model", expected_fingerprint=fp)
```

A saved model contains trusted-example features and training-feature
landmarks, so treat model files as containing training data. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

There is intentionally no `untrusted_labels=` parameter.

For generic non-image experiments, use `fit_views(view_a, view_b, joint, ...)`
with your own label-independent feature views. Only the CIFAR-like HOG+color
configuration has the full v1.0 qualification record.

## Full-scale qualification snapshot

### Gate 31 — centralized CIFAR-10

Full 50,000-train / 10,000-test CIFAR-10, five seeds per trusted budget:

| Trusted fraction | Teacher | LPA student | Trusted-only | All-label reference | Naive pairwise |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 39.52% | **38.57%** | 35.72% | 58.28% | 6.06% |
| 1% | 45.68% | **44.12%** | 41.12% | 58.28% | 6.06% |
| 2% | 51.45% | **47.61%** | 46.33% | 58.28% | 6.07% |
| 5% | 57.01% | 51.48% | **51.63%** | 58.28% | 6.28% |

Every one of the 20 runs reported zero target change when all untrusted labels
were replaced by a nonnumeric sentinel. At 5% trusted labels, trusted-only
training slightly overtook the frozen LPA student; this is documented as a
utility crossover, not an invariance failure.

### Gate 32 — federated/non-IID CIFAR-10

Full 50,000-train / 10,000-test, 10 clients, 2% trusted labels, five partition
seeds at each Dirichlet alpha:

- centralized LPA test accuracy: **47.59%**;
- federated full-participation accuracy: **47.59%** in all 15 partitions;
- maximum centralized/federated parameter difference: approximately `6.1e-11`;
- maximum poisoned-label target difference: **0**;
- maximum poisoned-label final-student parameter difference: **0**.

The main federated deployment boundary is partial participation under severe
non-IID skew, not label poisoning.

### Gate 33 — external baseline checkpoint

A model-capacity-matched CPU checkpoint compared FedAvg, FairMean, q-FFL, LPA,
and trusted-only learning. FairMean retained substantially more raw-label
utility when most labels remained clean, while LPA remained invariant and
crossed the raw-label methods only as corruption became extensive. This is a
**partial external-baseline qualification**, not a claim that LPA dominates
FairMean overall.

See [`docs/VALIDATED_CLAIMS.md`](docs/VALIDATED_CLAIMS.md) and
[`benchmarks/README.md`](benchmarks/README.md).

### Gate 34 — post-release research candidate

Swapping in label-free k-means patch features, with no change to the
no-untrusted-label rule, raises the full CIFAR-10 student to **49.6% / 55.9% /
61.4% / 66.4%** at 0.5% / 1% / 2% / 5% trusted labels. The student now beats
its own trusted-only teacher at every budget, which removes the v1.0 5%
crossover. Sentinel target and final-weight differences are zero in all 20
runs. This is not yet the library default; see
[`docs/GATE34_RESEARCH.md`](docs/GATE34_RESEARCH.md).

## Federated form

For client `i`:

```text
A_i = Phi_i^T Phi_i
B_i = Phi_i^T Q_i
```

The server solves:

```text
W = (lambda I + sum_i A_i)^-1 sum_i B_i
```

With full participation this is the same normal equation as centralized ridge,
regardless of how the same examples are partitioned among clients. See
[`docs/FEDERATED.md`](docs/FEDERATED.md).

## Reproduce the validated benchmarks

The exact Gate-31 and Gate-32 reference harnesses are preserved under
`experiments/reference/`.

Gate 31:

```bash
python experiments/reference/gate31_full_cifar_harness.py \
  --data-dir ./cifar-10-batches-bin \
  --trusted-per-class 25 50 100 250 \
  --seeds 31001 31002 31003 31004 31005
```

Gate 32 reuses the Gate-31 feature cache:

```bash
python experiments/reference/gate32_full_federated_harness.py \
  --data-dir ./cifar-10-batches-bin \
  --feature-cache ./gate31_features.npz \
  --trusted-per-class 100
```

The long paper-faithful FairMean comparison is **not** required to use LPA.
A one-seed external-code sanity harness is supplied in `experiments/external/`.

## Scope and nonclaims

LPA v1.0 does not claim robustness to corrupted trusted labels, input
poisoning, data injection/removal, malicious executable client updates,
availability attacks, privacy attacks, or arbitrary Byzantine behavior. It
also does not infer the true untrusted labels when the problem is not
identifiable from trusted evidence and features.

LPA is designed to make one channel explicit and auditable:
**the untrusted label field is not an input to learning.**

## Repository map

- `src/lpa/` — reusable implementation.
- `tests/` — invariant, serialization, feature, and federated tests.
- `docs/` — method, threat model, claims, deployment boundaries, research history.
- `benchmarks/` — preserved Gate-31/32/33 result tables and figures.
- `experiments/reference/` — exact validated full-data harnesses.
- `experiments/external/` — optional FairMean external sanity scripts.
- `examples/` — small synthetic demonstrations.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
mypy
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the two invariants every change
must preserve.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Richard Aragon.

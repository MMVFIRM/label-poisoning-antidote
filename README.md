# Label Poisoning Antidote (LPA) v2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

LPA is a learning architecture for settings where some labels
are explicitly **untrusted** but a smaller trusted label set is available.

The core design is deliberately simple:

```text
x ──> label-free features ──> trusted-only teacher ──> q_T(x)
|                                                     |
└────────> same x-only features ──────────────────────┘
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

Version: **2.0.0**

LPA 2.0 makes the Gate-34 design the default. The student is **11 to 15
points more accurate** than v1.0 on full CIFAR-10 at every tested trusted
budget, and it now beats its own trusted-only teacher. The invariance
property and exact federated aggregation are unchanged.

The frozen v1.0 architecture is still available as `LPAConfig.v1()` and
reproduces 1.0.0 exactly (bit-identical on the same platform). Model files saved by 1.0.0 load and predict unchanged.
See [`CHANGELOG.md`](CHANGELOG.md) for the breaking changes.

Independent reproduction of the full CIFAR-10 results by a second party has
not yet happened. Reports, whether they match or not, are welcome as GitHub
issues.

## Architecture

The default (`architecture="kmeans"`) image configuration uses:

- 6400-D Coates & Ng k-means patch features: 1600 centroids learned from
  training **images only**, triangle encoding, 2x2 quadrant pooling;
- the v1.0 576-D HOG and 48-D spatial RGB views for the teacher;
- a trusted-only teacher that blends an RBF kernel ridge on the k-means
  features with the v1.0 two-view HOG/color kernel teacher;
- direct soft teacher targets (temperature 0.05) on untrusted examples;
- hard one-hot targets only on trusted examples, upweighted x10;
- a linear ridge student on the k-means features;
- optional exact federated aggregation via ridge sufficient statistics.

`LPAConfig.v1()` selects the v1.0 design: HOG + color features, the two-view
teacher, and a ridge student over 256 x-only random RBF landmarks.

## Install

```bash
python -m pip install -e ".[test]"
```

Run the two invariant checks (both architectures):

```bash
lpa audit
lpa federated-audit
```

## Minimal API

```python
import numpy as np
from lpa import LabelPoisoningAntidote

# x_train is CIFAR-like RGB data: (N, 3072), (N, 3, 32, 32), or (N, 32, 32, 3),
# as floats in [0, 1] or integer 0-255 pixels.
# Only trusted labels are passed into fit_images().
trusted_idx = np.array([1, 7, 12, 25])
trusted_y = np.array([0, 4, 2, 1])

model = LabelPoisoningAntidote()          # LPA 2.0 default
model.fit_images(x_train, trusted_idx, trusted_y)

pred = model.predict_images(x_test)
model.save("lpa_model")
```

Feature extraction for 60,000 CIFAR images takes about 7 minutes on 4 CPU
cores, and the 6400-D features for 50,000 images need about 2.6 GB as float64.
Use `LabelPoisoningAntidote(LPAConfig.v1())` for the lighter v1.0 model.

Saved models are integrity-checked on load. To also detect deliberate
tampering, record the model fingerprint somewhere the model directory cannot
modify and pass it back at load time:

```python
fp = LabelPoisoningAntidote.fingerprint("lpa_model")  # store out-of-band
model = LabelPoisoningAntidote.load("lpa_model", expected_fingerprint=fp)
```

A saved model contains trusted-example features and a patch dictionary learned
from training images, so treat model files as containing training data. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

There is intentionally no `untrusted_labels=` parameter.

For generic non-image experiments, use `fit_views(view_a, view_b, joint, ...)`
with your own label-independent feature views. Under the default architecture
`joint` is both the student representation and the teacher's third view. Only
the CIFAR-like image configurations have a full qualification record.

## Full-scale qualification snapshot

### Gate 34 — LPA 2.0 default, centralized CIFAR-10

Full 50,000-train / 10,000-test CIFAR-10, the Gate-31 trusted seeds, five seeds
per budget:

| Trusted fraction | v1.0 student | 2.0 trusted-only linear | 2.0 teacher | **2.0 student** | All-label reference |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 38.57% | 40.73% | 46.51% | **49.63%** | 76.49% |
| 1% | 44.12% | 45.56% | 52.68% | **55.90%** | 76.49% |
| 2% | 47.61% | 50.77% | 58.43% | **61.35%** | 76.49% |
| 5% | 51.48% | 58.39% | 65.36% | **66.41%** | 76.49% |

- The 2.0 student beats the v1.0 student, the trusted-only linear model, and
  its own teacher in 20/20 runs.
- Replacing every untrusted label with a nonnumeric sentinel changed neither
  the targets nor the final student weights in any of the 20 runs.
- Hyperparameters were selected on a dev split held out of the training set,
  not on the test set.

The installed library reproduces these numbers through its public API
(`experiments/reference/gate34_library_check.py`). Details and negative
results are in [`docs/GATE34_RESEARCH.md`](docs/GATE34_RESEARCH.md).

### Gate 31 — v1.0, centralized CIFAR-10

| Trusted fraction | Teacher | LPA student | Trusted-only | All-label reference | Naive pairwise |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 39.52% | **38.57%** | 35.72% | 58.28% | 6.06% |
| 1% | 45.68% | **44.12%** | 41.12% | 58.28% | 6.06% |
| 2% | 51.45% | **47.61%** | 46.33% | 58.28% | 6.07% |
| 5% | 57.01% | 51.48% | **51.63%** | 58.28% | 6.28% |

In v1.0 the trusted-only kernel teacher was more accurate than the distilled
student, and at 5% trusted labels trusted-only training overtook it. Gate 34
removes both gaps.

### Gate 32 — v1.0, federated/non-IID CIFAR-10

Full 50,000-train / 10,000-test, 10 clients, 2% trusted labels, five partition
seeds at each Dirichlet alpha:

- centralized LPA test accuracy: **47.59%**;
- federated full-participation accuracy: **47.59%** in all 15 partitions;
- maximum centralized/federated parameter difference: approximately `6.1e-11`;
- maximum poisoned-label target difference: **0**;
- maximum poisoned-label final-student parameter difference: **0**.

The 2.0 student uses the same normal equations with trusted-row weights. Its
10-client Dirichlet(0.1) check matched centralized training to `2.6e-11`. The
partial-participation study has not been repeated for 2.0.

### Gate 33 — external baseline checkpoint (v1.0)

A model-capacity-matched CPU checkpoint compared FedAvg, FairMean, q-FFL, LPA,
and trusted-only learning. FairMean retained substantially more raw-label
utility when most labels remained clean, while LPA remained invariant and
crossed the raw-label methods only as corruption became extensive. This is a
**partial external-baseline qualification**, not a claim that LPA dominates
FairMean overall. It has not been repeated for 2.0.

See [`docs/VALIDATED_CLAIMS.md`](docs/VALIDATED_CLAIMS.md) and
[`benchmarks/README.md`](benchmarks/README.md).

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
regardless of how the same examples are partitioned among clients. In 2.0,
trusted rows carry weight 10, so `A_i = Phi_i^T W_i Phi_i` and
`B_i = Phi_i^T W_i Q_i` with a diagonal row-weight matrix `W_i`. See
[`docs/FEDERATED.md`](docs/FEDERATED.md).

## Reproduce the validated benchmarks

The exact Gate-31, Gate-32, and Gate-34 reference harnesses are preserved under
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

Gate 34 reuses the Gate-31 feature cache and builds the k-means features on
its first run:

```bash
python experiments/reference/gate34_full_cifar_harness.py \
  --data-dir ./cifar-10-batches-bin \
  --gate31-cache ./gate31_features.npz \
  --kmeans-cache ./gate34_kmeans_1600.npz
python experiments/reference/gate34_summarize.py gate34_results.csv
```

To check the installed library against the bundled Gate-34 results:

```bash
python experiments/reference/gate34_library_check.py --data-dir ./cifar-10-batches-bin
```

The long paper-faithful FairMean comparison is **not** required to use LPA.
A one-seed external-code sanity harness is supplied in `experiments/external/`.

## Scope and nonclaims

LPA does not claim robustness to corrupted trusted labels, input
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
- `benchmarks/` — preserved Gate-31/32/33/34 result tables and figures.
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

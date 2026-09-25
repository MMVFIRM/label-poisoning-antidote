# LPA v2.0.0 release notes

LPA 2.0.0 replaces the default architecture with the Gate-34 design and keeps
the v1.0 architecture available unchanged.

## What changed

The v1.0 student was limited by its HOG + color representation: even with all
labels it reached only 58% on CIFAR-10. It was also less accurate than its own
trusted-only kernel teacher, so distilling onto the untrusted pool cost
accuracy. LPA 2.0 adds label-free Coates & Ng k-means patch features, learned
from training images only, and uses them in both the trusted-only teacher and a
linear ridge student.

| Trusted fraction | 1.0.0 student | **2.0.0 student** |
|---:|---:|---:|
| 0.5% | 38.57% | **49.63%** |
| 1% | 44.12% | **55.90%** |
| 2% | 47.61% | **61.35%** |
| 5% | 51.48% | **66.41%** |

Full CIFAR-10, five seeds per budget. The 2.0 student beats its own teacher at
every budget, which removes the v1.0 high-trust crossover. Replacing every
untrusted label with a sentinel changed neither the targets nor the final
weights in any run.

## What did not change

- The training API still has no untrusted-label argument.
- Exact federated aggregation still holds; trusted rows now carry a fixed
  weight of 10.
- `LPAConfig.v1()` is bit-identical to 1.0.0, and 1.0.0 model files load and
  predict unchanged.

## Costs

The 2.0 default needs about 7 minutes of feature extraction for 60,000 CIFAR
images on 4 CPU cores and about 2.6 GB for the 50,000-image feature matrix.
Its federated statistics are about 78 MiB per client (float32), compared with
1.5 MiB for v1.0. Use `LPAConfig.v1()` where those costs matter.

See [`CHANGELOG.md`](CHANGELOG.md) for the full list of changes and
[`docs/GATE34_RESEARCH.md`](docs/GATE34_RESEARCH.md) for the evidence.

## Claims

Full-data qualification evidence is bundled in `benchmarks/results/`. These
results have not yet been independently reproduced by a second party.

LPA is intentionally conservative about claims. It guarantees direct non-use
of an explicitly untrusted label field in the supplied implementation; it does
not claim recovery of the latent true label or robustness to threats outside
the documented model. See [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).

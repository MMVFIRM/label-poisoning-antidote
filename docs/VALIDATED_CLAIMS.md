# Validated claims and claim boundaries

Claims 1-5 were established for the v1.0 architecture (`LPAConfig.v1()`).
Claim 6 covers the LPA 2.0 default architecture (Gate 34).

## Claim 1 — direct untrusted-label invariance

The core API has no untrusted-label training argument. Target construction uses
only trusted labels plus x-derived teacher outputs.

Full-scale Gate-31 evidence:

- CIFAR-10: 50,000 train / 10,000 official test;
- 20 conditions across 0.5%, 1%, 2%, and 5% trusted labels;
- all 20 report `sentinel_target_max_diff = 0` after replacing every untrusted
  label with a nonnumeric sentinel.

Full-scale Gate-32 evidence strengthens this to final model parameters:

- 15 federated/non-IID conditions;
- maximum `poison_target_max_diff = 0`;
- maximum `poison_student_weight_max_diff = 0`.

## Claim 2 — scarce trusted labels can gain from invariant pseudo-targets

Gate 31, five seeds per budget:

| Trusted fraction | LPA | Trusted-only | Mean LPA gain |
|---:|---:|---:|---:|
| 0.5% | 38.57% | 35.72% | +2.84 pt |
| 1% | 44.12% | 41.12% | +3.00 pt |
| 2% | 47.61% | 46.33% | +1.28 pt |
| 5% | 51.48% | 51.63% | -0.15 pt |

The benefit disappears at the tested 5% trusted regime for this frozen student.
That crossover is a deployment boundary, not a universal threshold.

## Claim 3 — exact full-participation federated form

For client sufficient statistics

```text
A_i = Phi_i^T Phi_i
B_i = Phi_i^T Q_i,
```

the server solution

```text
W = (lambda I + sum_i A_i)^-1 sum_i B_i
```

is algebraically identical to centralized ridge on the union of participating
examples.

Gate 32 full-scale evidence:

- 50,000 train / 10,000 test;
- 10 clients;
- 2% trusted labels;
- Dirichlet alpha 0.1, 0.5, 1.0;
- five partition seeds each;
- all 15 partitions produce 47.59% centralized and federated accuracy;
- maximum parameter discrepancy about `6.1e-11` from summation order.

## Claim 4 — partial participation is a separate non-IID boundary

Gate 32 mean full/partial-participation results:

| alpha | 100% | 80% | 50% |
|---:|---:|---:|---:|
| 0.1 | 47.59% | 46.78% | 45.30% |
| 0.5 | 47.59% | 47.43% | 47.07% |
| 1.0 | 47.59% | 47.48% | 47.20% |

The worst sampled 50%-participation condition reached 41.59%. LPA does not make
missing non-IID clients harmless.

## Claim 5 — external methods occupy a different utility/robustness point

Gate 33's matched CPU checkpoint shows FairMean retaining substantially more
raw-label accuracy than LPA under a two-client attack when many labels remain
clean, while FairMean/FedAvg decline as the attacked share grows and LPA remains
flat.

This supports a tradeoff claim, not an overall superiority claim.

## Claim 6 — LPA 2.0 default: higher utility, same invariance

Gate 34, full CIFAR-10 (50,000 train / 10,000 test), Gate-31 trusted seeds,
five seeds per budget:

| Trusted fraction | v1.0 student | 2.0 teacher | 2.0 student | Gain vs v1.0 | Gain vs 2.0 teacher |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 38.57% | 46.51% | 49.63% | +11.06 pt | +3.12 pt |
| 1% | 44.12% | 52.68% | 55.90% | +11.78 pt | +3.22 pt |
| 2% | 47.61% | 58.43% | 61.35% | +13.73 pt | +2.92 pt |
| 5% | 51.48% | 65.36% | 66.41% | +14.93 pt | +1.05 pt |

- Every gain holds in 5/5 seeds at every budget (paired t-test p < 0.01 against
  the 2.0 teacher, p < 1e-4 against the v1.0 student).
- `sentinel_target_max_diff = 0` and `sentinel_student_weight_max_diff = 0` in
  all 20 runs.
- The 5% utility crossover of Claim 2 does not occur: the 2.0 student beats a
  trusted-only linear ridge on the same features by 8.0 points at 5%.
- Exact federated aggregation (Claim 3) holds with trusted-row weights: a
  10-client Dirichlet(0.1) partition matched centralized training to `2.6e-11`.
- Hyperparameters were selected on a 5,000-image dev split held out of the
  training set. The test set was evaluated once, for the locked configuration.

Not yet repeated for 2.0: the Gate-32 partial-participation study, the
Gate-31 trusted-label contamination stress test, and the Gate-33 external
baselines.

## Nonclaims

LPA does not establish:

- universal immunity to all forms of poisoning;
- state-of-the-art accuracy on CIFAR-10;
- superiority to FairMean, q-FFL, Co-teaching, DivideMix, or cleanlab in every
  regime;
- robustness to corrupted trusted labels or features;
- privacy of federated sufficient statistics;
- correctness of pseudo-labels on non-identifiable examples.

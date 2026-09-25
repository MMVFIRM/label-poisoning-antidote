# Validated claims and claim boundaries

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

## Nonclaims

LPA v1.0 does not establish:

- universal immunity to all forms of poisoning;
- state-of-the-art accuracy on CIFAR-10;
- superiority to FairMean, q-FFL, Co-teaching, DivideMix, or cleanlab in every
  regime;
- robustness to corrupted trusted labels or features;
- privacy of federated sufficient statistics;
- correctness of pseudo-labels on non-identifiable examples.

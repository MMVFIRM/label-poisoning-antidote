# Gate 34 — stronger label-free features

Status: **adopted as the LPA 2.0 default architecture** (`architecture="kmeans"`).
The frozen v1.0 architecture remains available as `LPAConfig.v1()`, and every
v1.0 claim still applies to it. Gate 34 asked whether the v1.0 design could be
improved without weakening its central property.

## Summary

Replacing the student's representation with label-free Coates & Ng (2011)
k-means patch features, and adding the same features to the trusted teacher,
raises full CIFAR-10 accuracy by **+11 to +15 points at every trusted budget**.
It keeps the same invariance, and exact federated aggregation still holds.
It also changes two qualitative v1.0 findings:

1. **The student now beats its own teacher.** In v1.0 the student was 1–6 points
   *below* the trusted-only kernel teacher at every budget, so distillation onto
   the untrusted pool cost accuracy compared with deploying the teacher itself.
   In Gate 34 the student beats its teacher in all 20 of 20 runs.
2. **The 5% crossover is gone.** v1.0 lost to trusted-only learning at 5%
   trusted labels. The Gate-34 student beats the trusted-only linear model on
   the same features by about 8 points at 5%.

## What changed

```text
image x
  ├─ v1.0 HOG-576 + RGB-48 views ─────────────────────┐   (unchanged)
  └─ k-means patch features (6400-D, images only) ────┤
                                                      v
        trusted-only teacher: 0.5 * RBF-KRR(k-means; gamma .5, ridge .1)
                            + 0.5 * v1.0 two-view HOG/color teacher
                            softmax at T = .05
                                                      |
        targets: one-hot on trusted rows, teacher soft targets elsewhere
                                                      v
        linear ridge student on k-means features (ridge .03, trusted rows x10)

untrusted supplied label y~ ──X──> no training path
```

Feature pipeline (Coates & Ng single-layer network, no labels anywhere):

- 6x6 RGB patches at stride 1; per-patch mean/variance normalization (+10 on
  the 0–255 scale);
- ZCA whitening (eps 0.1) and 1600 k-means centroids (15 Lloyd iterations),
  both fitted to 400,000 random patches from training **images**;
- triangle encoding `max(0, mean_k d_k - d_k)`, then sum-pooling over the 2x2
  image quadrants gives 6400-D;
- sqrt, standardization using train-pool statistics, row normalization.

The student is still a closed-form ridge regression on x-only features, so the
v1.0 invariance argument (section 5 of [`METHOD.md`](METHOD.md)) and the
federated normal equations in [`FEDERATED.md`](FEDERATED.md) carry over without
modification.

## Model selection

All hyperparameters were chosen on a 5,000-example **dev split held out of the
50,000 training images** (fixed permutation seed 777), with trusted sets drawn
from the remaining 45,000 and seeds 31001–31003. The official test set was used
exactly once, for the locked configuration below.

| Hyperparameter | Grid searched on dev | Chosen |
|---|---|---|
| k-means teacher gamma | 0.5, 1, 2 | 0.5 |
| k-means teacher ridge | 0.01, 0.1 | 0.1 |
| k-means teacher weight in blend | 0.5, 0.65, 0.75, 1.0 | 0.5 |
| target temperature | 0.05, 0.1, 0.2 | 0.05 |
| student ridge | 0.03, 0.1, 0.3, 1 | 0.03 |
| trusted-row weight | 1, 10, 30 | 10 |

## Full-scale results (official protocol)

50,000 train / 10,000 official test, Gate-31 trusted seeds 31001–31005, five
seeds per budget. Produced by
[`experiments/reference/gate34_full_cifar_harness.py`](../experiments/reference/gate34_full_cifar_harness.py);
raw rows in `benchmarks/results/gate34_results.csv`, summary in
`benchmarks/results/gate34_summary.csv`.

| Trusted | v1.0 student | v1.0 teacher | G34 trusted-only linear | G34 teacher | **G34 student** | Gain vs v1.0 student |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5% | 38.57% | 39.52% | 40.73% | 46.51% | **49.63%** | +11.06 pt (5/5) |
| 1% | 44.12% | 45.68% | 45.56% | 52.68% | **55.90%** | +11.78 pt (5/5) |
| 2% | 47.61% | 51.44% | 50.77% | 58.43% | **61.35%** | +13.73 pt (5/5) |
| 5% | 51.48% | 57.01% | 58.39% | 65.36% | **66.41%** | +14.93 pt (5/5) |

Student gain over its own Gate-34 teacher: +3.12, +3.22, +2.92, +1.05 points
(5/5 seeds each; paired t-test p = 9e-5, 3e-4, 4e-4, 6e-3).

All-label reference with the same 6400-D linear ridge: **76.49%** (v1.0: 58.28%).

Invariance and federated checks, measured in the same run:

- `sentinel_target_max_diff = 0` in all 20 runs;
- `sentinel_student_weight_max_diff = 0` in all 20 runs, where every untrusted
  label is replaced by a nonnumeric sentinel and the final student is refit;
- 10-client Dirichlet(0.1) partition with trusted-row weighting: maximum
  federated/centralized parameter difference `2.6e-11`, 100% prediction
  agreement.

### Cross-check against the bundled v1.0 results

The harness recomputes the v1.0 teacher and student in the same run. The v1.0
student matches `gate31_full_results.csv` exactly in all 20 conditions. The v1.0
teacher matches in 19 of 20. In the remaining condition (100/class, seed 31005)
one test image flips (50.54% vs 50.55%), consistent with a solver or chunk-order
float difference on a near-tie.

The CIFAR-10 copy used here came from a different distribution channel than the
original release run. Reproducing Gate 31 to four decimals therefore also checks
the reference harness on independently obtained data.

## Negative results from the same round

Tested on the dev split and rejected:

- **More random landmarks for the v1.0 student** (256 → 1024 with the joint
  gamma-0.5 kernel): at most +1 pt.
- **Teacher-kernel landmark student on v1.0 features** (2048 x-only landmarks
  with the teacher's own HOG gamma 4 / color gamma 0.25 kernels, trusted rows
  x10): about +5 pts over the v1.0 student at 5% and roughly level with the
  teacher, but never clearly above it. This confirmed that the HOG/color
  representation is the bottleneck: at 5% the v1.0 teacher is within about
  1 pt of the 58% all-label ceiling.
- **Iterated self-training** (refit on the student's own sharpened
  predictions, T = 0.02 or 0.05): accuracy fell every round (for example
  39.7% → 34.6% at 25/class after three rounds). This is consistent with the
  Gate-27/28 findings that reshaping pseudo-targets does not help.

## Costs and boundaries

- **Compute.** Feature extraction takes about 7 minutes for 60,000 images on
  4 CPU cores (threaded NumPy), compared with about 25 seconds for HOG/color.
  Each full condition then takes 13–32 seconds.
- **Model size.** The teacher now stores 6400-D anchors for every trusted
  example. The student is 6400 x 10 weights, plus a 1600 x 108 dictionary and
  a 108 x 108 whitening matrix.
- **Federated deployment.** The dictionary, whitening, and standardization
  statistics must be fitted once and shared with every client, like the v1.0
  color statistics. They are label-free, but they are computed from images.
  A deployment that cannot pool images should fit them on public or
  trusted-only images; this variant was not measured.
- **Threat model unchanged.** Like v1.0, this is invariance to the untrusted
  label field only. Poisoned *images* can influence the label-free dictionary
  and standardization, as they already influenced the v1.0 color statistics and
  landmarks. This round did not measure that.
- **Library integration.** LPA 2.0 implements this path as the default
  `LabelPoisoningAntidote()` (`KMeansPatchFeatureExtractor`,
  `BlendedKernelTeacher`, `LinearRidgeStudent`). `LPAConfig.v1()` stays
  bit-identical to 1.0.0, and model files saved by 1.0.0 load unchanged.
  `experiments/reference/gate34_library_check.py` runs all 20 conditions
  through the public API of the installed 2.0.0 package. Student and teacher
  test accuracy match this harness exactly in 20 of 20 conditions
  (`benchmarks/results/gate34_library_check.csv`).

## Reproduce

```bash
# 1. v1.0 feature cache (also reproduces Gate 31)
python experiments/reference/gate31_full_cifar_harness.py \
  --data-dir ./cifar-10-batches-bin --feature-cache ./gate31_features.npz

# 2. Gate 34 (builds and caches k-means features on first run)
python experiments/reference/gate34_full_cifar_harness.py \
  --data-dir ./cifar-10-batches-bin --gate31-cache ./gate31_features.npz \
  --kmeans-cache ./gate34_kmeans_1600.npz --output gate34_results.csv

# 3. Summary + cross-check against bundled Gate 31
python experiments/reference/gate34_summarize.py gate34_results.csv

# 4. The installed library through its public API
python experiments/reference/gate34_library_check.py --data-dir ./cifar-10-batches-bin
```

# Bundled benchmark evidence

This directory contains the preserved result tables used to qualify LPA v1.0
release. The raw Gate-31 and Gate-32 CSVs were produced by the successful full-data
harness runs and copied into the release without editing.

## Gate 31

`gate31_full_results.csv` contains 20 full CIFAR-10 runs:

- 50,000 training examples;
- 10,000 official-test examples;
- trusted fractions 0.5%, 1%, 2%, 5%;
- five seeds per budget.

## Gate 32

`gate32_full_results.csv` contains 15 full federated/non-IID runs:

- 50,000 training examples;
- 10,000 official-test examples;
- 10 clients;
- 2% trusted labels;
- Dirichlet alpha 0.1, 0.5, 1.0;
- five partition seeds per alpha.

## Gate 33

The Gate-33 files contain a smaller model-capacity-matched method-level
comparison with FairMean/FedAvg/q-FFL plus an attack-fraction crossover sweep.
This is explicitly a **partial external-baseline qualification**.

`fairmean_published_cifar_reference_NOT_MATCHED.csv` records published FairMean
reference numbers for context only. They are not numerically comparable to the
matched fixed-feature checkpoint.

## Gate 34 (post-release research)

`gate34_results.csv` and `gate34_summary.csv` come from
`experiments/research/gate34_kmeans_harness.py`. The run uses the Gate-31
protocol (50,000 train / 10,000 test, same trusted seeds) and evaluates a
research candidate with label-free k-means patch features. It is not part of
the v1.0 qualification record and is not covered by `MANIFEST.json`. See
[`../docs/GATE34_RESEARCH.md`](../docs/GATE34_RESEARCH.md).

## Integrity

The root `MANIFEST.json` records SHA-256 hashes for every file in the release archive.

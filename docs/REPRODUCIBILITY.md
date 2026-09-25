# Reproducibility

## Core tests

```bash
python -m pip install -e '.[test]'
python -m pytest
lpa audit
lpa federated-audit
```

## Gate 31 full CIFAR

Download/extract the standard CIFAR-10 binary distribution so the directory
contains:

```text
data_batch_1.bin ... data_batch_5.bin
test_batch.bin
```

Then run:

```bash
python experiments/reference/gate31_full_cifar_harness.py \
  --data-dir ./cifar-10-batches-bin \
  --trusted-per-class 25 50 100 250 \
  --seeds 31001 31002 31003 31004 31005 \
  --feature-cache gate31_features.npz \
  --output gate31_reproduction.csv
```

## Gate 32 full federated/non-IID

Reuse the feature cache:

```bash
python experiments/reference/gate32_full_federated_harness.py \
  --data-dir ./cifar-10-batches-bin \
  --feature-cache gate31_features.npz \
  --trusted-per-class 100 \
  --trusted-seed 31001 \
  --alphas 0.1 0.5 1.0 \
  --partition-seeds 32301 32302 32303 32304 32305 \
  --output gate32_reproduction.csv
```

## Gate 34 full CIFAR (LPA 2.0 default)

Reuses the Gate-31 feature cache and builds the k-means feature cache on its
first run (about 7 minutes on 4 cores; the cache is about 1.5 GB):

```bash
python experiments/reference/gate34_full_cifar_harness.py \
  --data-dir ./cifar-10-batches-bin \
  --gate31-cache gate31_features.npz \
  --kmeans-cache gate34_kmeans_1600.npz \
  --output gate34_reproduction.csv
python experiments/reference/gate34_summarize.py gate34_reproduction.csv
```

The summarizer also checks the run's recomputed v1.0 columns against
`benchmarks/results/gate31_full_results.csv`.

To confirm that the installed `lpa` package reproduces the harness through its
public API:

```bash
python experiments/reference/gate34_library_check.py --data-dir ./cifar-10-batches-bin
```

## External baseline sanity

The default external script is deliberately one seed because the full CNN
matrix can be expensive:

```bash
bash experiments/external/run_official_fairmean_sanity.sh
```

The original 30-run matrix remains available as
`run_official_fairmean_full_optional.sh` for a paper/reviewer request.

## Result provenance

The exact successful Gate-31, Gate-32, and Gate-34 CSVs are preserved under
`benchmarks/results/`. SHA-256 hashes are recorded in `MANIFEST.json` at the
repository root when the release archive is built.

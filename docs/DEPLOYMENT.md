# Deployment guidance

LPA should be deployed as an explicit trust-boundary system, not as an
automatic label-cleaning heuristic.

## Data contract

Maintain separate inputs for:

- feature/example data;
- trusted example indices and trusted labels;
- any raw untrusted label field used elsewhere in the product.

Do not merge the untrusted label field back into LPA target construction,
hyperparameter tuning, stopping criteria, landmark selection, or feature
fitting.

## Choosing an architecture

The 2.0 default (`architecture="kmeans"`) is the more accurate choice on
CIFAR-like images: +11 to +15 points over v1.0 on full CIFAR-10. It costs more:

| | 2.0 default | v1.0 (`LPAConfig.v1()`) |
|---|---|---|
| Feature extraction, 60k CIFAR images, 4 cores | about 7 min | about 25 s |
| Feature memory, 50k images | about 2.6 GB (float64) | about 0.25 GB |
| Student parameters | 6400 x 10 | 880 x 10 |
| Federated payload per client (float32) | 78.4 MiB | 1.5 MiB |

Use v1.0 when compute, memory, or federated bandwidth is the binding
constraint.

## High-trust crossover

The v1.0 student lost to trusted-only learning at 5% trusted labels on full
CIFAR-10. The 2.0 default student did not: it beat a trusted-only linear model on
the same features by 8 points at 5%, and it beat its own trusted-only teacher at
every tested budget.

This does not generalize automatically. For a new domain, compare the
LPA student against trusted-only baselines, including the teacher itself, using
**trusted validation data**.

## Monitoring

Recommended production monitors include:

- trusted-set size and class coverage;
- teacher held-out performance on trusted validation data;
- disagreement between teacher and deployed student;
- fraction and identity of participating federated clients;
- client feature-distribution shift;
- worst-client or subgroup performance where legally/operationally appropriate;
- model and target hashes for reproducibility.

## Serialization

`LabelPoisoningAntidote.save()` writes:

- numeric model arrays in `model.npz` with `allow_pickle=False` on load;
- `metadata.json` (format 3) containing the architecture, the full config,
  target hash, student-weight hash, and a SHA-256 digest of every saved array.

`load()` rejects a model if any array is missing, added, or does not match its
recorded digest. Those digests live in the same directory as the model, so they
detect corruption, not deliberate tampering. For tamper evidence, record
`LabelPoisoningAntidote.fingerprint(path)` (the SHA-256 of `metadata.json`) in
a location the model directory cannot modify, and load with
`LabelPoisoningAntidote.load(path, expected_fingerprint=...)`.

Format-2 models written by 1.0.0 load as the v1.0 architecture and predict
identically. Format-1 models written by 1.0.0rc1 still load, with a warning that
only the student weights were checked. Re-save them to upgrade.

### Model files contain training data

A saved model is not a label-free summary. It contains:

- the trusted examples' feature vectors (teacher anchors; 6400-D k-means
  features plus the HOG and color views for the 2.0 default);
- for 2.0, a 1600-centroid patch dictionary, whitening, and feature statistics
  learned from the training images;
- for v1.0, 256 training examples' joint feature vectors (student landmarks);
- by default, trusted indices and trusted labels.

2.0 model files are larger. At 5% trusted CIFAR-10, the teacher anchors alone
are 2,500 x 6400 values.

Apply the same access controls to model files as to the training features.
`save(path, include_trusted_labels=False)` omits the trusted indices and labels,
which are not needed for prediction.

### Input validation

Training and prediction reject NaN or infinite features, non-integer labels or
indices (instead of silently truncating them), and feature widths that do not
match the fitted model. Federated aggregation rejects non-finite or asymmetric
client Gram matrices. These checks catch accidents, not adversaries: a malicious
client can still submit finite, well-formed statistics (see
[`THREAT_MODEL.md`](THREAT_MODEL.md)).

## Federated privacy

The Gate-32 result is a robustness result, not a privacy guarantee. Raw `A_i`
and `B_i` may reveal information about client data. LPA does not include
secure aggregation; the sufficient-statistic form works with any
additive secure-aggregation protocol, and deployments where that exposure is
unacceptable should use one.

## Release policy

LPA is production-qualified for its narrow causal claim, not for every
environment. Validate utility against a trusted-only baseline in each new
domain, and treat the bundled CIFAR-10 numbers as awaiting independent
reproduction. The 2.0 default has centralized full-scale qualification (Gate 34).
Its federated partial-participation behavior has not been re-measured.

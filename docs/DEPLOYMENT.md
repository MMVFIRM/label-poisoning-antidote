# Deployment guidance

LPA v1.0 should be deployed as an explicit trust-boundary system, not as an
automatic label-cleaning heuristic.

## Data contract

Maintain separate inputs for:

- feature/example data;
- trusted example indices and trusted labels;
- any raw untrusted label field used elsewhere in the product.

Do not merge the untrusted label field back into LPA target construction,
hyperparameter tuning, stopping criteria, or landmark selection.

## High-trust crossover

On full CIFAR-10, LPA improved over trusted-only learning at 0.5-2% trusted
labels but not at 5% for the frozen student.

Do not hardcode `5%` as a universal switch point. For a new domain, compare the
LPA student against a trusted-only baseline using **trusted validation data**.

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
- `metadata.json` (format 2) containing the frozen config, target hash,
  student-weight hash, and a SHA-256 digest of every saved array.

`load()` rejects a model if any array is missing, added, or does not match its
recorded digest. Those digests live in the same directory as the model, so they
detect corruption, not deliberate tampering. For tamper evidence, record
`LabelPoisoningAntidote.fingerprint(path)` (the SHA-256 of `metadata.json`) in
a location the model directory cannot modify, and load with
`LabelPoisoningAntidote.load(path, expected_fingerprint=...)`.

Format-1 models written by 1.0.0rc1 still load, with a warning that only the
student weights were checked. Re-save them to upgrade.

### Model files contain training data

A saved model is not a label-free summary. It contains:

- the trusted examples' feature vectors (teacher anchors);
- 256 training examples' joint feature vectors (student landmarks);
- by default, trusted indices and trusted labels.

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
and `B_i` may reveal information about client data. LPA v1.0 does not include
secure aggregation; the sufficient-statistic form works with any
additive secure-aggregation protocol, and deployments where that exposure is
unacceptable should use one.

## Release policy

v1.0 is production-qualified for its narrow causal claim, not for every
environment. Validate utility against a trusted-only baseline in each new
domain, and treat the bundled CIFAR-10 numbers as awaiting independent
reproduction.

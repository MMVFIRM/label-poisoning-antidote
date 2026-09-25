# Security policy

## Supported security claim

LPA v1.0 is designed for a **label-only threat model** in which an explicitly
untrusted label field may be replaced arbitrarily while the feature data, trusted
set membership, trusted labels, and training code remain unchanged.

The core training API does not accept untrusted labels.

## Out of scope

LPA v1.0 does not claim robustness to compromised trusted labels, feature or input
poisoning, sample injection/removal, malicious executable client updates,
availability attacks, privacy attacks on sufficient statistics, or arbitrary
Byzantine behavior.

## Reporting

Please report suspected violations with a minimal reproducible test. For
issues that should not be disclosed publicly, use GitHub's private
vulnerability reporting ("Report a vulnerability" under the repository's
Security tab). Do not include private production data in a public issue.

## Model files

Saved models contain training-feature vectors; see
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md#model-files-contain-training-data).
The in-directory array digests detect corruption only; use
`expected_fingerprint` on load for tamper evidence.

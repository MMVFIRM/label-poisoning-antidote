# Federated LPA

## Exact sufficient-statistic form

Once the trusted teacher and the shared feature basis are fixed, each client `i`
computes:

```text
Phi_i  = LPA student features for its local x values
Q_i    = trusted-teacher targets for those examples
D_i    = diag(row weights): 10 on trusted rows, 1 elsewhere (2.0 default);
         the identity for the v1.0 architecture
A_i    = Phi_i^T D_i Phi_i
B_i    = Phi_i^T D_i Q_i
```

The shared basis is the 256 landmarks and color statistics for v1.0. For the
2.0 default it is the k-means patch dictionary, whitening, feature
standardization, and color statistics. All of these are label-free, but they
are computed from images, so they must be fitted once and distributed to every
client. `client_sufficient_statistics(phi, targets, weights)` computes the
weighted statistics; `trusted_row_weights()` builds the weights.

The server computes:

```text
A = lambda I + sum_i A_i
B = sum_i B_i
W = A^-1 B.
```

With full participation, this is exactly the centralized ridge normal equation.
Client partitioning changes only how terms are grouped.

## Poisoning invariant

For an untrusted-label-only attack:

```text
Delta Phi_i = 0
Delta Q_i   = 0
Delta A_i   = 0
Delta B_i   = 0
Delta W     = 0.
```

Gate 32 measured zero target and final-weight change under the attacked client
label mutation. The row weights depend only on trusted-set membership, so
`Delta D_i = 0` as well. Gate 34 measured zero final-weight change for the 2.0
student and a `2.6e-11` federated/centralized difference with a 10-client
Dirichlet(0.1) partition.

## Communication

With 10 classes, a client communicates one symmetric `A_i` and one `B_i`.

| Student | `A_i` unique values | `B_i` values | float32 | float64 |
|---|---:|---:|---:|---:|
| v1.0, 880-D | 387,640 | 8,800 | 1.51 MiB | 3.02 MiB |
| 2.0, 6400-D | 20,483,200 | 64,000 | 78.4 MiB | 156.8 MiB |

These are payloads per client before protocol or compression overhead. The
2.0 payload is about 50x larger. That is usually acceptable for a few
cross-silo clients, but a deployment with many clients or constrained links
should budget for it, or use the v1.0 architecture.

## Privacy

Sufficient statistics are **not inherently private**. If the server should not
observe individual client statistics, use secure aggregation and evaluate
whether additional privacy protection is required.

## Partial participation

Exact centralized equivalence applies to the set of examples that actually
participate. If clients are missing, the training sample itself changes. Under
severe non-IID skew this produced meaningful utility loss in Gate 32.

Production systems should monitor participation patterns, not only the nominal
client count.

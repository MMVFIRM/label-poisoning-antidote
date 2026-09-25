# Federated LPA

## Exact sufficient-statistic form

Once the trusted teacher and shared landmark basis are fixed, each client `i`
computes:

```text
Phi_i  = LPA student features for its local x values
Q_i    = trusted-teacher targets for those examples
A_i    = Phi_i^T Phi_i
B_i    = Phi_i^T Q_i
```

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
label mutation.

## Communication

For the validated 880-D student and 10 classes, a client needs to communicate
one symmetric `A_i` and one `B_i`.

Unique values:

```text
A_i: 880*881/2 = 387,640
B_i: 880*10    =   8,800
```

Approximate payload before protocol/compression overhead:

```text
float32: 1.51 MiB/client
float64: 3.02 MiB/client
```

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

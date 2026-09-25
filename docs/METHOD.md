# Method specification

## 1. Goal

LPA separates a trusted semantic root from an explicitly untrusted label field.
LPA v1.0 does not attempt to decide which untrusted labels are correct. It
removes that variable from the default learning objective.

## 2. Validated feature path

For a 32x32 RGB image `x`, LPA computes two label-independent
views:

- `h(x)`: 576-D fixed HOG-style gradient descriptor;
- `c(x)`: 48-D spatial RGB block-mean descriptor.

After color standardization and row normalization, the student joint feature is

```text
z(x) = [h(x), c(x)] / sqrt(2).
```

No label is used in feature extraction or standardization.

## 3. Trusted-only teacher

Only trusted examples `(x_i, y_i)` enter the teacher fit.

For each view `v` the teacher uses RBF kernel ridge regression:

```text
K_v(i,j) = exp(-gamma_v ||v_i - v_j||^2)
alpha_v  = (K_v + lambda_T I)^-1 Y_T
```

with frozen defaults:

```text
gamma_HOG   = 4.0
gamma_color = 0.25
HOG weight  = 0.75
teacher ridge = 0.1
temperature   = 0.1
```

Raw teacher score:

```text
s(x) = .75 K_HOG(x,T) alpha_HOG + .25 K_color(x,T) alpha_color
```

Teacher distribution:

```text
p_T(x) = softmax(s(x) / .1).
```

## 4. Invariant target construction

For trusted examples:

```text
q_i = one_hot(y_i).
```

For every other example:

```text
q_i = p_T(x_i).
```

There is no expression containing the supplied untrusted label.

Therefore, for arbitrary assignments `Y_U` and `Y'_U`:

```text
Q(X, Y_T, Y_U) = Q(X, Y_T, Y'_U).
```

## 5. Student

The student chooses 256 training examples using a fixed random seed without
consulting labels.

For landmark `l_j`:

```text
k_j(x) = exp(-0.5 ||z(x) - z(l_j)||^2).
```

The final student feature is

```text
Phi(x) = [z(x), k_1(x), ..., k_256(x)] / sqrt(2).
```

The ridge objective is

```text
min_W ||Phi W - Q||_F^2 + ||W||_F^2.
```

The closed-form solution is

```text
W = (Phi^T Phi + I)^-1 Phi^T Q.
```

Since both `Phi` and `Q` are independent of `Y_U`, the fitted `W` is also
independent of `Y_U` for a deterministic solver.

## 6. Why this is different from label correction

LPA does not establish that `p_T(x)` equals the latent true label. Earlier gates
identified an information-theoretic boundary: if trusted evidence and features
cannot distinguish two latent label assignments, no label-blind method can
recover which assignment is true.

The release claim is therefore **invariance**, not oracle label recovery.

## 7. Auxiliary research components not in the default core

The research program also investigated conformal candidate sets, temporal
evidence, confidence routing, calibration, graph diffusion, geometric poison
detection, and certificate-aware training. They remain useful scientific
context, but repeated gates showed that several of these interventions reduced
utility or failed to justify their complexity.

The v1.0 default path intentionally excludes them.

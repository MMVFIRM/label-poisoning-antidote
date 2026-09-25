# Method specification

LPA 2.0 has two architectures that share the same invariance argument:

- `kmeans` (default since 2.0, Gate 34): sections 1-6 and 8;
- `landmark` (frozen v1.0, `LPAConfig.v1()`): sections 1-7.

Sections 2-5 describe v1.0 in full; section 8 gives the 2.0 changes.

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

The default paths intentionally exclude them.

## 8. LPA 2.0 default architecture (Gate 34)

### 8.1 Label-free k-means patch features

Following Coates & Ng (2011), for image pixels on the 0-255 scale:

1. sample 400,000 random 6x6 RGB patches from training **images** (seed 0);
2. normalize each patch: `p <- (p - mean(p)) / sqrt(var(p) + 10)`;
3. fit ZCA whitening `W_zca = V diag(1/sqrt(e + 0.1)) V^T` to those patches;
4. run 15 Lloyd iterations of k-means with `K = 1600` centroids `c_k`;
5. for every stride-1 patch `p` of an image, let `d_k = ||W_zca (p - mu) - c_k||`
   and encode `f_k = max(0, mean_j d_j - d_k)` (triangle activation);
6. sum-pool `f` over the four 13x13 image quadrants, giving `4K = 6400` values;
7. take `u = sqrt(pooled)`, standardize with the training-pool mean and
   standard deviation (+0.001), and row-normalize to obtain `z(x)`.

No label enters any step. All fitted quantities (patch mean, whitening, centroids,
standardization) are functions of the training images alone.

### 8.2 Blended trusted-only teacher

With the v1.0 two-view raw score `s_v1(x)` from section 3 and a joint-feature
kernel ridge fitted on trusted examples only,

```text
K_z(i,j) = exp(-0.5 ||z_i - z_j||^2)
alpha_z  = (K_z + 0.1 I)^-1 Y_T
s(x)     = 0.5 K_z(x,T) alpha_z + 0.5 s_v1(x)
p_T(x)   = softmax(s(x) / 0.05).
```

### 8.3 Targets

As in section 4: `q_i = one_hot(y_i)` on trusted examples and `q_i = p_T(x_i)`
elsewhere. No expression reads the untrusted label, so
`Q(X, Y_T, Y_U) = Q(X, Y_T, Y'_U)`.

### 8.4 Linear ridge student with trusted-row weights

With row weights `w_i = 10` on trusted examples and `w_i = 1` elsewhere (a
function of trusted-set membership only),

```text
min_W  sum_i w_i ||z(x_i) W - q_i||^2 + 0.03 ||W||_F^2
W = (Z^T D Z + 0.03 I)^-1 Z^T D Q,   D = diag(w).
```

`Z`, `D`, and `Q` are all independent of `Y_U`, so `W` is too. The feature map is
the identity, so the federated form of section 5 applies with
`A_i = Z_i^T D_i Z_i` and `B_i = Z_i^T D_i Q_i`.

### 8.5 Why the student can now beat the teacher

In v1.0 the student's 880-D landmark features could only approximate the
teacher's kernel function, so distillation lost 1-6 points to the teacher. The
2.0 student is linear in a 6400-D representation that is richer than the
teacher's trusted-anchor kernel expansion, and it is fitted on all 50,000
x-values. It therefore uses the untrusted pool's inputs (never its labels) to
generalize beyond the teacher: +1 to +3 points over the teacher on full
CIFAR-10.

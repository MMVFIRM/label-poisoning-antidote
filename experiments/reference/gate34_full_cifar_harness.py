#!/usr/bin/env python3
"""
LPA Gate 34 — label-free k-means patch features + direct-feature ridge student.

Research candidate, NOT the frozen v1.0 architecture. It keeps every LPA
invariance rule: nothing below reads an untrusted label.

  image x
    -> (a) v1.0 576-D HOG + 48-D spatial RGB views          (unchanged)
    -> (b) Coates & Ng (2011) single-layer k-means features:
           6x6 patches, per-patch normalization, ZCA whitening,
           K=1600 centroids learned from training *images only*,
           triangle encoding, 2x2 quadrant sum-pooling -> 6400-D,
           then sqrt, train-pool standardization, row normalization
    -> trusted-only teacher:
           0.5 * RBF kernel ridge on (b)  (gamma .5, ridge .1)
         + 0.5 * v1.0 two-view HOG/color teacher raw score
       softmax at temperature .05
    -> targets: one-hot on trusted rows, teacher soft targets elsewhere
    -> linear ridge student on (b) (ridge .03, trusted rows weighted x10)

The student is still a ridge regression on x-only features, so the exact
federated sufficient-statistic form of v1.0 carries over unchanged (checked
below). The k-means dictionary, whitening, and standardization are
label-free but must be shared by all clients, like the v1.0 color statistics.

Hyperparameters were chosen on a 5,000-example dev split held out of the
50,000-example training set (seeds 31001-31003), never on the test set.

Usage:
  python gate34_full_cifar_harness.py --data-dir ./cifar-10-batches-bin \
      --gate31-cache ./gate31_features.npz --kmeans-cache ./gate34_kmeans_1600.npz

Requires numpy and scipy only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NCLASS = 10

# v1.0 frozen constants (Gate 31).
GAMMA_HOG, GAMMA_COLOR, HOG_WEIGHT, TEACHER_RIDGE, TEACHER_TEMP = 4.0, 0.25, 0.75, 0.1, 0.1
LANDMARK_COUNT, LANDMARK_SEED, LANDMARK_GAMMA, STUDENT_RIDGE = 256, 29002, 0.5, 1.0

# Gate 34 constants (selected on the dev split).
KM_K, KM_PATCH, KM_SAMPLES, KM_ITERS, KM_SEED = 1600, 6, 400_000, 15, 0
KM_NORM_EPS, KM_ZCA_EPS = 10.0, 0.1
G34_KGAMMA, G34_KRIDGE, G34_KWEIGHT, G34_TEMP = 0.5, 0.1, 0.5, 0.05
G34_STUDENT_RIDGE, G34_TRUSTED_WEIGHT = 0.03, 10.0


# ---------------------------------------------------------------- data
def read_bin(path: Path):
    rec = np.fromfile(path, dtype=np.uint8).reshape(-1, 3073)
    return rec[:, 1:], rec[:, 0].astype(np.int64)


def load_cifar(d: Path):
    xs, ys = zip(*(read_bin(d / f"data_batch_{i}.bin") for i in range(1, 6)))
    xt, yt = read_bin(d / "test_batch.bin")
    return np.concatenate(xs), np.concatenate(ys), xt, yt


def nhwc255(raw_u8):
    return raw_u8.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)


# ---------------------------------------------------------------- k-means features
def _patch_norm(p):
    p = p - p.mean(-1, keepdims=True)
    return p / np.sqrt(p.var(-1, keepdims=True) + KM_NORM_EPS)


def fit_kmeans_dictionary(x):
    """Learn ZCA + centroids from random training patches. Uses images only."""
    rng = np.random.default_rng(KM_SEED)
    n, P = KM_SAMPLES, KM_PATCH
    ni, r, c = rng.integers(0, len(x), n), rng.integers(0, 33 - P, n), rng.integers(0, 33 - P, n)
    s = np.stack([x[ni, r + i, c + j] for i in range(P) for j in range(P)], 1).reshape(n, -1)
    s = _patch_norm(s).astype(np.float64)
    mu = s.mean(0)
    ev, v = np.linalg.eigh(np.cov(s - mu, rowvar=False))
    zca = (v @ np.diag(1 / np.sqrt(ev + KM_ZCA_EPS)) @ v.T).astype(np.float32)
    sw = ((s - mu) @ zca).astype(np.float32)
    cen = sw[rng.choice(n, KM_K, replace=False)].copy()
    for _ in range(KM_ITERS):
        cc = (cen * cen).sum(1)
        assign = np.concatenate([(cc[None] - 2 * sw[a:a + 50000] @ cen.T).argmin(1) for a in range(0, n, 50000)])
        cnt = np.bincount(assign, minlength=KM_K)
        new = np.zeros_like(cen)
        np.add.at(new, assign, sw)
        empty = cnt == 0
        new[~empty] /= cnt[~empty, None]
        new[empty] = sw[rng.choice(n, int(empty.sum()), replace=False)]
        cen = new
    return mu.astype(np.float32), zca, cen


def encode(x, mu, zca, cen, chunk=100, workers=None):
    P, K = KM_PATCH, len(cen)
    g = 33 - P
    half = g // 2
    cc = (cen * cen).sum(1)
    out = np.empty((len(x), 4 * K), np.float32)

    def job(a):
        im = x[a:a + chunk]
        p = sliding_window_view(im, (P, P), axis=(1, 2)).transpose(0, 1, 2, 4, 5, 3)
        p = _patch_norm(p.reshape(len(im), g, g, -1))
        pw = (p.reshape(-1, p.shape[-1]) - mu) @ zca
        d = np.sqrt(np.maximum((pw * pw).sum(1)[:, None] + cc[None] - 2 * pw @ cen.T, 0))
        f = np.maximum(0, d.mean(1, keepdims=True) - d).reshape(len(im), g, g, K)
        quads = (f[:, :half, :half], f[:, :half, half:], f[:, half:, :half], f[:, half:, half:])
        out[a:a + len(im)] = np.concatenate([q.sum((1, 2)) for q in quads], 1)

    with ThreadPoolExecutor(workers or os.cpu_count()) as ex:
        list(ex.map(job, range(0, len(x), chunk)))
    return out


def kmeans_features(x_u8, xt_u8, cache: Path | None):
    if cache and cache.exists():
        f = np.load(cache)
        return f["train"], f["test"]
    x, xt = nhwc255(x_u8), nhwc255(xt_u8)
    mu, zca, cen = fit_kmeans_dictionary(x)
    f, ft = encode(x, mu, zca, cen), encode(xt, mu, zca, cen)
    if cache:
        np.savez(cache, train=f, test=ft)
    return f, ft


def postprocess(f, ft):
    """sqrt -> standardize with train-pool statistics -> row-normalize. Label-free."""
    a, b = np.sqrt(f.astype(np.float64)), np.sqrt(ft.astype(np.float64))
    mu, sd = a.mean(0), a.std(0) + 1e-3
    a, b = (a - mu) / sd, (b - mu) / sd
    return a / np.linalg.norm(a, axis=1, keepdims=True), b / np.linalg.norm(b, axis=1, keepdims=True)


# ---------------------------------------------------------------- shared math
def sqdist(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0)


def rbf(a, b, g):
    return np.exp(-g * sqdist(a, b))


def onehot(y):
    return np.eye(NCLASS)[y]


def softmax_temp(r, t):
    z = r / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def acc(s, y):
    return float(np.mean(np.argmax(s, 1) == y))


def balanced_trusted(y, per_class, seed):
    rng = np.random.default_rng(seed)
    ids = []
    for c in range(NCLASS):
        ids.extend(rng.choice(np.flatnonzero(y == c), per_class, replace=False))
    return np.sort(np.asarray(ids, np.int64))


def chunked(fn, n, chunk=5000):
    return np.concatenate([fn(slice(s, s + chunk)) for s in range(0, n, chunk)])


def kernel_ridge(K, Y, lam):
    return np.linalg.solve(K + lam * np.eye(len(K)), Y)


def build_targets(raw_teacher, trusted, observed_labels):
    """Targets read the observed label field only at trusted indices."""
    q = softmax_temp(raw_teacher, G34_TEMP)
    q[trusted] = onehot(np.asarray(observed_labels[trusted], dtype=np.int64))
    return q


def weighted_ridge(A, X, Q, lam, trusted, wT):
    XT = X[trusted]
    lhs = A + lam * np.eye(len(A)) + (wT - 1) * XT.T @ XT
    rhs = X.T @ Q + (wT - 1) * XT.T @ Q[trusted]
    return np.linalg.solve(lhs, rhs)


# ---------------------------------------------------------------- one condition
def run_condition(g31, X, Xt, A, y, yt, per_class, seed):
    h, c, z, ht, ct, zt = g31
    t0 = time.time()
    trusted = balanced_trusted(y, per_class, seed)
    Y = onehot(y[trusted])
    n = len(y)

    # v1.0 teacher raw scores (identical to Gate 31).
    ah = kernel_ridge(rbf(h[trusted], h[trusted], GAMMA_HOG), Y, TEACHER_RIDGE)
    ac = kernel_ridge(rbf(c[trusted], c[trusted], GAMMA_COLOR), Y, TEACHER_RIDGE)

    def v1_raw(hq, cq):
        return HOG_WEIGHT * rbf(hq, h[trusted], GAMMA_HOG) @ ah + (1 - HOG_WEIGHT) * rbf(cq, c[trusted], GAMMA_COLOR) @ ac

    v1_tr = chunked(lambda s: v1_raw(h[s], c[s]), n)
    v1_te = chunked(lambda s: v1_raw(ht[s], ct[s]), len(yt))

    # v1.0 frozen student (for a same-run side-by-side).
    L = z[np.random.default_rng(LANDMARK_SEED).choice(n, LANDMARK_COUNT, replace=False)]

    def phi(zz):
        return np.concatenate([zz, rbf(zz, L, LANDMARK_GAMMA)], 1) / np.sqrt(2.0)

    Pz = chunked(lambda s: phi(z[s]), n)
    q1 = softmax_temp(v1_tr, TEACHER_TEMP)
    q1[trusted] = Y
    w1 = np.linalg.solve(Pz.T @ Pz + STUDENT_RIDGE * np.eye(Pz.shape[1]), Pz.T @ q1)
    v1_student = acc(chunked(lambda s: phi(zt[s]), len(yt)) @ w1, yt)

    # Gate 34 teacher.
    ak = kernel_ridge(rbf(X[trusted], X[trusted], G34_KGAMMA), Y, G34_KRIDGE)
    k_tr = chunked(lambda s: rbf(X[s], X[trusted], G34_KGAMMA) @ ak, n)
    k_te = chunked(lambda s: rbf(Xt[s], X[trusted], G34_KGAMMA) @ ak, len(yt))
    raw_tr = G34_KWEIGHT * k_tr + (1 - G34_KWEIGHT) * v1_tr
    raw_te = G34_KWEIGHT * k_te + (1 - G34_KWEIGHT) * v1_te

    # Gate 34 student, plus an explicit untrusted-label mutation audit on the
    # final parameters: every untrusted label replaced by a nonnumeric sentinel.
    observed = y.astype(object)
    q = build_targets(raw_tr, trusted, observed)
    W = weighted_ridge(A, X, q, G34_STUDENT_RIDGE, trusted, G34_TRUSTED_WEIGHT)
    mutated = observed.copy()
    mutated[np.setdiff1d(np.arange(n), trusted)] = "UNTRUSTED_DO_NOT_READ"
    q_m = build_targets(raw_tr, trusted, mutated)
    W_m = weighted_ridge(A, X, q_m, G34_STUDENT_RIDGE, trusted, G34_TRUSTED_WEIGHT)

    # Trusted-only baselines on the same 6400-D features.
    XT = X[trusted]
    w_tr = np.linalg.solve(XT.T @ XT + G34_STUDENT_RIDGE * np.eye(X.shape[1]), XT.T @ Y)

    return dict(
        trusted_per_class=per_class,
        trusted_total=len(trusted),
        trusted_fraction=len(trusted) / n,
        seed=seed,
        v1_teacher_test_accuracy=acc(v1_te, yt),
        v1_student_test_accuracy=v1_student,
        g34_kernel_teacher_only_test_accuracy=acc(k_te, yt),
        g34_teacher_test_accuracy=acc(raw_te, yt),
        g34_teacher_untrusted_accuracy=acc(np.delete(raw_tr, trusted, 0), np.delete(y, trusted)),
        g34_student_test_accuracy=acc(Xt @ W, yt),
        g34_trusted_only_linear_test_accuracy=acc(Xt @ w_tr, yt),
        sentinel_target_max_diff=float(np.max(np.abs(q_m - q))),
        sentinel_student_weight_max_diff=float(np.max(np.abs(W_m - W))),
        seconds=time.time() - t0,
    )


def federated_check(X, Xt, A, y, yt, seed=31001, per_class=100, clients=10, alpha=0.1):
    """Dirichlet non-IID partition; sum of client stats must equal centralized ridge."""
    rng = np.random.default_rng(seed)
    owner = np.empty(len(y), np.int64)
    for cl in range(NCLASS):
        ids = rng.permutation(np.flatnonzero(y == cl))
        cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(clients)))[:-1] * len(ids)).astype(int)
        for k, part in enumerate(np.split(ids, cuts)):
            owner[part] = k
    trusted = balanced_trusted(y, per_class, seed)
    # Any fixed target matrix exercises the algebra; use trusted one-hot + uniform.
    q = np.full((len(y), NCLASS), 1.0 / NCLASS)
    q[trusted] = onehot(y[trusted])
    wts = np.ones(len(y))
    wts[trusted] = G34_TRUSTED_WEIGHT
    Asum = np.zeros_like(A)
    Bsum = np.zeros((A.shape[0], NCLASS))
    for k in range(clients):
        m = owner == k
        Xk = X[m] * np.sqrt(wts[m])[:, None]
        Asum += Xk.T @ Xk
        Bsum += Xk.T @ (q[m] * np.sqrt(wts[m])[:, None])
    Wf = np.linalg.solve(Asum + G34_STUDENT_RIDGE * np.eye(len(A)), Bsum)
    Wc = weighted_ridge(A, X, q, G34_STUDENT_RIDGE, trusted, G34_TRUSTED_WEIGHT)
    return dict(
        fed_max_param_diff=float(np.max(np.abs(Wf - Wc))),
        fed_prediction_agreement=float(np.mean((Xt @ Wf).argmax(1) == (Xt @ Wc).argmax(1))),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--gate31-cache", type=Path, required=True,
                    help="feature cache written by gate31_full_cifar_harness.py")
    ap.add_argument("--kmeans-cache", type=Path, default=Path("gate34_kmeans_1600.npz"))
    ap.add_argument("--trusted-per-class", type=int, nargs="+", default=[25, 50, 100, 250])
    ap.add_argument("--seeds", type=int, nargs="+", default=[31001, 31002, 31003, 31004, 31005])
    ap.add_argument("--output", type=Path, default=Path("gate34_results.csv"))
    args = ap.parse_args()

    t0 = time.time()
    x, y, xt, yt = load_cifar(args.data_dir)
    f, ft = kmeans_features(x, xt, args.kmeans_cache)
    print("kmeans features", f.shape, ft.shape, f"{time.time() - t0:.0f}s", flush=True)
    X, Xt = postprocess(f, ft)
    del f, ft
    g = np.load(args.gate31_cache)
    g31 = tuple(g[k] for k in ("h_train", "c_train", "z_train", "h_test", "c_test", "z_test"))
    A = X.T @ X

    Wall = np.linalg.solve(A + G34_STUDENT_RIDGE * np.eye(len(A)), X.T @ onehot(y))
    reference = dict(g34_all_label_reference_accuracy=acc(Xt @ Wall, yt))
    reference.update(federated_check(X, Xt, A, y, yt))
    print(json.dumps(reference), flush=True)

    rows = []
    for k in args.trusted_per_class:
        for seed in args.seeds:
            r = run_condition(g31, X, Xt, A, y, yt, k, seed)
            r.update(reference)
            rows.append(r)
            print(json.dumps(r), flush=True)
    with args.output.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", args.output)


if __name__ == "__main__":
    main()

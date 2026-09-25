#!/usr/bin/env python3
"""
LPA Gate 31 — full CIFAR-10 production-qualification harness.

Frozen architecture:
  image x
    -> fixed 576-D HOG-style view + 48-D spatial RGB view
    -> trusted-only two-view kernel teacher
    -> Gate-26 direct soft pseudo-target
    -> 256 x-only random RBF landmarks (gamma=.5)
    -> ridge student

The untrusted supplied label is never read when constructing:
  - HOG/color features
  - trusted teacher
  - pseudo-targets for untrusted examples
  - landmark selection
  - landmark features
  - student fit

Requirements:
  numpy
  scipy

Expected data directory:
  cifar-10-batches-bin/
    data_batch_1.bin ... data_batch_5.bin
    test_batch.bin

Example:
  python gate31_full_cifar_harness.py \
      --data-dir ./cifar-10-batches-bin \
      --trusted-per-class 25 50 100 250 \
      --seeds 31001 31002 31003 31004 31005 \
      --output gate31_full_results.csv

On the full 50,000-example training set these trusted budgets are:
  25/class  = 250 total  = 0.5%
  50/class  = 500 total  = 1%
  100/class = 1000 total = 2%
  250/class = 2500 total = 5%

This harness is deliberately CPU-oriented and uses chunked teacher scoring and
student normal equations so the full kernel query matrices need not be retained.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

NCLASS = 10
GAMMA_HOG = 4.0
GAMMA_COLOR = 0.25
HOG_WEIGHT = 0.75
TEACHER_RIDGE = 0.1
TEACHER_TEMP = 0.1

LANDMARK_COUNT = 256
LANDMARK_SEED = 29002
LANDMARK_GAMMA = 0.5
STUDENT_RIDGE = 1.0


def read_bin(path: Path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 3073:
        raise ValueError(f"Bad CIFAR binary size: {path}")
    rec = raw.reshape(-1, 3073)
    return rec[:, 1:].astype(np.float32) / 255.0, rec[:, 0].astype(np.int64)


def load_cifar(data_dir: Path):
    xs, ys = [], []
    for i in range(1, 6):
        x, y = read_bin(data_dir / f"data_batch_{i}.bin")
        xs.append(x); ys.append(y)
    xt, yt = read_bin(data_dir / "test_batch.bin")
    return np.concatenate(xs), np.concatenate(ys), xt, yt


def row_normalize(a, eps=1e-12):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, eps)


def hog576(x, chunk=2000):
    """Matches the fixed Gate-26/29 HOG-style descriptor."""
    outs = []
    for start in range(0, len(x), chunk):
        im = x[start:start+chunk].reshape(-1, 3, 32, 32)
        gray = .299*im[:,0] + .587*im[:,1] + .114*im[:,2]
        gx = gray[:,1:-1,2:] - gray[:,1:-1,:-2]
        gy = gray[:,2:,1:-1] - gray[:,:-2,1:-1]
        mag = np.sqrt(gx*gx + gy*gy)
        ang = np.mod(np.arctan2(gy, gx), np.pi)
        bins = np.minimum((9.0*ang/np.pi).astype(np.int64), 8)

        n = len(im)
        h = np.zeros((n, 8, 8, 9), dtype=np.float32)
        rows = np.arange(30) // 4
        cols = np.arange(30) // 4
        sample = np.arange(n)
        # 900 pixel positions; add across the batch for each position.
        for i in range(30):
            for j in range(30):
                np.add.at(
                    h[:, rows[i], cols[j], :],
                    (sample, bins[:, i, j]),
                    mag[:, i, j],
                )
        h = np.sqrt(h.reshape(n, -1))
        outs.append(row_normalize(h).astype(np.float32))
    return np.concatenate(outs)


def coarse48(x):
    im = x.reshape(-1, 3, 32, 32)
    return im.reshape(-1,3,4,8,4,8).mean(axis=(3,5)).reshape(len(x), -1)


def sqdist(a, b):
    # float64 accumulation improves kernel stability without retaining huge mats.
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.maximum(
        (a*a).sum(1)[:,None] + (b*b).sum(1)[None,:] - 2*a@b.T,
        0.0,
    )


def rbf(a, b, gamma):
    return np.exp(-gamma * sqdist(a, b))


def onehot(y):
    return np.eye(NCLASS, dtype=np.float64)[np.asarray(y, dtype=np.int64)]


def softmax_temp(raw, temp=TEACHER_TEMP):
    z = raw / temp
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def balanced_trusted(y, per_class, seed):
    rng = np.random.default_rng(seed)
    ids = []
    for c in range(NCLASS):
        cids = np.flatnonzero(y == c)
        ids.extend(rng.choice(cids, per_class, replace=False))
    return np.sort(np.asarray(ids, dtype=np.int64))


def fit_kernel_teacher(h, c, y, trusted):
    target = onehot(y[trusted])

    kh = rbf(h[trusted], h[trusted], GAMMA_HOG)
    kc = rbf(c[trusted], c[trusted], GAMMA_COLOR)

    fh = cho_factor(kh + TEACHER_RIDGE*np.eye(len(trusted)), check_finite=False)
    fc = cho_factor(kc + TEACHER_RIDGE*np.eye(len(trusted)), check_finite=False)

    ah = cho_solve(fh, target, check_finite=False)
    ac = cho_solve(fc, target, check_finite=False)
    return ah, ac


def teacher_probs(hq, cq, h_anchor, c_anchor, ah, ac):
    raw = (
        HOG_WEIGHT * (rbf(hq, h_anchor, GAMMA_HOG) @ ah)
        + (1-HOG_WEIGHT) * (rbf(cq, c_anchor, GAMMA_COLOR) @ ac)
    )
    return softmax_temp(raw)


def accuracy(scores, y):
    return float(np.mean(np.argmax(scores, axis=1) == y))


def landmark_phi(z, landmarks):
    k = np.exp(-LANDMARK_GAMMA * sqdist(z, landmarks))
    return np.concatenate([z, k], axis=1) / np.sqrt(2.0)


def prepare_features(x, xt, cache: Path | None = None):
    if cache and cache.exists():
        z = np.load(cache)
        return (
            z["h_train"], z["c_train"], z["z_train"],
            z["h_test"], z["c_test"], z["z_test"],
        )

    h = hog576(x)
    ht = hog576(xt)
    c0 = coarse48(x)
    ct0 = coarse48(xt)
    mu = c0.mean(0)
    sd = c0.std(0) + .01
    c = row_normalize((c0-mu)/sd).astype(np.float32)
    ct = row_normalize((ct0-mu)/sd).astype(np.float32)
    z = np.concatenate([h,c],axis=1)/np.sqrt(2.0)
    zt = np.concatenate([ht,ct],axis=1)/np.sqrt(2.0)

    if cache:
        np.savez_compressed(
            cache,
            h_train=h, c_train=c, z_train=z,
            h_test=ht, c_test=ct, z_test=zt,
        )
    return h,c,z,ht,ct,zt


def run_condition(h,c,z,ht,ct,zt,y,yt,per_class,seed,chunk=2000):
    t0=time.time()
    trusted=balanced_trusted(y, per_class, seed)
    is_trusted=np.zeros(len(y),dtype=bool)
    is_trusted[trusted]=True
    untrusted=np.flatnonzero(~is_trusted)

    ah,ac=fit_kernel_teacher(h,c,y,trusted)

    # Teacher outputs.
    p_train=np.empty((len(y),NCLASS),np.float64)
    for s in range(0,len(y),chunk):
        p_train[s:s+chunk]=teacher_probs(
            h[s:s+chunk], c[s:s+chunk],
            h[trusted], c[trusted], ah, ac
        )
    p_test=np.empty((len(yt),NCLASS),np.float64)
    for s in range(0,len(yt),chunk):
        p_test[s:s+chunk]=teacher_probs(
            ht[s:s+chunk], ct[s:s+chunk],
            h[trusted], c[trusted], ah, ac
        )

    q=p_train.copy()
    q[trusted]=onehot(y[trusted])

    # Label-independent landmarks.
    rng=np.random.default_rng(LANDMARK_SEED)
    landmark_idx=rng.choice(len(y),LANDMARK_COUNT,replace=False)
    landmarks=z[landmark_idx]

    d=z.shape[1]+LANDMARK_COUNT
    normal=STUDENT_RIDGE*np.eye(d)
    rhs=np.zeros((d,NCLASS))
    rhs_clean=np.zeros((d,NCLASS))
    rhs_pair=np.zeros((d,NCLASS))

    for s in range(0,len(y),chunk):
        ids=np.arange(s,min(s+chunk,len(y)))
        phi=landmark_phi(z[ids],landmarks)
        normal += phi.T@phi
        rhs += phi.T@q[ids]
        rhs_clean += phi.T@onehot(y[ids])

        pair=y[ids].copy()
        bad=~is_trusted[ids]
        pair[bad]=9-pair[bad]
        rhs_pair += phi.T@onehot(pair)

    w=np.linalg.solve(normal,rhs)
    w_clean=np.linalg.solve(normal,rhs_clean)
    w_pair=np.linalg.solve(normal,rhs_pair)

    phi_tr=landmark_phi(z[trusted],landmarks)
    w_tr=np.linalg.solve(
        phi_tr.T@phi_tr + STUDENT_RIDGE*np.eye(d),
        phi_tr.T@onehot(y[trusted])
    )

    def score(w):
        outs=[]
        for s in range(0,len(yt),chunk):
            outs.append(landmark_phi(zt[s:s+chunk],landmarks)@w)
        return np.concatenate(outs)

    # Explicit sentinel mutation audit. Untrusted values are nonnumeric;
    # the target builder reads labels only at trusted indices.
    observed=np.empty(len(y),dtype=object)
    observed[:]=y.astype(object)
    observed[untrusted]="UNTRUSTED_DO_NOT_READ"
    q2=p_train.copy()
    q2[trusted]=onehot(np.asarray(observed[trusted],dtype=np.int64))

    return dict(
        trusted_per_class=per_class,
        trusted_total=len(trusted),
        trusted_fraction=len(trusted)/len(y),
        seed=seed,
        teacher_untrusted_accuracy=accuracy(p_train[untrusted],y[untrusted]),
        teacher_test_accuracy=accuracy(p_test,yt),
        lpa_student_test_accuracy=accuracy(score(w),yt),
        trusted_only_test_accuracy=accuracy(score(w_tr),yt),
        all_label_landmark_reference_accuracy=accuracy(score(w_clean),yt),
        naive_pairwise_untrusted_accuracy=accuracy(score(w_pair),yt),
        sentinel_target_max_diff=float(np.max(np.abs(q2-q))),
        seconds=time.time()-t0,
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--trusted-per-class",type=int,nargs="+",default=[25,50,100,250])
    ap.add_argument("--seeds",type=int,nargs="+",default=[31001,31002,31003,31004,31005])
    ap.add_argument("--feature-cache",type=Path,default=Path("gate31_features.npz"))
    ap.add_argument("--output",type=Path,default=Path("gate31_full_results.csv"))
    args=ap.parse_args()

    x,y,xt,yt=load_cifar(args.data_dir)
    print("loaded",x.shape,xt.shape)

    t=time.time()
    h,c,z,ht,ct,zt=prepare_features(x,xt,args.feature_cache)
    print("features",z.shape,zt.shape,"seconds",time.time()-t)

    rows=[]
    for pc in args.trusted_per_class:
        for seed in args.seeds:
            row=run_condition(h,c,z,ht,ct,zt,y,yt,pc,seed)
            rows.append(row)
            print(json.dumps(row))

    with args.output.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader();w.writerows(rows)
    print("wrote",args.output)


if __name__=="__main__":
    main()

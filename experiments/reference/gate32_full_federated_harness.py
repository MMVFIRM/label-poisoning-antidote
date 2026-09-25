#!/usr/bin/env python3
"""
LPA Gate 32 — full-CIFAR federated / non-IID qualification harness.

Recommended use:
  python gate32_full_federated_harness.py \
    --data-dir ./cifar-10-batches-bin \
    --feature-cache ./gate31_features.npz \
    --trusted-per-class 100 \
    --trusted-seed 31001 \
    --alphas 0.1 0.5 1.0 \
    --partition-seeds 32301 32302 32303 32304 32305 \
    --output gate32_full_results.csv

This uses the frozen Gate-31 release-candidate architecture:
- HOG + spatial RGB
- global trusted-only kernel teacher
- Gate-26 direct soft targets
- 256 fixed-RNG random RBF landmarks
- ridge student

Federated training is exact additive sufficient-statistic aggregation:
  A_i = Phi_i^T Phi_i
  B_i = Phi_i^T Q_i
  W = (lambda I + sum_i A_i)^-1 sum_i B_i

Raw untrusted labels are not used in Phi_i or Q_i.

The harness assumes the Gate-31 feature cache contains:
  h_train, c_train, z_train, h_test, c_test, z_test.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

K=10
GAMMA_HOG=4.0
GAMMA_COLOR=0.25
HOG_WEIGHT=0.75
TEACHER_RIDGE=0.1
TEACHER_TEMP=0.1
LANDMARK_COUNT=256
LANDMARK_SEED=29002
LANDMARK_GAMMA=0.5
STUDENT_RIDGE=1.0

def read_labels(data_dir: Path):
    ys=[]
    for b in range(1,6):
        raw=np.fromfile(data_dir/f"data_batch_{b}.bin",dtype=np.uint8)
        rec=raw.reshape(-1,3073)
        ys.append(rec[:,0].astype(np.int64))
    raw=np.fromfile(data_dir/"test_batch.bin",dtype=np.uint8)
    return np.concatenate(ys), raw.reshape(-1,3073)[:,0].astype(np.int64)

def onehot(y):
    return np.eye(K,dtype=np.float64)[np.asarray(y,dtype=np.int64)]

def sqdist(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64)
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0.0)

def rbf(a,b,gamma):
    return np.exp(-gamma*sqdist(a,b))

def softmax_temp(raw,temp=TEACHER_TEMP):
    z=raw/temp
    z-=z.max(1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(1,keepdims=True)

def accuracy(scores,y):
    return float(np.mean(np.argmax(scores,axis=1)==y))

def balanced_trusted(y,per_class,seed):
    rng=np.random.default_rng(seed)
    ids=[]
    for c in range(K):
        cids=np.flatnonzero(y==c)
        ids.extend(rng.choice(cids,per_class,replace=False))
    return np.sort(np.asarray(ids,dtype=np.int64))

def fit_teacher(h,c,y,trusted):
    target=onehot(y[trusted])
    kh=rbf(h[trusted],h[trusted],GAMMA_HOG)
    kc=rbf(c[trusted],c[trusted],GAMMA_COLOR)
    fh=cho_factor(kh+TEACHER_RIDGE*np.eye(len(trusted)),check_finite=False)
    fc=cho_factor(kc+TEACHER_RIDGE*np.eye(len(trusted)),check_finite=False)
    return cho_solve(fh,target,check_finite=False), cho_solve(fc,target,check_finite=False)

def teacher_probs(hq,cq,h,c,trusted,ah,ac):
    raw=HOG_WEIGHT*(rbf(hq,h[trusted],GAMMA_HOG)@ah)
    raw+=(1-HOG_WEIGHT)*(rbf(cq,c[trusted],GAMMA_COLOR)@ac)
    return softmax_temp(raw)

def landmark_phi(z,landmarks):
    k=np.exp(-LANDMARK_GAMMA*sqdist(z,landmarks))
    return np.concatenate([z,k],axis=1)/np.sqrt(2.0)

def alloc_counts(n,p):
    raw=n*p
    cnt=np.floor(raw).astype(int)
    rem=n-cnt.sum()
    if rem:
        order=np.argsort(-(raw-cnt))[:rem]
        cnt[order]+=1
    return cnt

def dirichlet_partition(y_train,y_test,alpha,seed,n_clients):
    rng=np.random.default_rng(seed)
    train=[[] for _ in range(n_clients)]
    test=[[] for _ in range(n_clients)]
    probs=np.empty((K,n_clients),dtype=float)
    for c in range(K):
        p=rng.dirichlet(np.full(n_clients,alpha))
        probs[c]=p
        ids=rng.permutation(np.flatnonzero(y_train==c))
        cnt=alloc_counts(len(ids),p)
        pos=0
        for j,n in enumerate(cnt):
            train[j].extend(ids[pos:pos+n]); pos+=n

        ids=rng.permutation(np.flatnonzero(y_test==c))
        cnt=alloc_counts(len(ids),p)
        pos=0
        for j,n in enumerate(cnt):
            test[j].extend(ids[pos:pos+n]); pos+=n

    return (
        [np.sort(np.asarray(x,dtype=np.int64)) for x in train],
        [np.sort(np.asarray(x,dtype=np.int64)) for x in test],
        probs,
    )

def normalized_entropy(v):
    v=v[v>0]
    return float(-(v*np.log(v)).sum()/np.log(K)) if len(v) else 0.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--feature-cache",type=Path,required=True)
    ap.add_argument("--trusted-per-class",type=int,default=100)
    ap.add_argument("--trusted-seed",type=int,default=31001)
    ap.add_argument("--alphas",type=float,nargs="+",default=[.1,.5,1.0])
    ap.add_argument("--partition-seeds",type=int,nargs="+",default=[32301,32302,32303,32304,32305])
    ap.add_argument("--num-clients",type=int,default=10)
    ap.add_argument("--poisoned-clients",type=int,default=2)
    ap.add_argument("--participation-repeats",type=int,default=5)
    ap.add_argument("--chunk",type=int,default=2000)
    ap.add_argument("--output",type=Path,default=Path("gate32_full_results.csv"))
    args=ap.parse_args()

    y,yt=read_labels(args.data_dir)
    cache=np.load(args.feature_cache)
    h=cache["h_train"]; c=cache["c_train"]; z=cache["z_train"]
    ht=cache["h_test"]; ct=cache["c_test"]; zt=cache["z_test"]
    assert len(y)==len(z)==50000 and len(yt)==len(zt)==10000

    trusted=balanced_trusted(y,args.trusted_per_class,args.trusted_seed)
    trusted_mask=np.zeros(len(y),dtype=bool); trusted_mask[trusted]=True
    untrusted=np.flatnonzero(~trusted_mask)

    ah,ac=fit_teacher(h,c,y,trusted)
    p=np.empty((len(y),K),dtype=np.float64)
    ptest=np.empty((len(yt),K),dtype=np.float64)
    for s in range(0,len(y),args.chunk):
        p[s:s+args.chunk]=teacher_probs(h[s:s+args.chunk],c[s:s+args.chunk],h,c,trusted,ah,ac)
    for s in range(0,len(yt),args.chunk):
        ptest[s:s+args.chunk]=teacher_probs(ht[s:s+args.chunk],ct[s:s+args.chunk],h,c,trusted,ah,ac)

    q=p.copy(); q[trusted]=onehot(y[trusted])

    rng=np.random.default_rng(LANDMARK_SEED)
    landmark_idx=rng.choice(len(y),LANDMARK_COUNT,replace=False)
    landmarks=z[landmark_idx]
    phi=landmark_phi(z,landmarks)
    phitest=landmark_phi(zt,landmarks)
    dim=phi.shape[1]

    Acentral=phi.T@phi + STUDENT_RIDGE*np.eye(dim)
    Bcentral=phi.T@q
    wcentral=np.linalg.solve(Acentral,Bcentral)
    central_test=accuracy(phitest@wcentral,yt)

    # Clean label-using reference for the same student class.
    wclean=np.linalg.solve(Acentral,phi.T@onehot(y))
    clean_direct_test=accuracy(phitest@wclean,yt)

    rows=[]
    for alpha in args.alphas:
        for pseed in args.partition_seeds:
            clients,tests,probs=dirichlet_partition(y,yt,alpha,pseed,args.num_clients)

            As=[phi[idx].T@phi[idx] for idx in clients]
            Bs=[phi[idx].T@q[idx] for idx in clients]
            wf=np.linalg.solve(
                STUDENT_RIDGE*np.eye(dim)+sum(As),
                sum(Bs),
            )

            # Poison the largest clients; trusted labels on them stay intact.
            sizes=np.array([len(x) for x in clients])
            poisoned=np.argsort(-sizes)[:args.poisoned_clients]
            observed=y.astype(object)
            poison_indices=[]
            for j in poisoned:
                ids=clients[j][~trusted_mask[clients[j]]]
                poison_indices.extend(ids.tolist())
                observed[ids]="UNTRUSTED_DO_NOT_READ"

            # Explicit target rebuild: untrusted observed values are never parsed.
            q2=p.copy()
            q2[trusted]=onehot(np.asarray(observed[trusted],dtype=np.int64))
            target_diff=float(np.max(np.abs(q2-q)))

            Bs2=[phi[idx].T@q2[idx] for idx in clients]
            wf2=np.linalg.solve(
                STUDENT_RIDGE*np.eye(dim)+sum(As),
                sum(Bs2),
            )

            # Pairwise label poisoning for a label-using direct baseline.
            ypoison=y.copy()
            poison_numeric=np.asarray(poison_indices,dtype=np.int64)
            ypoison[poison_numeric]=9-ypoison[poison_numeric]
            Bpoison=[phi[idx].T@onehot(ypoison[idx]) for idx in clients]
            wpoison=np.linalg.solve(
                STUDENT_RIDGE*np.eye(dim)+sum(As),
                sum(Bpoison),
            )

            client_lpa=[]
            client_naive=[]
            for idx in tests:
                if len(idx):
                    client_lpa.append(accuracy(phitest[idx]@wf,yt[idx]))
                    client_naive.append(accuracy(phitest[idx]@wpoison,yt[idx]))

            rngp=np.random.default_rng(900000+pseed)
            p80=[]; p50=[]
            for _ in range(args.participation_repeats):
                active=rngp.choice(args.num_clients,int(round(.8*args.num_clients)),replace=False)
                w80=np.linalg.solve(
                    STUDENT_RIDGE*np.eye(dim)+sum(As[j] for j in active),
                    sum(Bs[j] for j in active),
                )
                p80.append(accuracy(phitest@w80,yt))

                active=rngp.choice(args.num_clients,int(round(.5*args.num_clients)),replace=False)
                w50=np.linalg.solve(
                    STUDENT_RIDGE*np.eye(dim)+sum(As[j] for j in active),
                    sum(Bs[j] for j in active),
                )
                p50.append(accuracy(phitest@w50,yt))

            trusted_counts=np.array([trusted_mask[idx].sum() for idx in clients])
            entropy=np.mean([normalized_entropy(probs[:,j]) for j in range(args.num_clients)])

            row=dict(
                alpha=alpha,
                partition_seed=pseed,
                num_clients=args.num_clients,
                trusted_total=len(trusted),
                trusted_fraction=len(trusted)/len(y),
                teacher_test_accuracy=accuracy(ptest,yt),
                central_lpa_test_accuracy=central_test,
                federated_lpa_test_accuracy=accuracy(phitest@wf,yt),
                clean_direct_test_accuracy=clean_direct_test,
                central_federated_weight_max_diff=float(np.max(np.abs(wf-wcentral))),
                poison_target_max_diff=target_diff,
                poison_student_weight_max_diff=float(np.max(np.abs(wf2-wf))),
                poison_fraction=len(poison_indices)/len(y),
                naive_pairwise_poison_test_accuracy=accuracy(phitest@wpoison,yt),
                client_train_min=int(sizes.min()),
                client_train_max=int(sizes.max()),
                trusted_client_min=int(trusted_counts.min()),
                trusted_client_max=int(trusted_counts.max()),
                trusted_zero_clients=int(np.sum(trusted_counts==0)),
                mean_client_entropy=entropy,
                lpa_worst_client_accuracy=float(np.min(client_lpa)),
                lpa_client_accuracy_variance=float(np.var(client_lpa,ddof=1)),
                naive_poison_worst_client_accuracy=float(np.min(client_naive)),
                participation80_mean=float(np.mean(p80)),
                participation80_min=float(np.min(p80)),
                participation50_mean=float(np.mean(p50)),
                participation50_min=float(np.min(p50)),
            )
            rows.append(row)
            print(json.dumps(row))

    with args.output.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print("wrote",args.output)

if __name__=="__main__":
    main()

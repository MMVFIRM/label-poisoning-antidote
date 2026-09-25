#!/usr/bin/env python3
"""Demonstrate exact ridge federation under full participation."""

import numpy as np

from lpa import federated_equivalence_audit

rng=np.random.default_rng(2)
phi=rng.normal(size=(1000,80))
q=rng.random(size=(1000,10));q/=q.sum(axis=1,keepdims=True)
parts=[x.astype(np.int64) for x in np.array_split(rng.permutation(len(phi)),10)]

diff=federated_equivalence_audit(phi,q,parts,ridge=1.)
print(f"max |W_fed-W_central| = {diff:.3e}")

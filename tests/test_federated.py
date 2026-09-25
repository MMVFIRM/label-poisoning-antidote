import numpy as np

from lpa.federated import federated_equivalence_audit, packed_symmetric_payload_bytes


def test_full_participation_federated_equals_centralized():
    rng=np.random.default_rng(4)
    phi=rng.normal(size=(240,31))
    q=rng.random(size=(240,6));q/=q.sum(axis=1,keepdims=True)
    parts=[x.astype(np.int64) for x in np.array_split(rng.permutation(len(phi)),9)]
    diff=federated_equivalence_audit(phi,q,parts,ridge=1.)
    assert diff < 1e-10


def test_gate32_float32_payload_matches_documented_value():
    size=packed_symmetric_payload_bytes(880,10,4)
    mib=size/(1024**2)
    assert 1.50 < mib < 1.52

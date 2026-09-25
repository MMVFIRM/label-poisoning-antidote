import numpy as np

from lpa.features import CIFARFeatureExtractor, coarse48, hog576


def test_cifar_feature_shapes_and_finiteness():
    rng=np.random.default_rng(2)
    x=rng.random((5,3072),dtype=np.float32)
    h=hog576(x,chunk_size=2)
    c=coarse48(x)
    assert h.shape==(5,576)
    assert c.shape==(5,48)
    assert np.isfinite(h).all()
    assert np.isfinite(c).all()
    assert np.allclose(np.linalg.norm(h,axis=1),1.,atol=1e-5)


def test_feature_extractor_is_deterministic():
    rng=np.random.default_rng(3)
    x=rng.random((6,3072),dtype=np.float32)
    ex=CIFARFeatureExtractor().fit(x)
    a=ex.transform(x,chunk_size=3)
    b=ex.transform(x,chunk_size=2)
    assert np.allclose(a.view_a,b.view_a)
    assert np.allclose(a.view_b,b.view_b)
    assert np.allclose(a.joint,b.joint)

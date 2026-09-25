import json
from pathlib import Path

import numpy as np
import pytest
from conftest import synthetic_images, synthetic_views

from lpa.audit import mutation_invariance_audit
from lpa.config import KMeansFeatureConfig, KMeansTeacherConfig, LinearStudentConfig, LPAConfig
from lpa.features import KMeansPatchFeatureExtractor
from lpa.federated import federated_equivalence_audit
from lpa.pipeline import LabelPoisoningAntidote
from lpa.student import LinearRidgeStudent, trusted_row_weights
from lpa.teacher import BlendedKernelTeacher

TINY_KMEANS=KMeansFeatureConfig(centroids=8,patch_samples=3000,iterations=4,chunk_size=16,workers=2)


def tiny_config(classes=3):
    return LPAConfig(n_classes=classes,kmeans=TINY_KMEANS,kmeans_teacher=KMeansTeacherConfig(gamma=1.))


def test_default_architecture_is_kmeans_and_old_configs_load_as_v1():
    assert LPAConfig().architecture=="kmeans"
    assert LPAConfig.v1().architecture=="landmark"
    old=LPAConfig.v1().to_dict();del old["architecture"]
    assert LPAConfig.from_dict(old).architecture=="landmark"
    with pytest.raises(ValueError):
        LPAConfig(architecture="other")


def test_kmeans_features_are_deterministic_normalized_and_layout_independent():
    x,_,_=synthetic_images()
    f1=KMeansPatchFeatureExtractor(TINY_KMEANS).fit_transform(x)
    ex=KMeansPatchFeatureExtractor(TINY_KMEANS).fit(x)
    assert f1.shape==(len(x),4*TINY_KMEANS.centroids)
    assert np.array_equal(f1,ex.transform(x))
    assert np.allclose(np.linalg.norm(f1,axis=1),1.)
    assert np.array_equal(f1,ex.transform(np.transpose(x,(0,2,3,1))))
    assert np.array_equal(f1,ex.transform(x.reshape(len(x),-1)))


def test_kmeans_features_accept_uint8_pixels():
    x,_,_=synthetic_images()
    u8=np.round(x*255).astype(np.uint8)
    ex=KMeansPatchFeatureExtractor(TINY_KMEANS).fit(u8)
    assert np.allclose(ex.transform(u8),ex.transform(u8.astype(np.float32)/255),atol=1e-4)


def test_linear_student_matches_weighted_closed_form():
    a,b,z,y,trusted=synthetic_views()
    rng=np.random.default_rng(0)
    q=rng.random((len(z),3))
    cfg=LinearStudentConfig(ridge=.2,trusted_weight=7.)
    s=LinearRidgeStudent(3,cfg).fit(z,q,trusted_indices=trusted)
    w=trusted_row_weights(len(z),trusted,7.)
    expected=np.linalg.solve(z.T@(z*w[:,None])+.2*np.eye(z.shape[1]),(z*w[:,None]).T@q)
    assert np.allclose(s.weights_,expected)
    assert federated_equivalence_audit(z,q,np.array_split(np.arange(len(z)),5),.2,weights=w)<1e-10


def test_kmeans_pipeline_invariance_audit():
    a,b,z,y,trusted=synthetic_views()
    teacher=BlendedKernelTeacher(3,KMeansTeacherConfig(gamma=1.)).fit(a[trusted],b[trusted],z[trusted],y[trusted])
    p=teacher.predict_proba(a,b,z)
    la=y.astype(object);lb=y.astype(object)
    mask=np.ones(len(y),bool);mask[trusted]=False
    lb[mask]="UNTRUSTED_DO_NOT_READ"
    report=mutation_invariance_audit(p,z,trusted,la,lb,3)
    assert report.passed


def test_kmeans_fit_views_ignores_everything_but_trusted_labels():
    a,b,z,y,trusted=synthetic_views()
    m1=LabelPoisoningAntidote(tiny_config()).fit_views(a,b,z,trusted,y[trusted])
    m2=LabelPoisoningAntidote(tiny_config()).fit_views(a,b,z,trusted,y[trusted])
    assert m1.student.weight_hash()==m2.student.weight_hash()
    with pytest.raises(ValueError,match="joint"):
        m1.teacher_probabilities_views(a,b)
    assert m1.teacher_probabilities_views(a,b,z).shape==(len(z),3)


def test_kmeans_image_pipeline_round_trip_and_tamper_evidence(tmp_path: Path):
    x,y,trusted=synthetic_images()
    model=LabelPoisoningAntidote(tiny_config()).fit_images(x,trusted,y[trusted])
    assert model.score_images(x,y)>.9
    pred=model.predict_images(x)
    path=model.save(tmp_path/"m")
    meta=json.loads((path/"metadata.json").read_text())
    assert meta["architecture"]=="kmeans" and meta["format_version"]==3
    loaded=LabelPoisoningAntidote.load(path,expected_fingerprint=LabelPoisoningAntidote.fingerprint(path))
    assert loaded.architecture=="kmeans"
    assert np.array_equal(pred,loaded.predict_images(x))
    assert np.array_equal(model.predict_scores_images(x),loaded.predict_scores_images(x))
    with np.load(path/"model.npz") as f:
        state={k:f[k] for k in f.files}
    assert "student_landmarks" not in state and "kmeans_centroids" in state
    state["kmeans_centroids"]=state["kmeans_centroids"]+1
    np.savez_compressed(path/"model.npz",**state)
    with pytest.raises(ValueError,match="kmeans_centroids"):
        LabelPoisoningAntidote.load(path)


def test_unfitted_kmeans_image_prediction_raises():
    x,_,_=synthetic_images()
    with pytest.raises(RuntimeError):
        LabelPoisoningAntidote(tiny_config()).predict_images(x)

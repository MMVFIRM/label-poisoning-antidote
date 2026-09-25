"""LPA 1.0.0 model files and the v1 preset must keep working unchanged in 2.x.

The fixtures under tests/fixtures/ were written by LPA 1.0.0 (model format 2).
Predicted labels must match exactly. Scores and weights must match to
floating-point precision: they are bit-identical on the platform that wrote the
fixtures, but other BLAS builds may differ in the last bits.
"""
from pathlib import Path

import numpy as np
from conftest import synthetic_images, synthetic_views

from lpa.config import LPAConfig, StudentConfig, TeacherConfig
from lpa.pipeline import LabelPoisoningAntidote

FIXTURES=Path(__file__).parent/"fixtures"


RTOL,ATOL=1e-9,1e-12


def _fixture_weights(name):
    with np.load(FIXTURES/name/"model.npz") as f:
        return f["student_weights"]


def _expected():
    with np.load(FIXTURES/"v1_expected.npz") as f:
        return {k:f[k] for k in f.files}


def test_v1_views_model_file_loads_and_predicts_identically():
    exp=_expected()
    a,b,z,y,trusted=synthetic_views()
    model=LabelPoisoningAntidote.load(FIXTURES/"v1_views_model")
    assert model.architecture=="landmark"
    assert np.allclose(model.predict_scores_views(z),exp["views_scores"],rtol=RTOL,atol=ATOL)
    assert np.array_equal(model.predict_views(z),exp["views_scores"].argmax(1))


def test_v1_image_model_file_loads_and_predicts_identically():
    exp=_expected()
    x,y,trusted=synthetic_images()
    model=LabelPoisoningAntidote.load(FIXTURES/"v1_image_model")
    assert np.array_equal(model.predict_images(x),exp["image_pred"])
    assert np.allclose(model.predict_scores_images(x),exp["image_scores"],rtol=RTOL,atol=ATOL)


def test_v1_preset_refits_identically_to_1_0_0():
    exp=_expected()
    a,b,z,y,trusted=synthetic_views()
    cfg=LPAConfig.v1(
        n_classes=3,
        teacher=TeacherConfig(gamma_view_a=1.,gamma_view_b=1.),
        student=StudentConfig(landmark_count=20,landmark_seed=9),
    )
    model=LabelPoisoningAntidote(cfg).fit_views(a,b,z,trusted,y[trusted])
    assert np.allclose(model.student.weights_,_fixture_weights("v1_views_model"),rtol=RTOL,atol=ATOL)
    x,yy,tr=synthetic_images()
    img=LabelPoisoningAntidote(LPAConfig.v1(n_classes=3,student=StudentConfig(landmark_count=16)))
    img.fit_images(x,tr,yy[tr])
    assert np.allclose(img.student.weights_,_fixture_weights("v1_image_model"),rtol=RTOL,atol=ATOL)
    assert np.array_equal(img.predict_images(x),exp["image_pred"])

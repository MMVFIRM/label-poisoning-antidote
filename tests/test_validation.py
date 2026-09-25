import numpy as np
import pytest
from conftest import small_config, synthetic_views

from lpa.audit import validate_trusted_pairs
from lpa.features import hog576
from lpa.federated import (
    RidgeSufficientStatistics,
    aggregate_sufficient_statistics,
    client_sufficient_statistics,
)
from lpa.pipeline import LabelPoisoningAntidote


def test_integral_float_labels_are_accepted():
    idx,y=validate_trusted_pairs(np.array([0,1]),np.array([0.,2.]),5,3)
    assert y.dtype==np.int64 and y.tolist()==[0,2]


@pytest.mark.parametrize(
    "labels",
    [np.array([0.9,1.7]),np.array([True,False]),np.array(["0","1"]),np.array([0,np.nan])],
)
def test_non_integer_trusted_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        validate_trusted_pairs(np.array([0,1]),labels,5,3)


def test_non_integer_trusted_indices_are_rejected():
    with pytest.raises(ValueError):
        validate_trusted_pairs(np.array([0.5,1.]),np.array([0,1]),5,3)


def test_nan_features_are_rejected_at_fit():
    a,b,z,y,trusted=synthetic_views()
    bad_a=a.copy();bad_a[trusted[0],0]=np.nan
    with pytest.raises(ValueError,match="NaN"):
        LabelPoisoningAntidote(small_config()).fit_views(bad_a,b,z,trusted,y[trusted])
    bad_z=z.copy();bad_z[-1,0]=np.inf
    with pytest.raises(ValueError,match="NaN"):
        LabelPoisoningAntidote(small_config()).fit_views(a,b,bad_z,trusted,y[trusted])


def test_nan_features_are_rejected_at_predict(fitted_views_model):
    model,(a,b,z,y,trusted)=fitted_views_model
    bad=z[:3].copy();bad[1,2]=np.nan
    with pytest.raises(ValueError,match="NaN"):
        model.predict_views(bad)


def test_wrong_feature_width_gives_clear_error(fitted_views_model):
    model,(a,b,z,y,trusted)=fitted_views_model
    with pytest.raises(ValueError,match="expects 13 joint features; got 7"):
        model.predict_views(np.zeros((2,7)))
    with pytest.raises(ValueError,match="Teacher expects views"):
        model.teacher_probabilities_views(np.zeros((2,3)),b[:2])


def test_nan_image_data_is_rejected():
    x=np.zeros((2,3072),dtype=np.float32);x[0,0]=np.nan
    with pytest.raises(ValueError,match="NaN"):
        hog576(x)


def test_federated_aggregation_rejects_non_finite_or_asymmetric_statistics():
    rng=np.random.default_rng(1)
    good=client_sufficient_statistics(rng.normal(size=(20,4)),rng.random(size=(20,2)))
    nan_a=good.a.copy();nan_a[0,0]=np.nan
    with pytest.raises(ValueError,match="NaN"):
        aggregate_sufficient_statistics([good,RidgeSufficientStatistics(nan_a,good.b,20)],ridge=1.)
    skew=good.a.copy();skew[0,1]+=5.
    with pytest.raises(ValueError,match="symmetric"):
        aggregate_sufficient_statistics([good,RidgeSufficientStatistics(skew,good.b,20)],ridge=1.)
    with pytest.raises(ValueError,match="NaN"):
        client_sufficient_statistics(np.full((3,4),np.nan),np.zeros((3,2)))

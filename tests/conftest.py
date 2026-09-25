import numpy as np
import pytest

from lpa.config import LPAConfig, StudentConfig, TeacherConfig
from lpa.pipeline import LabelPoisoningAntidote


def synthetic_views(seed=5, n=120, classes=3):
    rng=np.random.default_rng(seed)
    y=np.repeat(np.arange(classes),n//classes);rng.shuffle(y)
    ca=rng.normal(size=(classes,9))*1.4
    cb=rng.normal(size=(classes,4))*1.1
    a=ca[y]+.7*rng.normal(size=(n,9))
    b=cb[y]+.7*rng.normal(size=(n,4))
    a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    z=np.concatenate([a,b],axis=1)/np.sqrt(2.)
    trusted=np.concatenate([np.flatnonzero(y==c)[:7] for c in range(classes)])
    return a,b,z,y,trusted


def small_config(classes=3):
    return LPAConfig(
        n_classes=classes,
        teacher=TeacherConfig(gamma_view_a=1.,gamma_view_b=1.),
        student=StudentConfig(landmark_count=20,landmark_seed=9),
    )


@pytest.fixture
def fitted_views_model():
    a,b,z,y,trusted=synthetic_views()
    model=LabelPoisoningAntidote(small_config()).fit_views(a,b,z,trusted,y[trusted])
    return model,(a,b,z,y,trusted)

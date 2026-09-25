from pathlib import Path

import numpy as np

from lpa.config import LPAConfig, StudentConfig, TeacherConfig
from lpa.pipeline import LabelPoisoningAntidote


def test_pipeline_fit_predict_save_load(tmp_path: Path):
    rng=np.random.default_rng(5)
    n=120; classes=3
    y=np.repeat(np.arange(classes),n//classes);rng.shuffle(y)
    ca=rng.normal(size=(classes,9))*1.4
    cb=rng.normal(size=(classes,4))*1.1
    a=ca[y]+.7*rng.normal(size=(n,9))
    b=cb[y]+.7*rng.normal(size=(n,4))
    a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    z=np.concatenate([a,b],axis=1)/np.sqrt(2.)
    trusted=np.concatenate([np.flatnonzero(y==c)[:7] for c in range(classes)])
    cfg=LPAConfig(
        n_classes=classes,
        teacher=TeacherConfig(gamma_view_a=1.,gamma_view_b=1.),
        student=StudentConfig(landmark_count=20,landmark_seed=9),
    )
    model=LabelPoisoningAntidote(cfg).fit_views(a,b,z,trusted,y[trusted])
    pred=model.predict_views(z)
    assert pred.shape==(n,)
    path=model.save(tmp_path/"model")
    loaded=LabelPoisoningAntidote.load(path)
    assert np.array_equal(pred,loaded.predict_views(z))
    assert model.student.weight_hash()==loaded.student.weight_hash()

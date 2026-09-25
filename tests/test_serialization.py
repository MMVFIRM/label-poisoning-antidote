import json
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

import lpa
from lpa.config import LPAConfig, StudentConfig
from lpa.pipeline import LabelPoisoningAntidote


def _load_state(path):
    with np.load(path/"model.npz") as f:
        return {k:f[k] for k in f.files}


@pytest.mark.parametrize(
    "name",
    ["teacher_alpha_a","teacher_view_b_anchor","student_landmarks","student_weights"],
)
def test_tampering_with_any_saved_array_is_detected(fitted_views_model,tmp_path,name):
    model,_=fitted_views_model
    path=model.save(tmp_path/"m")
    state=_load_state(path);state[name]=state[name]+1.
    np.savez_compressed(path/"model.npz",**state)
    with pytest.raises(ValueError):
        LabelPoisoningAntidote.load(path)


def test_added_arrays_are_detected(fitted_views_model,tmp_path):
    model,_=fitted_views_model
    path=model.save(tmp_path/"m")
    state=_load_state(path);state["extra"]=np.zeros(1)
    np.savez_compressed(path/"model.npz",**state)
    with pytest.raises(ValueError,match="manifest"):
        LabelPoisoningAntidote.load(path)


def test_fingerprint_detects_consistent_rewrite(fitted_views_model,tmp_path):
    model,_=fitted_views_model
    path=model.save(tmp_path/"m")
    fp=LabelPoisoningAntidote.fingerprint(path)
    LabelPoisoningAntidote.load(path,expected_fingerprint=fp)
    meta=json.loads((path/"metadata.json").read_text())
    meta["target_hash"]="0"*64
    (path/"metadata.json").write_text(json.dumps(meta))
    LabelPoisoningAntidote.load(path)  # still internally consistent
    with pytest.raises(ValueError,match="fingerprint"):
        LabelPoisoningAntidote.load(path,expected_fingerprint=fp)


def test_save_can_omit_trusted_labels(fitted_views_model,tmp_path):
    model,(a,b,z,y,trusted)=fitted_views_model
    path=model.save(tmp_path/"m",include_trusted_labels=False)
    state=_load_state(path)
    assert "trusted_indices" not in state and "trusted_labels" not in state
    loaded=LabelPoisoningAntidote.load(path)
    assert np.array_equal(loaded.predict_views(z),model.predict_views(z))


def test_format_1_models_still_load_with_warning(fitted_views_model,tmp_path):
    model,(a,b,z,y,trusted)=fitted_views_model
    path=model.save(tmp_path/"m")
    meta=json.loads((path/"metadata.json").read_text())
    meta["format_version"]=1
    del meta["array_sha256"]
    (path/"metadata.json").write_text(json.dumps(meta))
    with pytest.warns(UserWarning,match="format-1"):
        loaded=LabelPoisoningAntidote.load(path)
    assert np.array_equal(loaded.predict_views(z),model.predict_views(z))


def test_image_pipeline_round_trip(tmp_path):
    rng=np.random.default_rng(11)
    n=60;classes=3
    y=np.repeat(np.arange(classes),n//classes)
    base=rng.random((classes,3,32,32),dtype=np.float32)
    x=np.clip(base[y]+.15*rng.normal(size=(n,3,32,32)).astype(np.float32),0,1)
    trusted=np.concatenate([np.flatnonzero(y==c)[:5] for c in range(classes)])
    cfg=LPAConfig.v1(n_classes=classes,student=StudentConfig(landmark_count=16))
    model=LabelPoisoningAntidote(cfg).fit_images(x,trusted,y[trusted])
    pred=model.predict_images(x)
    assert model.score_images(x,y)>.9
    path=model.save(tmp_path/"img")
    loaded=LabelPoisoningAntidote.load(path)
    assert np.array_equal(pred,loaded.predict_images(x))
    nhwc=np.transpose(x,(0,2,3,1))
    assert np.array_equal(pred,loaded.predict_images(nhwc))


def test_version_has_a_single_source():
    pyproject=(Path(__file__).resolve().parents[1]/"pyproject.toml").read_text()
    assert re.search(r'^dynamic = \["version"\]',pyproject,re.M)
    assert not re.search(r'^version\s*=\s*"',pyproject,re.M)
    assert version("label-poisoning-antidote")==lpa.__version__

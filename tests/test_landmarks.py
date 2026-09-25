import numpy as np

from lpa.config import StudentConfig
from lpa.student import LandmarkRidgeStudent


def test_landmark_selection_is_label_independent_and_deterministic():
    rng=np.random.default_rng(8)
    z=rng.normal(size=(80,12))
    config=StudentConfig(landmark_count=16,landmark_seed=29002)
    a=LandmarkRidgeStudent(4,config).choose_landmarks(z)
    b=LandmarkRidgeStudent(4,config).choose_landmarks(z)
    assert np.array_equal(a,b)
    assert len(np.unique(a))==16

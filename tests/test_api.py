import inspect

from lpa.pipeline import LabelPoisoningAntidote


def test_core_fit_api_has_no_untrusted_label_parameter():
    params = inspect.signature(LabelPoisoningAntidote.fit_views).parameters
    assert "untrusted_labels" not in params
    assert "labels" not in params
    assert "trusted_indices" in params
    assert "trusted_labels" in params

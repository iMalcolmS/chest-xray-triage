import numpy as np
import pytest

from cxr.thresholds import pick, sensitivity_specificity, threshold_for_sensitivity


def test_threshold_reaches_target_sensitivity():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.65, 0.3, 0.1])
    t = threshold_for_sensitivity(y, s, target=0.8)
    assert t == 0.6
    sens, spec = sensitivity_specificity(y, s, t)
    assert sens == 0.8 and spec == pytest.approx(2 / 3)


def test_no_positives_raises():
    with pytest.raises(ValueError):
        threshold_for_sensitivity([0, 0], [0.1, 0.2])


def test_pick_falls_back_to_any():
    thresholds = {"Effusion": {"any": 0.5, "AP": 0.7}}
    assert pick(thresholds, "Effusion", "AP") == 0.7
    assert pick(thresholds, "Effusion", "PA") == 0.5
    assert pick(thresholds, "Effusion", None) == 0.5

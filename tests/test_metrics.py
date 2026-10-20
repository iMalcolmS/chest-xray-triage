import numpy as np
import pytest

from cxr.metrics import auc_with_patient_bootstrap


def test_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    groups = np.array(["a", "b", "c", "d", "e", "f"])
    auc, low, high = auc_with_patient_bootstrap(y, s, groups, n_boot=200)
    assert auc == 1.0 and low == 1.0 and high == 1.0


def test_interval_contains_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y * 0.5 + rng.normal(0, 0.5, 300)
    groups = np.repeat(np.arange(100), 3)  # 3 images per patient
    first = auc_with_patient_bootstrap(y, s, groups, n_boot=300, seed=7)
    second = auc_with_patient_bootstrap(y, s, groups, n_boot=300, seed=7)
    auc, low, high = first
    assert first == second
    assert 0 <= low <= auc <= high <= 1


def test_single_class_raises():
    with pytest.raises(ValueError):
        auc_with_patient_bootstrap([1, 1, 1], [0.2, 0.5, 0.9], ["a", "b", "c"])

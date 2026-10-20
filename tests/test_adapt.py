import numpy as np

from cxr.adapt import CS, choose_c, make_splits, out_of_fold_scores


def _toy(n_patients=60, per_patient=3, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_patients), per_patient)
    y = np.repeat(rng.integers(0, 2, n_patients), per_patient)
    X = rng.normal(size=(len(y), 8)) + y[:, None] * 1.5
    return X, y, groups


def test_patient_split_never_shares_a_patient():
    _, y, groups = _toy()
    for train, test in make_splits(y, groups):
        assert not set(groups[train]) & set(groups[test])


def test_choose_c_returns_a_grid_value():
    X, y, groups = _toy()
    assert choose_c(X, y, groups) in CS


def test_every_image_is_scored_once_and_signal_is_learned():
    X, y, groups = _toy()
    scores, chosen = out_of_fold_scores(X, y, groups)
    assert scores.shape == y.shape
    assert len(chosen) == 5 and all(c in CS for c in chosen)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert scores[y == 1].mean() > scores[y == 0].mean()

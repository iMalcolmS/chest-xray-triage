"""Adapt a pretrained model to a new hospital by training only a new final layer.

The DenseNet backbone stays frozen. A logistic regression is trained per
finding on top of frozen inputs (image features and/or the CheXpert model's
own outputs). Out-of-fold predictions from 5-fold cross-validation give every
image a score from a model that never saw it.

Two lessons from the first version (results/adaptation_v1.md):
  * A fixed C=0.1 overfit 1024 features with only 62-967 positives per finding,
    so the regularisation strength is now chosen by an inner cross-validation
    on the training folds only (nested CV, so the test fold never picks C).
  * A head trained from scratch throws away what CheXpert learned from ~224k
    images, so we also offer the CheXpert outputs as inputs ("stacking").

Splitting by patient matters: the same patient's images look alike, so if
they land on both sides of a split the test score is inflated (leakage).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5
INNER_FOLDS = 3
CS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)  # strong L2 grid: v1 showed C=0.1 overfits


def make_splits(y, groups=None, n_folds=N_FOLDS, seed=0):
    """Train/test index pairs. With groups, no patient appears on both sides."""
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


def _fit(X, y, C):
    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight="balanced", max_iter=3000),
    )
    return head.fit(X, y)


def choose_c(X, y, groups=None, seed=0, cs=CS):
    """Pick C by inner cross-validated AUC, using only the data it is given."""
    splits = make_splits(y, groups, n_folds=INNER_FOLDS, seed=seed)
    mean_auc = []
    for C in cs:
        aucs = [roc_auc_score(y[val], _fit(X[tr], y[tr], C).predict_proba(X[val])[:, 1])
                for tr, val in splits]
        mean_auc.append(np.mean(aucs))
    return cs[int(np.argmax(mean_auc))]


def out_of_fold_scores(X, y, groups=None, seed=0, cs=CS):
    """Score every image with a head tuned and trained on the other folds.

    Returns (scores, chosen_cs). Inner splits follow the outer rule: by patient
    when groups are given, by image otherwise.
    """
    X, y = np.asarray(X), np.asarray(y)
    g = None if groups is None else np.asarray(groups)
    scores = np.zeros(len(y), dtype=float)
    chosen = []
    for train, test in make_splits(y, g, seed=seed):
        C = choose_c(X[train], y[train], None if g is None else g[train], seed=seed, cs=cs)
        chosen.append(C)
        scores[test] = _fit(X[train], y[train], C).predict_proba(X[test])[:, 1]
    return scores, chosen

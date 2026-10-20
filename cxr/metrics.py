"""AUC with confidence intervals from a patient-level (cluster) bootstrap.

Many patients have several images. Resampling images independently would
treat them as independent evidence and give intervals that are too narrow,
so we resample patients and keep all of each patient's images together.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_with_patient_bootstrap(y_true, y_score, groups, n_boot=1000, seed=0, alpha=0.05):
    """Return (auc, ci_low, ci_high).

    y_true: 0/1 labels, y_score: model scores, groups: patient id per image.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    groups = np.asarray(groups)
    if y_true.min() == y_true.max():
        raise ValueError("AUC needs both positive and negative examples")

    auc = roc_auc_score(y_true, y_score)

    patients, inverse = np.unique(groups, return_inverse=True)
    rows_by_patient = [np.flatnonzero(inverse == k) for k in range(len(patients))]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(patients), len(patients))
        idx = np.concatenate([rows_by_patient[k] for k in chosen])
        y = y_true[idx]
        if y.min() == y.max():  # resample with one class only: AUC undefined, skip
            continue
        boots.append(roc_auc_score(y, y_score[idx]))
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(auc), float(low), float(high)

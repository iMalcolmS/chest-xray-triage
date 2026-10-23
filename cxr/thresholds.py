"""Operating thresholds for triage: one per finding and per view (AP / PA).

A triage tool should miss few true cases, so each threshold is set to reach a
target sensitivity on a tuning set of patients, then checked on held-out
patients. The view analysis showed the CheXpert model scores portable AP
films higher whether or not disease is present, so a single threshold
over-flags AP films; separate thresholds per view correct for that.
"""

import math

import numpy as np

VIEWS = ("AP", "PA")


def threshold_for_sensitivity(y_true, scores, target=0.8):
    """Highest threshold whose sensitivity on (y_true, scores) is at least target."""
    pos = np.sort(np.asarray(scores)[np.asarray(y_true) == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("need at least one positive example")
    k = math.ceil(target * len(pos))
    return float(pos[k - 1])


def sensitivity_specificity(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    pos, neg = y_true == 1, y_true == 0
    sens = flagged[pos].mean() if pos.any() else float("nan")
    spec = (~flagged[neg]).mean() if neg.any() else float("nan")
    return float(sens), float(spec)


MIN_VIEW_AUC = 0.60  # below this the model cannot separate cases on that view


def is_supported(thresholds, finding, view):
    """False when the model was too weak on this view to report the finding."""
    return view not in thresholds[finding].get("unsupported_views", [])


def pick(thresholds, finding, view):
    """Threshold for a finding and view, falling back to the all-views one."""
    entry = thresholds[finding]
    return entry.get(view, entry["any"]) if view in VIEWS else entry["any"]

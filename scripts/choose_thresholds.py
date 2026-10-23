"""Choose triage thresholds per finding and view, and check them on held-out patients.

Patients are split 50/50. On the tuning half, each threshold is set to reach
the target sensitivity (default 80%). On the test half we compare a single
threshold for all films with separate AP and PA thresholds.

Usage:  python -m scripts.choose_thresholds      (needs external_validation)
Writes: service/thresholds.json   used by the API
        results/thresholds.md     the held-out comparison
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from cxr.data import load_labels
from cxr.thresholds import (MIN_VIEW_AUC, VIEWS, sensitivity_specificity,
                            threshold_for_sensitivity)

FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Pneumonia", "Pneumothorax")
MIN_TUNE_POSITIVES = 10  # fewer than this in a view: fall back to the all-views threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--target", type=float, default=0.8, help="target sensitivity")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    preds = pd.read_csv(Path(args.out) / "predictions.csv")
    assert list(preds["image"]) == list(labels["image"]), "predictions out of order"

    split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    tune, test = next(split.split(labels, groups=labels["patient_id"]))
    view = labels["view"].to_numpy()

    thresholds, rows = {}, []
    for finding in FINDINGS:
        y = labels[finding].to_numpy()
        s = preds[f"chex:{finding}"].to_numpy()
        if y[tune].sum() == 0:
            print(f"Skipping {finding}: no positives in the tuning half")
            continue
        entry = {"any": threshold_for_sensitivity(y[tune], s[tune], args.target)}
        for v in VIEWS:
            idx = tune[view[tune] == v]
            if y[idx].sum() >= MIN_TUNE_POSITIVES:
                entry[v] = threshold_for_sensitivity(y[idx], s[idx], args.target)
                if roc_auc_score(y[idx], s[idx]) < MIN_VIEW_AUC:
                    entry.setdefault("unsupported_views", []).append(v)
        thresholds[finding] = entry

        for v in VIEWS:
            idx = test[view[test] == v]
            if y[idx].sum() == 0:
                continue
            single = sensitivity_specificity(y[idx], s[idx], entry["any"])
            per_view = sensitivity_specificity(y[idx], s[idx], entry.get(v, entry["any"]))
            rows.append({
                "Finding": finding, "View": v, "Test positives": int(y[idx].sum()),
                "Single threshold: sens / spec": f"{single[0]:.2f} / {single[1]:.2f}",
                "Per-view threshold: sens / spec": f"{per_view[0]:.2f} / {per_view[1]:.2f}",
                "Own view threshold": "yes" if v in entry else "no (fallback)",
                "Reported on this view": "no (AUC < 0.60)" if v in entry.get("unsupported_views", []) else "yes",
            })

    Path("service").mkdir(exist_ok=True)
    payload = {"model": "densenet121-res224-chex", "target_sensitivity": args.target,
               "thresholds": thresholds}
    Path("service/thresholds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = (f"# Triage thresholds: single vs per-view\n\n"
          f"Thresholds tuned for {args.target:.0%} sensitivity on half of the patients "
          f"({len(np.unique(labels['patient_id'].to_numpy()[tune]))}), evaluated on the other half "
          f"({len(np.unique(labels['patient_id'].to_numpy()[test]))}). sens = sensitivity, "
          f"spec = specificity. A finding is not reported on a view where its tuning-half AUC "
          f"is below {MIN_VIEW_AUC:.2f}.\n\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    Path("results").mkdir(exist_ok=True)
    Path("results/thresholds.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

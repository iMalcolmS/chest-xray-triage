"""Can a little local data improve an external model? And how much does leakage inflate it?

For each NIH finding, compares on the same images (out-of-fold, split by patient):
  zero_shot    the CheXpert model as-is
  recal        a small head on the CheXpert model's 11 outputs only: re-maps what
               CheXpert already detects to NIH's labels (e.g. Lung Lesion -> Mass)
  stacked      a head on the 1024 image features + those 11 outputs
  leaky        "stacked" again but split by image, so a patient's images can sit
               on both sides of the split; the gap to "stacked" is the leakage
  nih          the NIH-trained model (internal reference, optimistic)
Each head's L2 strength is tuned by nested cross-validation.

Usage:  python -m scripts.adapt     (needs extract_features and external_validation)
Writes: outputs/adaptation.csv, outputs/adaptation.md
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv

from cxr.adapt import out_of_fold_scores
from cxr.data import NIH_LABELS, load_labels
from cxr.metrics import auc_with_patient_bootstrap

MIN_POSITIVES = 20
WEIGHTS = "densenet121-res224-chex"
LABELS = {
    "zero_shot": "CheXpert zero-shot",
    "recal": "Recalibrated (11 CheXpert outputs)",
    "stacked": "Stacked head (features + outputs)",
    "leaky": "Stacked, image split (leaky)",
    "nih": "NIH model (internal ref.)",
}


def chexpert_logits(features):
    """The CheXpert model's raw outputs, recomputed from the cached features.

    Its classifier is a single linear layer on these features, so no image
    needs to be read again. Untrained outputs ("") are dropped.
    """
    model = xrv.models.DenseNet(weights=WEIGHTS).eval()
    with torch.no_grad():
        logits = model.classifier(torch.from_numpy(features)).numpy()
    keep = [j for j, name in enumerate(model.pathologies) if name]
    return logits[:, keep], [model.pathologies[j] for j in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.out)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    order = pd.read_csv(out / "features_index.csv")["image"]
    assert list(order) == list(labels["image"]), "features and labels are out of order"
    features = np.load(out / "features_chex.npy")
    preds = pd.read_csv(out / "predictions.csv")
    assert list(preds["image"]) == list(labels["image"]), "predictions out of order"
    groups = labels["patient_id"].to_numpy()

    logits, logit_names = chexpert_logits(features)
    print(f"CheXpert outputs used as inputs: {', '.join(logit_names)}")
    stacked_X = np.hstack([features, logits])

    rows, table = [], []
    for finding in NIH_LABELS:
        y = labels[finding].to_numpy()
        if y.sum() < MIN_POSITIVES:
            continue
        start = time.time()
        scores, chosen = {}, {}
        if f"chex:{finding}" in preds:
            scores["zero_shot"] = preds[f"chex:{finding}"].to_numpy()
        scores["recal"], chosen["recal"] = out_of_fold_scores(logits, y, groups)
        scores["stacked"], chosen["stacked"] = out_of_fold_scores(stacked_X, y, groups)
        scores["leaky"], chosen["leaky"] = out_of_fold_scores(stacked_X, y, None)
        if f"nih:{finding}" in preds:
            scores["nih"] = preds[f"nih:{finding}"].to_numpy()

        cells = {"Finding": finding, "Positives": int(y.sum())}
        for method, s in scores.items():
            auc, low, high = auc_with_patient_bootstrap(y, s, groups, n_boot=args.n_boot)
            rows.append({"finding": finding, "method": method, "auc": auc, "ci_low": low,
                         "ci_high": high, "chosen_C": chosen.get(method)})
            cells[LABELS[method]] = f"{auc:.3f} ({low:.3f}–{high:.3f})"
        table.append(cells)
        print(f"{finding:<20} done in {time.time() - start:.0f}s")

    results = pd.DataFrame(rows)
    results.to_csv(out / "adaptation.csv", index=False)

    wide = pd.DataFrame(table).fillna("–")
    wide = wide[["Finding", "Positives"] + [v for v in LABELS.values() if v in wide.columns]]

    auc = results.pivot(index="finding", columns="method", values="auc")
    covered = auc["zero_shot"].dropna().index
    lines = []
    for method in ("recal", "stacked"):
        gain = auc.loc[covered, method] - auc.loc[covered, "zero_shot"]
        lines.append(f"Mean AUC change vs zero-shot, {LABELS[method]}: {gain.mean():+.3f} "
                     f"(better on {(gain > 0).sum()} of {len(gain)} findings).")
    leak = auc["leaky"] - auc["stacked"]
    lines.append(f"Mean AUC inflation from splitting by image instead of patient: "
                 f"{leak.mean():+.3f} (largest: {leak.idxmax()} {leak.max():+.3f}).")

    md = (f"# Adapting the CheXpert model to NIH data\n\n"
          f"{len(labels)} images from {labels['patient_id'].nunique()} patients. Out-of-fold "
          f"AUC from 5-fold cross-validation split by patient (L2 strength tuned by nested "
          f"CV), 95% CI from a patient-level bootstrap.\n\n"
          + wide.to_markdown(index=False) + "\n\n" + "\n\n".join(lines) + "\n")
    (out / "adaptation.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()

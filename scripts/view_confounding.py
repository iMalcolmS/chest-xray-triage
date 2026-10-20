"""Is the CheXpert model partly detecting "portable AP film" rather than disease?

The Grad-CAM audit found that 90-97% of the model's most confident Cardiomegaly
and Effusion calls are AP (portable) views, against ~40% AP in the sample.
This checks it with every image:
  * view alone      AUC of the rule "AP = positive": how much the view leaks the label
  * overall AUC     the model on all images
  * within PA / AP  the model inside each view; if it leaned on the view,
                    these drop below the overall AUC
  * AP share        among the model's top 5% scores vs the whole sample

Usage:  python -m scripts.view_confounding     (needs external_validation)
Writes: results/view_confounding.md, results/view_confounding.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cxr.data import load_labels
from cxr.metrics import auc_with_patient_bootstrap

FINDINGS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Pneumonia", "Pneumothorax")


def auc_cell(y, s, groups, n_boot):
    if len(y) == 0 or y.min() == y.max() or y.sum() < 10:
        return "–", np.nan
    auc, low, high = auc_with_patient_bootstrap(y, s, groups, n_boot=n_boot)
    return f"{auc:.3f} ({low:.3f}–{high:.3f})", auc


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    data_dir, results_dir = Path(args.data_dir), Path(args.results)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    preds = pd.read_csv(Path(args.out) / "predictions.csv")
    assert list(preds["image"]) == list(labels["image"]), "predictions out of order"
    is_ap = (labels["view"] == "AP").to_numpy()
    groups = labels["patient_id"].to_numpy()

    rows, table = [], []
    for finding in FINDINGS:
        y = labels[finding].to_numpy()
        s = preds[f"chex:{finding}"].to_numpy()
        top = s >= np.quantile(s, 0.95)
        cells = {"Finding": finding, "Positives (AP/PA)": f"{y[is_ap].sum()}/{y[~is_ap].sum()}"}
        record = {"finding": finding}
        for name, mask, score in [
            ("View alone (AP = positive)", np.ones_like(y, bool), is_ap.astype(float)),
            ("Model, all images", np.ones_like(y, bool), s),
            ("Model, PA only", ~is_ap, s),
            ("Model, AP only", is_ap, s),
        ]:
            cells[name], record[name] = auc_cell(y[mask], score[mask], groups[mask], args.n_boot)
        cells["AP share: top 5% scores"] = f"{is_ap[top].mean():.0%}"
        cells["AP share: positives"] = f"{is_ap[y == 1].mean():.0%}"
        record["ap_share_top5"] = is_ap[top].mean()
        record["ap_share_positives"] = is_ap[y == 1].mean()
        rows.append(record)
        table.append(cells)

    pd.DataFrame(rows).to_csv(results_dir / "view_confounding.csv", index=False)
    md = ("# View confounding: AP (portable) vs PA films\n\n"
          f"{len(labels)} images, {is_ap.mean():.0%} AP. AUC with 95% CI from a patient-level "
          "bootstrap; findings with fewer than 10 positives in a view show \"–\".\n\n"
          + pd.DataFrame(table).to_markdown(index=False) + "\n")
    (results_dir / "view_confounding.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()

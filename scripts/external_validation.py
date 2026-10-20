"""External validation: do chest X-ray models trained at other hospitals work on NIH data?

Models (pretrained DenseNet121 from torchxrayvision):
  chex      trained on CheXpert (Stanford)      -> external
  mimic_ch  trained on MIMIC-CXR (Boston)       -> external
  nih       trained on NIH ChestX-ray14         -> internal reference only: it may
            have seen these very images in training, so its AUC is optimistic.

Usage:
  python -m scripts.external_validation --limit 64      # quick smoke test
  python -m scripts.external_validation                 # full run (~10-20 min on CPU)

Writes:
  outputs/predictions.csv          one row per image, one column per model:finding
  outputs/external_validation.csv  AUC with 95% CI per model and finding
  outputs/external_validation.md   the same as a table for the README
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torchxrayvision as xrv
from tqdm import tqdm

from cxr.data import NIH_LABELS, CXRDataset, load_labels
from cxr.metrics import auc_with_patient_bootstrap

MODELS = {
    "chex": "densenet121-res224-chex",
    "mimic_ch": "densenet121-res224-mimic_ch",
    "nih": "densenet121-res224-nih",
}
EXTERNAL = {"chex", "mimic_ch"}
MIN_POSITIVES = 20  # below this an AUC is too unstable to report


def predict(labels, images_dir, batch_size, workers):
    models = {name: xrv.models.DenseNet(weights=w).eval() for name, w in MODELS.items()}
    loader = torch.utils.data.DataLoader(
        CXRDataset(labels, images_dir), batch_size=batch_size, num_workers=workers
    )
    columns = {}
    with torch.no_grad():
        for images, idx in tqdm(loader, desc="Predicting", unit="batch"):
            for name, model in models.items():
                out = model(images).numpy()
                for j, finding in enumerate(model.pathologies):
                    if finding in NIH_LABELS:  # skip untrained ("") and non-NIH outputs
                        columns.setdefault(f"{name}:{finding}", {}).update(
                            zip(idx.tolist(), out[:, j].tolist())
                        )
    preds = pd.DataFrame({col: pd.Series(vals) for col, vals in columns.items()}).sort_index()
    return pd.concat([labels[["image"]], preds], axis=1)


def evaluate(labels, preds, n_boot):
    rows = []
    for col in preds.columns.drop("image"):
        model, finding = col.split(":")
        n_pos = int(labels[finding].sum())
        if n_pos < MIN_POSITIVES:
            continue
        auc, low, high = auc_with_patient_bootstrap(
            labels[finding], preds[col], labels["patient_id"], n_boot=n_boot
        )
        rows.append({"model": model, "external": model in EXTERNAL, "finding": finding,
                     "n_pos": n_pos, "auc": auc, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def to_markdown(results, n_images, n_patients):
    table = results.assign(cell=lambda d: d.apply(
        lambda r: f"{r.auc:.3f} ({r.ci_low:.3f}–{r.ci_high:.3f})", axis=1))
    wide = table.pivot(index=["finding", "n_pos"], columns="model", values="cell")
    # A model with no output for a finding (e.g. Infiltration in CheXpert) shows "–".
    wide = wide[[m for m in MODELS if m in wide.columns]].fillna("–").reset_index()
    wide = wide.rename(columns={"finding": "Finding", "n_pos": "Positives",
                                "chex": "CheXpert model (external)",
                                "mimic_ch": "MIMIC model (external)",
                                "nih": "NIH model (internal ref.)"})
    header = (f"# External validation on NIH ChestX-ray14 sample\n\n"
              f"{n_images} images from {n_patients} patients. AUC with 95% CI from a "
              f"patient-level bootstrap. Findings with fewer than {MIN_POSITIVES} "
              f"positives are not reported.\n\n")
    return header + wide.to_markdown(index=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N images")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--recompute", action="store_true", help="ignore cached predictions")
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(exist_ok=True)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    if args.limit:
        labels = labels.head(args.limit)
    print(f"{len(labels)} images from {labels['patient_id'].nunique()} patients")

    pred_path = out / ("predictions.csv" if not args.limit else f"predictions_{args.limit}.csv")
    if pred_path.exists() and not args.recompute:
        preds = pd.read_csv(pred_path)
        print(f"Using cached predictions from {pred_path}")
    else:
        preds = predict(labels, data_dir / "images", args.batch_size, args.workers)
        preds.to_csv(pred_path, index=False)

    results = evaluate(labels, preds, args.n_boot)
    if results.empty:
        print(f"No finding has {MIN_POSITIVES}+ positives in this subset; run without --limit.")
        return
    results.to_csv(out / "external_validation.csv", index=False)
    md = to_markdown(results, len(labels), labels["patient_id"].nunique())
    (out / "external_validation.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()

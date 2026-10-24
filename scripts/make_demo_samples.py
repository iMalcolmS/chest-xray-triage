"""Pick a few NIH images and save them as DICOM samples for the public demo.

NIH ChestX-ray14 may be redistributed with attribution (NIH Clinical Center,
Wang et al., CVPR 2017). Only a handful of images are copied, with patient IDs
replaced, so the demo works without anyone needing the dataset.

Usage:  python -m scripts.make_demo_samples
Writes: demo/samples/*.dcm and demo/samples/samples.json (committed)
"""

import json
from pathlib import Path

from cxr.data import NIH_LABELS, load_labels
from scripts.make_dicom import to_dicom

# (file name, view, finding to look for or None for "No Finding")
WANTED = [
    ("ap_cardiomegaly", "AP", "Cardiomegaly"),
    ("pa_effusion", "PA", "Effusion"),
    ("pa_pneumothorax", "PA", "Pneumothorax"),
    ("pa_no_finding", "PA", None),
]


def main():
    data_dir = Path("data/nih_sample")
    out = Path("demo/samples")
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    n_findings = labels[NIH_LABELS].sum(axis=1)

    meta = []
    for k, (name, view, finding) in enumerate(WANTED):
        rows = labels[labels.view == view]
        if finding:
            # a clear example: the finding present, with at most one other label
            rows = rows[(rows[finding] == 1) & (n_findings[rows.index] <= 2)]
        else:
            rows = rows[n_findings[rows.index] == 0]
        row = rows.iloc[0]
        to_dicom(data_dir / "images" / row.image, view, out / f"{name}.dcm", 900 + k)
        truth = [c for c in NIH_LABELS if row[c] == 1] or ["No Finding"]
        meta.append({"file": f"{name}.dcm", "view": view, "nih_labels": truth})
        print(f"{name}.dcm  {view}  {', '.join(truth)}")

    (out / "samples.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

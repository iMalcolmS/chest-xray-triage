"""Grad-CAM audit: where does the CheXpert model look when it is confident?

For each audited finding, takes the N most confident true positives and the N
most confident false positives (the model's worst mistakes), computes Grad-CAM
heatmaps and measures how much of each heatmap falls inside the chest
(lungs + heart + mediastinum, from torchxrayvision's segmentation model).

"Lift" compares that share with the chest's share of the image: 1.0 means the
heatmap is no more focused on the chest than a uniform one would be.

Usage:  python -m scripts.gradcam_audit            (needs extract_features + external_validation)
Writes: results/gradcam_audit.md, results/gradcam_audit.csv, results/gradcam_<finding>.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchxrayvision as xrv  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cxr.data import IMAGE_SIZE, load_labels  # noqa: E402
from cxr.explain import chest_masks, fraction_in_mask, gradcam  # noqa: E402

FINDINGS = ("Cardiomegaly", "Effusion", "Pneumothorax")
WEIGHTS = "densenet121-res224-chex"


def select(labels, preds, finding, n):
    """Indices of the n most confident true positives and false positives.

    At most one image per patient (their highest-scoring one), so repeat films
    of the same patient cannot fill the sample and the gallery.
    """
    score = preds[f"chex:{finding}"]
    pos = labels[finding] == 1

    def top_per_patient(mask):
        ranked = score[mask].sort_values(ascending=False)
        first = ~labels.loc[ranked.index, "patient_id"].duplicated()
        return ranked.index[first.to_numpy()][:n]

    tp, fp = top_per_patient(pos), top_per_patient(~pos)
    return [("true positive", i) for i in tp] + [("false positive", i) for i in fp]


def gallery(path, finding, images, heatmaps, masks, groups, scores):
    """Four confident true positives (top row) and four false positives (bottom)."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
    for row, group in enumerate(("true positive", "false positive")):
        idx = [i for i, g in enumerate(groups) if g == group][:4]
        for col, ax in enumerate(axes[row]):
            ax.axis("off")
            if col >= len(idx):
                continue
            i = idx[col]
            ax.imshow(images[i], cmap="gray")
            ax.imshow(heatmaps[i], cmap="jet", alpha=0.35)
            ax.contour(masks[i], levels=[0.5], colors="white", linewidths=0.8)
            ax.set_title(f"{group}\nscore {scores[i]:.2f}", fontsize=9)
    fig.suptitle(f"{finding}: Grad-CAM (CheXpert model) with chest outline in white")
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--results", default="results")
    parser.add_argument("--n", type=int, default=30, help="examples per group and finding")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    data_dir, out, results_dir = Path(args.data_dir), Path(args.out), Path(args.results)
    results_dir.mkdir(exist_ok=True)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    preds = pd.read_csv(out / "predictions.csv")
    assert list(preds["image"]) == list(labels["image"]), "predictions out of order"
    cache = np.load(data_dir.parent / "cache" / "images_224.npy", mmap_mode="r")

    model = xrv.models.DenseNet(weights=WEIGHTS).eval()
    seg = xrv.baseline_models.chestx_det.PSPNet().eval()

    rows = []
    for finding in FINDINGS:
        out_index = list(model.pathologies).index(finding)
        chosen = select(labels, preds, finding, args.n)
        heat_all, mask_all, img_all = [], [], []
        for start in tqdm(range(0, len(chosen), args.batch_size), desc=finding):
            batch = chosen[start:start + args.batch_size]
            idx = [i for _, i in batch]
            x = torch.from_numpy(cache[idx].astype(np.float32))
            heat = gradcam(model, x, out_index)
            masks = chest_masks(seg, x, IMAGE_SIZE)
            heat_all.append(heat)
            mask_all.append(masks)
            img_all.append(x[:, 0].numpy())
        heat_all = np.concatenate(heat_all)
        mask_all = np.concatenate(mask_all)
        img_all = np.concatenate(img_all)

        inside = fraction_in_mask(heat_all, mask_all)
        area = mask_all.mean(axis=(1, 2))
        for (group, i), f_in, a in zip(chosen, inside, area):
            rows.append({"finding": finding, "group": group, "image": labels.loc[i, "image"],
                         "view": labels.loc[i, "view"], "score": preds.loc[i, f"chex:{finding}"],
                         "cam_in_chest": f_in, "chest_area": a})
        gallery(results_dir / f"gradcam_{finding.lower()}.png", finding, img_all, heat_all,
                mask_all, [g for g, _ in chosen], [preds.loc[i, f"chex:{finding}"] for _, i in chosen])

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "gradcam_audit.csv", index=False)

    summary = (df.groupby(["finding", "group"], sort=False)
                 .agg(n=("image", "size"),
                      cam_in_chest=("cam_in_chest", "mean"),
                      chest_area=("chest_area", "mean"),
                      mostly_outside=("cam_in_chest", lambda s: (s < 0.5).mean()),
                      ap_share=("view", lambda s: (s == "AP").mean()))
                 .reset_index())
    summary["lift"] = summary["cam_in_chest"] / summary["chest_area"]
    table = pd.DataFrame({
        "Finding": summary["finding"],
        "Group": summary["group"],
        "Images": summary["n"],
        "Heatmap inside chest": (summary["cam_in_chest"] * 100).round(1).astype(str) + "%",
        "Chest share of image": (summary["chest_area"] * 100).round(1).astype(str) + "%",
        "Lift": summary["lift"].round(2),
        "Mostly outside chest": (summary["mostly_outside"] * 100).round(0).astype(int).astype(str) + "%",
        "AP (portable) views": (summary["ap_share"] * 100).round(0).astype(int).astype(str) + "%",
    })
    md = ("# Grad-CAM audit of the CheXpert model on NIH images\n\n"
          f"For each finding, the {args.n} most confident true positives and the {args.n} most "
          "confident false positives, one image per patient. \"Heatmap inside chest\" is the share of Grad-CAM weight "
          "inside the lungs, heart and mediastinum. \"Lift\" divides it by the chest's share of "
          "the image (1.0 = no more focused than a uniform heatmap). \"Mostly outside chest\" "
          "counts heatmaps with under half their weight inside.\n\n"
          + table.to_markdown(index=False) + "\n")
    (results_dir / "gradcam_audit.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()

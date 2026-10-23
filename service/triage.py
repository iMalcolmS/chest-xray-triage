"""Score an X-ray, apply view-aware thresholds, and draw a Grad-CAM overlay."""

import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchxrayvision as xrv  # noqa: E402

from cxr.explain import gradcam  # noqa: E402
from cxr.thresholds import pick  # noqa: E402

WEIGHTS = "densenet121-res224-chex"
THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


class Triage:
    def __init__(self, model=None, thresholds=None):
        self.model = (model or xrv.models.DenseNet(weights=WEIGHTS)).eval()
        if thresholds is None:
            thresholds = json.loads(THRESHOLDS_PATH.read_text())["thresholds"]
        self.thresholds = thresholds
        self.findings = [f for f in self.model.pathologies if f in thresholds]

    def predict(self, pixels, view):
        """Scores and flags for every finding that has a threshold."""
        with torch.no_grad():
            out = self.model(torch.from_numpy(pixels)[None]).numpy()[0]
        results = []
        for j, name in enumerate(self.model.pathologies):
            if name not in self.thresholds:
                continue
            threshold = pick(self.thresholds, name, view)
            results.append({"finding": name, "score": round(float(out[j]), 4),
                            "threshold": round(threshold, 4), "flagged": bool(out[j] >= threshold)})
        return sorted(results, key=lambda r: r["score"] - r["threshold"], reverse=True)

    def heatmap_png(self, pixels, finding):
        """Grad-CAM for one finding over the image, as a base64 PNG."""
        index = list(self.model.pathologies).index(finding)
        heat = gradcam(self.model, torch.from_numpy(pixels)[None], index)[0]
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
        ax.imshow(pixels[0], cmap="gray")
        ax.imshow(heat, cmap="jet", alpha=0.35)
        ax.set_title(f"Grad-CAM: {finding}", fontsize=9)
        ax.axis("off")
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def summary(results):
    flagged = [r["finding"] for r in results if r["flagged"]]
    return ("Flagged for radiologist review: " + ", ".join(flagged)) if flagged \
        else "No finding above triage thresholds"


"""Grad-CAM heatmaps and an anatomy check: does the model look inside the chest?

A heatmap alone is easy to over-read. Here every heatmap is scored against an
anatomical mask (lungs, heart, mediastinum) predicted by torchxrayvision's
chest segmentation model, so "where the model looks" becomes a number.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CHEST_STRUCTURES = ("Left Lung", "Right Lung", "Heart", "Mediastinum")


class Logits(torch.nn.Module):
    """A torchxrayvision DenseNet without its output calibration.

    The library applies a sigmoid and per-finding threshold scaling after the
    classifier; Grad-CAM is cleaner on the raw logits it starts from.
    """

    def __init__(self, densenet):
        super().__init__()
        self.densenet = densenet

    def forward(self, x):
        return self.densenet.classifier(self.densenet.features2(x))


def gradcam(model, images, output_index):
    """Heatmaps in [0, 1] at image resolution, shape (B, H, W), for one output."""
    wrapper = Logits(model).eval()
    with GradCAM(model=wrapper, target_layers=[model.features.denseblock4]) as cam:
        return cam(input_tensor=images,
                   targets=[ClassifierOutputTarget(output_index)] * len(images))


def chest_masks(seg_model, images, size):
    """Boolean (B, size, size) masks of lungs + heart + mediastinum."""
    keep = [seg_model.targets.index(name) for name in CHEST_STRUCTURES]
    with torch.no_grad():
        probs = torch.sigmoid(seg_model(images))[:, keep]
    mask = (probs > 0.5).any(dim=1, keepdim=True).float()
    mask = F.interpolate(mask, size=(size, size), mode="nearest")
    return mask[:, 0].numpy().astype(bool)


def fraction_in_mask(heatmaps, masks):
    """Share of each heatmap's total weight that falls inside its mask."""
    heatmaps = np.asarray(heatmaps, dtype=float)
    inside = (heatmaps * masks).sum(axis=(1, 2))
    total = heatmaps.sum(axis=(1, 2))
    return np.divide(inside, total, out=np.zeros_like(inside), where=total > 0)

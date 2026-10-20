"""Load the NIH ChestX-ray14 sample and preprocess images for torchxrayvision models.

The NIH labels were text-mined from radiology reports, so they are noisy
(Oakden-Rayner, 2019). They are still the standard benchmark, and every
result in this project should be read with that in mind.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchxrayvision as xrv
from skimage.io import imread

# The 14 findings in ChestX-ray14 ("No Finding" is the absence of all of them).
NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

IMAGE_SIZE = 224


def load_labels(csv_path, images_dir):
    """One row per image: ids, view, demographics and a 0/1 column per finding.

    Rows whose image file is missing are dropped, so a partial download still works.
    """
    raw = pd.read_csv(csv_path)
    images_dir = Path(images_dir)
    df = pd.DataFrame({
        "image": raw["Image Index"],
        "patient_id": raw["Patient ID"].astype(str),
        "view": raw["View Position"],
        "sex": raw["Patient Gender"],
        # Ages look like "060Y"; a handful are in months or days, which we leave as NaN.
        "age": pd.to_numeric(raw["Patient Age"].str.extract(r"^(\d+)Y$")[0], errors="coerce"),
    })
    findings = raw["Finding Labels"].str.split("|")
    for label in NIH_LABELS:
        df[label] = findings.apply(lambda f, label=label: int(label in f))
    exists = df["image"].apply(lambda name: (images_dir / name).is_file())
    return df[exists].reset_index(drop=True)


_transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(IMAGE_SIZE),
])


def preprocess(path):
    """Read a PNG and return a (1, 224, 224) float32 array in the [-1024, 1024] range
    that torchxrayvision models expect."""
    img = imread(path)
    if img.ndim == 3:  # a few NIH files are RGB/RGBA: drop alpha, average to grey
        img = img[..., :3].mean(axis=2)
    max_value = 65535 if img.max() > 255 else 255
    img = xrv.datasets.normalize(img.astype(np.float32), max_value)
    return _transform(img[None, ...]).astype(np.float32)


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, labels, images_dir):
        self.labels = labels.reset_index(drop=True)
        self.images_dir = Path(images_dir)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        name = self.labels.loc[i, "image"]
        return torch.from_numpy(preprocess(self.images_dir / name)), i

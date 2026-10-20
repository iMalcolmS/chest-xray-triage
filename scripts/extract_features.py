"""Extract 1024-d image features from the frozen CheXpert DenseNet, once.

It also caches the preprocessed 224x224 images (float16, ~560 MB) in
data/cache/, so later steps (Grad-CAM, the API tests) skip the slow
PNG decoding and resizing.

Usage:  python -m scripts.extract_features          (~25-35 min on an Intel Mac CPU)
Writes: outputs/features_chex.npy, data/cache/images_224.npy
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from tqdm import tqdm

from cxr.data import IMAGE_SIZE, CXRDataset, load_labels

WEIGHTS = "densenet121-res224-chex"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(exist_ok=True)
    cache_dir = data_dir.parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    n = len(labels)
    cache_path = cache_dir / "images_224.npy"
    images_cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float16, shape=(n, 1, IMAGE_SIZE, IMAGE_SIZE)
    )
    features = np.zeros((n, 1024), dtype=np.float32)

    model = xrv.models.DenseNet(weights=WEIGHTS).eval()
    loader = torch.utils.data.DataLoader(
        CXRDataset(labels, data_dir / "images"), batch_size=args.batch_size,
        num_workers=args.workers,
    )
    with torch.no_grad():
        for images, idx in tqdm(loader, desc="Extracting", unit="batch"):
            features[idx.numpy()] = model.features2(images).numpy()
            images_cache[idx.numpy()] = images.numpy().astype(np.float16)
    images_cache.flush()

    np.save(out / "features_chex.npy", features)
    labels[["image"]].to_csv(out / "features_index.csv", index=False)
    print(f"Saved features {features.shape} and image cache {cache_path}")


if __name__ == "__main__":
    main()

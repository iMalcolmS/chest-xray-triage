import numpy as np
import pandas as pd
from skimage.io import imsave

from cxr.data import NIH_LABELS, load_labels, preprocess


def _write_sample(tmp_path, rgba=False):
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(0)
    shape = (300, 260, 4) if rgba else (300, 260)
    imsave(images / "a.png", rng.integers(0, 256, shape).astype(np.uint8), check_contrast=False)
    pd.DataFrame({
        "Image Index": ["a.png", "missing.png"],
        "Finding Labels": ["Effusion|Pneumonia", "No Finding"],
        "Patient ID": [12, 13],
        "Patient Age": ["060Y", "003M"],
        "Patient Gender": ["M", "F"],
        "View Position": ["PA", "AP"],
    }).to_csv(tmp_path / "labels.csv", index=False)
    return images


def test_load_labels_one_hot_and_drops_missing_files(tmp_path):
    images = _write_sample(tmp_path)
    df = load_labels(tmp_path / "labels.csv", images)
    assert list(df["image"]) == ["a.png"]
    row = df.iloc[0]
    assert row["Effusion"] == 1 and row["Pneumonia"] == 1
    assert sum(row[label] for label in NIH_LABELS) == 2
    assert row["age"] == 60 and row["patient_id"] == "12"


def test_preprocess_shape_and_range(tmp_path):
    for rgba in (False, True):
        sub = tmp_path / ("rgba" if rgba else "grey")
        sub.mkdir()
        images = _write_sample(sub, rgba=rgba)
        x = preprocess(images / "a.png")
        assert x.shape == (1, 224, 224)
        assert x.dtype == np.float32
        assert -1024 <= x.min() and x.max() <= 1024

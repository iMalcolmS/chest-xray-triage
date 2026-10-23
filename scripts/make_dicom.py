"""Wrap a few NIH PNGs as DICOM files, with ViewPosition from the labels CSV.

The NIH sample ships as PNG, but hospitals send DICOM. These files let you
test the API the way a PACS would call it. Patient IDs are replaced.

Usage:  python -m scripts.make_dicom --n 6
Writes: data/dicom_samples/*.dcm   (git-ignored, like the rest of data/)
"""

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
from skimage.io import imread

from cxr.data import load_labels


def to_dicom(png_path, view, out_path, index):
    img = imread(png_path)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(np.uint8)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.PatientID = f"ANON-{index:04d}"
    ds.PatientName = "Anonymous"
    ds.Modality = "DX"
    ds.BodyPartExamined = "CHEST"
    ds.ViewPosition = view
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = img.tobytes()
    ds.save_as(out_path, enforce_file_format=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/nih_sample")
    parser.add_argument("--out", default="data/dicom_samples")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = load_labels(data_dir / "sample_labels.csv", data_dir / "images")
    # Half AP, half PA, so both thresholds get exercised
    picks = (list(labels[labels.view == "AP"].index[: args.n // 2])
             + list(labels[labels.view == "PA"].index[: args.n - args.n // 2]))
    for k, i in enumerate(picks):
        row = labels.loc[i]
        findings = [c for c in labels.columns[5:] if row[c] == 1] or ["No Finding"]
        name = f"sample_{k:02d}_{row.view}.dcm"
        to_dicom(data_dir / "images" / row.image, row.view, out / name, k)
        print(f"{name}: labels {', '.join(findings)}")


if __name__ == "__main__":
    main()

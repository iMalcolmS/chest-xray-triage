"""Read an uploaded chest X-ray (DICOM or PNG/JPEG) into model input.

Only the pixels and the view position are used. Patient identifiers in the
DICOM header are never read into the response or the logs.
"""

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from skimage.io import imread

from cxr.data import preprocess_array


class UnsupportedImage(ValueError):
    pass


@dataclass
class XRay:
    pixels: np.ndarray          # (1, 224, 224) float32, model-ready
    view: str | None            # "AP", "PA" or None if unknown
    source: str                 # "dicom" or "image"


def _from_dicom(data):
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and ds.get("SamplesPerPixel", 1) == 1:
        arr = arr[0]  # multi-frame: use the first frame
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr  # MONOCHROME1 stores bright as low values
    max_value = 2 ** int(ds.get("BitsStored", 8)) - 1
    view = str(ds.get("ViewPosition", "")).strip().upper() or None
    return XRay(preprocess_array(arr, max_value), view, "dicom")


def read_xray(data: bytes, view: str | None = None) -> XRay:
    """Parse DICOM if possible, otherwise a regular image.

    For PNG/JPEG the view must come from the caller; for DICOM the header's
    ViewPosition is used unless the caller overrides it.
    """
    try:
        xray = _from_dicom(data)
    except (InvalidDicomError, AttributeError, KeyError):
        try:
            img = imread(io.BytesIO(data))
        except Exception as error:  # noqa: BLE001 - any decoder failure means unsupported
            raise UnsupportedImage("Upload a DICOM, PNG or JPEG chest X-ray") from error
        xray = XRay(preprocess_array(img), None, "image")
    if view:
        xray.view = view.strip().upper()
    if xray.view not in (None, "AP", "PA"):
        xray.view = None  # lateral or unknown views: use the all-views threshold
    return xray

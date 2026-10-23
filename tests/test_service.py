import io

import numpy as np
import torchxrayvision as xrv
from fastapi.testclient import TestClient
from skimage.io import imsave

from scripts.make_dicom import to_dicom
from service.app import create_app
from service.inputs import read_xray
from service.triage import Triage

THRESHOLDS = {"Effusion": {"any": 0.5, "AP": 0.9, "PA": 0.1},
              "Cardiomegaly": {"any": 0.5},
              "Pneumothorax": {"any": 0.5, "AP": 0.05, "unsupported_views": ["AP"]}}


def _client():
    model = xrv.models.DenseNet(weights=None)  # untrained: no download in tests
    model.pathologies = list(xrv.datasets.default_pathologies)
    return TestClient(create_app(Triage(model=model, thresholds=THRESHOLDS)))


def _png_bytes():
    img = np.random.default_rng(0).integers(0, 256, (300, 260)).astype(np.uint8)
    buffer = io.BytesIO()
    imsave(buffer, img, format="png", check_contrast=False)
    return buffer.getvalue()


def _dicom_bytes(tmp_path, view):
    png = tmp_path / "x.png"
    png.write_bytes(_png_bytes())
    out = tmp_path / "x.dcm"
    to_dicom(png, view, out, index=1)
    return out.read_bytes()


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_dicom_view_is_read_and_used(tmp_path):
    r = _client().post("/predict", files={"file": ("x.dcm", _dicom_bytes(tmp_path, "AP"))})
    body = r.json()
    assert r.status_code == 200
    assert body["view"] == "AP" and body["view_source"] == "dicom header"
    effusion = next(f for f in body["findings"] if f["finding"] == "Effusion")
    assert effusion["threshold"] == 0.9  # the AP-specific threshold
    assert body["fhir"]["resourceType"] == "DiagnosticReport"
    assert len(body["fhir"]["contained"]) == len(body["findings"]) == 3
    ptx = next(f for f in body["findings"] if f["finding"] == "Pneumothorax")
    assert ptx["threshold"] is None and ptx["flagged"] is False  # not reported on AP


def test_unsupported_view_still_reported_on_other_view(tmp_path):
    r = _client().post("/predict", files={"file": ("x.dcm", _dicom_bytes(tmp_path, "PA"))})
    ptx = next(f for f in r.json()["findings"] if f["finding"] == "Pneumothorax")
    assert ptx["threshold"] == 0.5 and ptx["status"] == "reported"


def test_root_redirects_to_docs():
    r = _client().get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_png_needs_view_from_request():
    r = _client().post("/predict?view=pa&heatmap=true", files={"file": ("x.png", _png_bytes())})
    body = r.json()
    assert body["view"] == "PA" and body["view_source"] == "request"
    assert body["heatmap_png_base64"].startswith("iVBOR")  # PNG signature in base64


def test_unknown_view_falls_back(tmp_path):
    xray = read_xray(_dicom_bytes(tmp_path, "LL"))
    assert xray.view is None


def test_rejects_non_image():
    r = _client().post("/predict", files={"file": ("x.txt", b"not an image")})
    assert r.status_code == 415


def test_no_patient_id_in_response(tmp_path):
    r = _client().post("/predict", files={"file": ("x.dcm", _dicom_bytes(tmp_path, "PA"))})
    assert "ANON-0001" not in r.text

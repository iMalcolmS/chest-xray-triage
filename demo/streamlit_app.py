"""Public demo of the chest X-ray triage service (Streamlit Community Cloud).

It uses the same code as the API (service.triage), so what you see here is
exactly what /predict returns: view-aware thresholds, findings the model is
too weak to report on a view, a Grad-CAM overlay and a FHIR DiagnosticReport.
"""

import base64
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so "cxr" and "service" import when run from anywhere

from service.fhir import RESEARCH_NOTE, diagnostic_report  # noqa: E402
from service.inputs import UnsupportedImage, read_xray  # noqa: E402
from service.triage import WEIGHTS, Triage, summary  # noqa: E402

SAMPLES_DIR = ROOT / "demo" / "samples"
REPO = "https://github.com/iMalcolmS/chest-xray-triage"

st.set_page_config(page_title="Chest X-ray triage", layout="wide")


@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def load_triage():
    return Triage()


def load_samples():
    path = SAMPLES_DIR / "samples.json"
    return json.loads(path.read_text()) if path.exists() else []


st.title("Chest X-ray triage")
st.caption(
    "Research prototype: a CheXpert-trained DenseNet with thresholds chosen separately for "
    "AP (portable) and PA films, after external validation showed the model over-flags AP films. "
    f"[Code, evaluation and limitations]({REPO})"
)
st.warning(RESEARCH_NOTE)

samples = load_samples()
left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("1. Choose an X-ray")
    source = st.radio("Source", ["Sample from NIH ChestX-ray14", "Upload my own"],
                      horizontal=True, label_visibility="collapsed")
    data, truth = None, None
    if source.startswith("Sample") and samples:
        options = {f"{s['view']} film, NIH labels: {', '.join(s['nih_labels'])}": s for s in samples}
        choice = options[st.selectbox("Sample", list(options))]
        data = (SAMPLES_DIR / choice["file"]).read_bytes()
        truth = choice["nih_labels"]
    elif source.startswith("Upload"):
        upload = st.file_uploader("DICOM, PNG or JPEG", type=["dcm", "dicom", "png", "jpg", "jpeg"])
        if upload:
            data = upload.getvalue()

    view_choice = st.selectbox(
        "View position",
        ["Read from DICOM header", "AP (portable)", "PA"],
        help="PNG/JPEG files have no header, so pick the view yourself.",
    )
    show_heatmap = st.toggle("Show Grad-CAM for the top finding", value=True)

if data is None:
    with right:
        st.info("Pick a sample or upload an image to see the triage result.")
    st.stop()

override = None if view_choice.startswith("Read") else view_choice.split()[0]
try:
    xray = read_xray(data, override)
except UnsupportedImage as error:
    st.error(str(error))
    st.stop()

triage = load_triage()
results = triage.predict(xray.pixels, xray.view)
conclusion = summary(results)

with left:
    # model input is in [-1024, 1024]; rescale to [0, 1] for display
    st.image((xray.pixels[0] + 1024) / 2048, clamp=True, width=360,
             caption=f"Model input (224 px), view: {xray.view or 'unknown'}")

with right:
    st.subheader("2. Triage result")
    flagged = [r for r in results if r["flagged"]]
    (st.error if flagged else st.success)(conclusion)
    if truth:
        st.caption(f"NIH labels for this image (text-mined, can be wrong): {', '.join(truth)}")

    table = pd.DataFrame([{
        "Finding": r["finding"],
        "Score": r["score"],
        "Threshold": "–" if r["threshold"] is None else r["threshold"],
        "Result": "Flagged" if r["flagged"] else
                  ("Not reported on this view" if r["threshold"] is None else "Below threshold"),
    } for r in results])
    st.markdown(table.to_markdown(index=False))  # markdown table: no pyarrow needed
    if any(r["threshold"] is None for r in results):
        st.caption("Not reported: on this view the model's AUC was below 0.60 in validation, "
                   "so any flag would be close to guessing.")

    reported = [r for r in results if r["threshold"] is not None]
    if show_heatmap and reported:
        png = base64.b64decode(triage.heatmap_png(xray.pixels, reported[0]["finding"]))
        st.image(png, caption=f"Where the model looked for {reported[0]['finding']}", width=320)

    with st.expander("FHIR R4 DiagnosticReport (what an EHR would store)"):
        st.json(diagnostic_report(results, xray.view, conclusion, f"{WEIGHTS}@demo"))

st.divider()
st.caption(
    "Data: NIH Clinical Center ChestX-ray14 (Wang et al., CVPR 2017). Model: torchxrayvision "
    "(Cohen et al.). Thresholds target 80% sensitivity per view on held-out NIH patients."
)

"""Chest X-ray triage API.

  GET  /health      liveness and model status
  POST /predict     upload a DICOM/PNG/JPEG; returns scores, flags, a FHIR
                    DiagnosticReport and optionally a Grad-CAM overlay
Run:   uvicorn service.app:app --port 8000      then open http://localhost:8000/docs
"""

import json
import logging
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import RedirectResponse

from service.fhir import RESEARCH_NOTE, diagnostic_report
from service.inputs import UnsupportedImage, read_xray
from service.triage import WEIGHTS, Triage, summary

VERSION = "0.1.0"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024

log = logging.getLogger("cxr-triage")
logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app(triage: Triage | None = None) -> FastAPI:
    app = FastAPI(title="Chest X-ray triage", version=VERSION, description=RESEARCH_NOTE)
    state = {"triage": triage}

    def get_triage():
        if state["triage"] is None:  # load once, on first use
            state["triage"] = Triage()
        return state["triage"]

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": VERSION, "model": WEIGHTS,
                "model_loaded": state["triage"] is not None}

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(..., description="DICOM, PNG or JPEG chest X-ray"),
        view: str | None = Query(None, description="AP or PA; overrides the DICOM header"),
        heatmap: bool = Query(False, description="include a Grad-CAM overlay for the top finding"),
    ):
        start = time.perf_counter()
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large (max 30 MB)")
        try:
            xray = read_xray(data, view)
        except UnsupportedImage as error:
            raise HTTPException(415, str(error)) from error

        model = get_triage()
        results = model.predict(xray.pixels, xray.view)
        conclusion = summary(results)
        body = {
            "view": xray.view or "unknown",
            "view_source": "request" if view else ("dicom header" if xray.view else "unknown"),
            "findings": results,
            "conclusion": conclusion,
            "fhir": diagnostic_report(results, xray.view, conclusion, f"{WEIGHTS}@{VERSION}"),
            "disclaimer": RESEARCH_NOTE,
        }
        if heatmap and results and results[0]["threshold"] is not None:
            body["heatmap_png_base64"] = model.heatmap_png(xray.pixels, results[0]["finding"])

        # One structured log line per request: no identifiers, only what monitoring needs.
        log.info(json.dumps({
            "event": "predict", "latency_ms": round(1000 * (time.perf_counter() - start)),
            "source": xray.source, "view": xray.view,
            "flagged": [r["finding"] for r in results if r["flagged"]],
            "max_score": max((r["score"] for r in results), default=None),
        }))
        return body

    return app


app = create_app()

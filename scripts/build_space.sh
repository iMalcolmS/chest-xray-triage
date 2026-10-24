#!/usr/bin/env bash
# Assemble the files a Hugging Face Docker Space needs into ../cxr-triage-space.
# The Space has its own README (Hugging Face reads its settings from the YAML header),
# so we build a separate folder instead of pushing this whole repository.
set -euo pipefail
OUT="${1:-../cxr-triage-space}"
mkdir -p "$OUT"
rm -rf "$OUT/cxr" "$OUT/service"
cp -R cxr service "$OUT/"
cp Dockerfile requirements-service.txt .dockerignore "$OUT/"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
cat > "$OUT/README.md" <<'MD'
---
title: Chest X-ray Triage API
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: View-aware chest X-ray triage API with FHIR output
---

# Chest X-ray triage API (research prototype)

Upload a DICOM, PNG or JPEG chest X-ray to `/predict` (try it at `/docs`). Returns per-finding
scores with view-aware (AP/PA) thresholds, a FHIR R4 DiagnosticReport and an optional Grad-CAM
overlay. Not a medical device. Not for clinical use.

Code, evaluation and model card: https://github.com/iMalcolmS/chest-xray-triage
MD
echo "Space files ready in $OUT"

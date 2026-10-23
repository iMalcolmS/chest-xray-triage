"""Wrap triage results in a FHIR R4 DiagnosticReport so an EHR can store them.

Each finding becomes a contained Observation with its score and whether it
was flagged. The report is "preliminary" and labelled research-only: a triage
suggestion for a radiologist, never a final read.
"""

from datetime import datetime, timezone

RESEARCH_NOTE = "AI triage research prototype. Not a medical device. Not for clinical use."


def diagnostic_report(results, view, conclusion, model_version):
    observations = []
    for i, r in enumerate(results):
        observations.append({
            "resourceType": "Observation",
            "id": f"obs-{i}",
            "status": "preliminary",
            "code": {"text": r["finding"]},
            "valueQuantity": {"value": r["score"], "unit": "model score"},
            "interpretation": [{"text": "flagged" if r["flagged"] else
                                ("not reported" if r["threshold"] is None else "not flagged")}],
            "note": [{"text": r.get("status") if r["threshold"] is None
                      else f"Threshold {r['threshold']} for view {view or 'unknown'}"}],
        })
    return {
        "resourceType": "DiagnosticReport",
        "status": "preliminary",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
            "code": "RAD", "display": "Radiology"}]}],
        "code": {"text": "Chest X-ray AI triage"},
        "issued": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conclusion": f"{conclusion}. {RESEARCH_NOTE}",
        "contained": observations,
        "result": [{"reference": f"#{o['id']}"} for o in observations],
        "extension": [{"url": "urn:cxr-triage:model-version", "valueString": model_version}],
    }

# Model card: chest X-ray triage service

**Status:** research prototype. Not a medical device, not UKCA/CE marked, not for clinical use.
**Version:** 0.1.0 · **Owner:** Malcolm Sanha · **Code:** https://github.com/iMalcolmS/chest-xray-triage

This card follows the structure of Hugging Face model cards and adds the documents an NHS organisation would ask for before a pilot: an intended-use statement, a hazard log in the style of DCB0129, and notes against the NHS Digital Technology Assessment Criteria (DTAC). They are written as an exercise in how such a system would be assessed, not as a claim of compliance.

## 1. Model details

| | |
|---|---|
| Model | DenseNet121, `densenet121-res224-chex` from torchxrayvision (Cohen et al.), trained on CheXpert (Stanford). Not retrained here. |
| Input | One frontal chest radiograph (DICOM, PNG or JPEG), resized to 224 × 224 |
| Output | A score per finding for 7 findings (Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Pneumonia, Pneumothorax), a flag when the score is at or above the threshold for that finding and view, and a FHIR R4 `DiagnosticReport` (status `preliminary`) |
| What this project adds | External validation on NIH data, thresholds chosen separately for AP and PA films, suppression of findings the model cannot separate on a view, the API, and this documentation |

## 2. Intended use

**Intended purpose (as it would be written for a pilot):** to help prioritise a radiology reporting worklist by flagging frontal adult chest radiographs that may show one of seven findings, so that a radiologist reviews them sooner. Every image is still read by a radiologist; the output never replaces a report.

**Intended users:** radiologists and reporting radiographers, through a PACS or worklist integration.

**Out of scope:**

- Diagnosis, or any decision made without a radiologist reading the image
- Lateral films, children, and images other than frontal chest radiographs
- Findings outside the seven listed (for example nodules, masses, fractures, devices)
- Any population or site where it has not been validated locally (see section 4)

## 3. Data

- **External validation:** NIH ChestX-ray14 random sample, 5,606 images from 4,230 patients, 39% AP films. Labels were text-mined from reports and are known to be noisy (Oakden-Rayner, 2019).
- **Threshold selection:** patients split 50/50 (2,115 each). Thresholds chosen on one half, reported on the other, so no patient appears in both.
- **No training on patient data** in this project. The model weights are public; the NIH images are public and de-identified.

## 4. Performance

Full tables: [external validation](results/external_validation.md), [AP vs PA](results/view_confounding.md), [thresholds](results/thresholds.md), [Grad-CAM audit](results/gradcam_audit.md).

- **Discrimination on NIH (external site):** AUC 0.65–0.78 for the seven findings, below the NIH-trained reference for six of them.
- **By view:** AUC is 0.08–0.17 lower on AP (portable) films than on PA films, and the model scores AP films higher whether or not disease is present.
- **At the operating point** (80% target sensitivity, per-view thresholds, held-out patients):
  - sensitivity close to 0.80 on both views for most findings;
  - specificity mostly 0.2–0.6, so a large share of flagged films will be normal.
- **Not reported:** where the AUC on a view was below 0.60 on the tuning half, the finding is returned as "not reported on this view" instead of being flagged. The current list is in `service/thresholds.json`; on the full sample, Atelectasis (0.58) and Pneumothorax (0.59) on AP films fall below this bar.
- **Explanations:** only 37–57% of Grad-CAM weight falls inside the lungs, heart and mediastinum, and false positives look in the same places as true positives.

**Not yet evaluated:** performance by sex and age group, calibration, and behaviour on images from other hospitals, scanners or countries. These are prerequisites for any pilot.

## 5. Hazard log (DCB0129 style)

Severity and likelihood use a 1–5 scale (5 = worst). Risk = severity × likelihood, before and after controls. The scores are a first-pass engineering view; in a real deployment they would be set with a Clinical Safety Officer.

| ID | Hazard | Cause | Effect on patient | Controls in this build | Sev | Lik before → after |
|---|---|---|---|---|---|---|
| H1 | Missed finding (false negative) | Threshold set for 80% sensitivity: 1 in 5 cases can be missed | Delayed review if a flag is wrongly read as "all clear" | Radiologist reads every film; output is "preliminary" and labelled triage only; non-flagged does not mean normal | 4 | 3 → 2 |
| H2 | Alert fatigue from false positives | Specificity 0.2–0.6 at the operating point | Real urgent cases lost among false flags | Per-view thresholds; findings with AUC < 0.60 on a view are not reported; specificity documented | 3 | 4 → 3 |
| H3 | Wrong threshold applied | Missing or incorrect `ViewPosition` in the DICOM header | Over- or under-flagging | Unknown views fall back to the all-views threshold; the view and its source are returned in every response | 3 | 3 → 2 |
| H4 | Out-of-scope image | Lateral film, child, non-chest image | Meaningless output | Intended-use statement; lateral views are not given view-specific thresholds. **Gap:** no automatic image-type check | 4 | 3 → 3 |
| H5 | Performance drift at a new site | Different scanners, population or protocols (shown here: AP vs PA) | Silent loss of accuracy | Per-request monitoring log (latency, view, flags, max score). **Gap:** no live drift alerting yet (planned in project C) | 4 | 3 → 3 |
| H6 | Output used as a diagnosis | Misuse by staff or downstream systems | Wrong clinical decision | FHIR status `preliminary`; research disclaimer in every response and in the report conclusion | 5 | 2 → 2 |
| H7 | Patient data exposure | Identifiers in DICOM headers | Privacy breach | Only pixels and view are read; no identifiers are returned or logged (covered by a test) | 3 | 2 → 1 |

## 6. DTAC notes

| DTAC area | Where this prototype stands |
|---|---|
| Clinical safety (DCB0129/0160) | Hazard log above. Missing: a named Clinical Safety Officer, clinical safety case report and safety incident process |
| Data protection (UK GDPR) | No identifiers read or logged; no training on patient data. Missing: DPIA, data processing agreement, retention policy |
| Technical security | Containerised, dependency versions pinned, CI on every change. Missing: authentication, rate limiting, penetration test, Cyber Essentials |
| Interoperability | Accepts DICOM; returns FHIR R4 `DiagnosticReport` with contained `Observation`s. Missing: coded terminology (SNOMED CT, LOINC), DICOMweb or HL7 v2 integration |
| Usability and accessibility | Demo UI and OpenAPI docs. Missing: user research with radiologists, accessibility audit |

## 7. Monitoring plan

Each request logs latency, view, flagged findings and the maximum score, with no identifiers. In a pilot these would be reviewed weekly for:

- **Input drift:** the share of AP films and the score distribution compared with validation.
- **Flag rate per view:** a sudden rise suggests miscalibration at the new site.
- **Agreement with radiologist reports**, sampled, to re-estimate sensitivity and specificity.

A silent-mode pilot, in which the model runs without affecting care, and this monitoring are the next project ([plan](README.md#triage-service)).

## 8. Ethical considerations and caveats

- NIH labels are noisy, so reported performance is approximate.
- The 224-pixel input hides small findings, which is why nodules and masses are not offered.
- Subgroup fairness (sex, age) has not been measured yet. Until it has, the system should not be trusted equally across patient groups.

## References

- Wang et al., ChestX-ray8, CVPR 2017. NIH Clinical Center.
- Cohen et al., TorchXRayVision. https://github.com/mlmed/torchxrayvision
- Oakden-Rayner, Exploring large-scale public medical image datasets, 2019. https://arxiv.org/abs/1907.12720
- NHS England, Digital Technology Assessment Criteria (DTAC).
- NHS Digital, DCB0129 Clinical Risk Management: its Application in the Manufacture of Health IT Systems.

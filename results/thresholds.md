# Triage thresholds: single vs per-view

Thresholds tuned for 80% sensitivity on half of the patients (2115), evaluated on the other half (2115). sens = sensitivity, spec = specificity. A finding is not reported on a view where its tuning-half AUC is below 0.60.

| Finding       | View   |   Test positives | Single threshold: sens / spec   | Per-view threshold: sens / spec   | Own view threshold   | Reported on this view   |
|:--------------|:-------|-----------------:|:--------------------------------|:----------------------------------|:---------------------|:------------------------|
| Atelectasis   | AP     |              125 | 0.93 / 0.23                     | 0.81 / 0.39                       | yes                  | no (AUC < 0.60)         |
| Atelectasis   | PA     |              135 | 0.65 / 0.63                     | 0.79 / 0.48                       | yes                  | yes                     |
| Cardiomegaly  | AP     |               42 | 1.00 / 0.15                     | 0.86 / 0.32                       | yes                  | yes                     |
| Cardiomegaly  | PA     |               46 | 0.70 / 0.57                     | 0.85 / 0.50                       | yes                  | yes                     |
| Consolidation | AP     |               76 | 0.91 / 0.34                     | 0.83 / 0.45                       | yes                  | yes                     |
| Consolidation | PA     |               47 | 0.62 / 0.73                     | 0.85 / 0.60                       | yes                  | yes                     |
| Edema         | AP     |               57 | 0.93 / 0.27                     | 0.93 / 0.24                       | yes                  | no (AUC < 0.60)         |
| Edema         | PA     |                8 | 0.50 / 0.73                     | 0.50 / 0.73                       | no (fallback)        | yes                     |
| Effusion      | AP     |              156 | 0.96 / 0.30                     | 0.79 / 0.52                       | yes                  | yes                     |
| Effusion      | PA     |              159 | 0.72 / 0.71                     | 0.81 / 0.62                       | yes                  | yes                     |
| Pneumonia     | AP     |               23 | 0.91 / 0.23                     | 0.83 / 0.44                       | yes                  | yes                     |
| Pneumonia     | PA     |                9 | 0.56 / 0.46                     | 0.67 / 0.42                       | yes                  | no (AUC < 0.60)         |
| Pneumothorax  | AP     |               48 | 0.85 / 0.22                     | 0.85 / 0.21                       | yes                  | no (AUC < 0.60)         |
| Pneumothorax  | PA     |              101 | 0.84 / 0.49                     | 0.82 / 0.57                       | yes                  | yes                     |

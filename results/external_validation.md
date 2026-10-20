# External validation on NIH ChestX-ray14 sample

5606 images from 4230 patients. AUC with 95% CI from a patient-level bootstrap. Findings with fewer than 20 positives are not reported.

| Finding            |   Positives | CheXpert model (external)   | MIMIC model (external)   | NIH model (internal ref.)   |
|:-------------------|------------:|:----------------------------|:-------------------------|:----------------------------|
| Atelectasis        |         508 | 0.670 (0.649–0.692)         | 0.674 (0.650–0.696)      | 0.736 (0.716–0.756)         |
| Cardiomegaly       |         141 | 0.679 (0.633–0.721)         | 0.633 (0.588–0.677)      | 0.857 (0.825–0.885)         |
| Consolidation      |         226 | 0.757 (0.728–0.784)         | 0.688 (0.654–0.718)      | 0.777 (0.750–0.805)         |
| Edema              |         118 | 0.779 (0.742–0.813)         | 0.781 (0.745–0.812)      | 0.804 (0.768–0.835)         |
| Effusion           |         644 | 0.749 (0.729–0.766)         | 0.763 (0.742–0.781)      | 0.815 (0.798–0.832)         |
| Emphysema          |         127 | –                           | –                        | 0.651 (0.608–0.692)         |
| Fibrosis           |          84 | –                           | –                        | 0.729 (0.673–0.776)         |
| Infiltration       |         967 | –                           | –                        | 0.657 (0.637–0.677)         |
| Mass               |         284 | –                           | –                        | 0.775 (0.744–0.802)         |
| Nodule             |         313 | –                           | –                        | 0.656 (0.624–0.687)         |
| Pleural_Thickening |         176 | –                           | –                        | 0.722 (0.683–0.760)         |
| Pneumonia          |          62 | 0.647 (0.579–0.712)         | 0.595 (0.523–0.660)      | 0.723 (0.667–0.775)         |
| Pneumothorax       |         271 | 0.695 (0.657–0.731)         | 0.649 (0.617–0.683)      | 0.690 (0.658–0.724)         |

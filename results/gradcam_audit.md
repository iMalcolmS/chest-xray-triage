# Grad-CAM audit of the CheXpert model on NIH images

For each finding, the 30 most confident true positives and the 30 most confident false positives, one image per patient. "Heatmap inside chest" is the share of Grad-CAM weight inside the lungs, heart and mediastinum. "Lift" divides it by the chest's share of the image (1.0 = no more focused than a uniform heatmap). "Mostly outside chest" counts heatmaps with under half their weight inside.

| Finding      | Group          |   Images | Heatmap inside chest   | Chest share of image   |   Lift | Mostly outside chest   | AP (portable) views   |
|:-------------|:---------------|---------:|:-----------------------|:-----------------------|-------:|:-----------------------|:----------------------|
| Cardiomegaly | true positive  |       30 | 56.7%                  | 29.3%                  |   1.94 | 30%                    | 93%                   |
| Cardiomegaly | false positive |       30 | 52.2%                  | 27.6%                  |   1.89 | 40%                    | 97%                   |
| Effusion     | true positive  |       30 | 41.8%                  | 26.6%                  |   1.57 | 73%                    | 90%                   |
| Effusion     | false positive |       30 | 37.2%                  | 22.8%                  |   1.63 | 90%                    | 97%                   |
| Pneumothorax | true positive  |       30 | 47.8%                  | 34.3%                  |   1.4  | 57%                    | 17%                   |
| Pneumothorax | false positive |       30 | 48.8%                  | 34.6%                  |   1.41 | 63%                    | 17%                   |

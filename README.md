# DeepHis2Exp

Rethinking the generalization of deep learning models for predicting spatial gene expression profiles using histology images.

![Benchmarking Overview]('./Figures/Cover.png')

# Summary of Data Preprocessing Methods

| **Model** | **Tile Size (pixels)** | **Data Augmentation**                                        | **Gene Expression Pre-Processing** |
| --------- | ---------------------- | ------------------------------------------------------------ | ---------------------------------- |
| ST-Net    | 224 x 224              | Random Rotation & Flipping                                   | Log Transformation                 |
| HisToGene | 112 x 112              | Random Rotation & Flipping + ColorJitter                     | Normalization + Log Transformation |
| Hist2ST   | 112 x 112              | Random Rotation & Flipping + ColorJitter (Self-distillation strategy) | Normalization + Log Transformation |
| STimage   | 299 x 299              | Color Normalization (Vahadane) + Random one of flipping, cropping, noise addition, blurring, distortion, contrast adjustment, colour-shifting + Remove tiles with low tissue coverage (< 70%) | Log Transformation                 |
| DeepSpaCE | 224 x 224              | Remove tiles with high RGB values                            | SCTransform + MinMax Scaling       |
| BLEEP     | 224 x 224              | Random Rotation & Flip                                       | Normalization + Log Transformation |

# Summary of Image Encoders and Loss Functions from 6 Models

| **Model** | **Image Encoder**         | **Loss Function**                                            |
| --------- | ------------------------- | ------------------------------------------------------------ |
| ST-Net    | Densenet121 (Pre-trained) | Mean Squared Error                                           |
| HisToGene | MLP + Transformer         | Mean Squared Error                                           |
| Hist2ST   | CNN + Transformer + GCN   | Mean Squared Error (Regression) <br />Mean Squared Error (Self-distillation)<br />Negative log likelihood |
| STimage   | Resnet50 (Pre-trained)    | Negative log likelihood                                      |
| DeepSpaCE | VGG16 (Pre-trained)       | Mean Squared Error                                           |
| BLEEP     | Resnet50 (Pre-trained)    | Cross Entropy                                                |



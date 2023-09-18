# DeepHis2Exp

Generalization of deep learning models for predicting spatial gene expression profiles using histology images: A breast cancer case study.

![Benchmarking Overview](https://github.com/BiomedicalMachineLearning/DeepHis2Exp/blob/main/Figures/Cover.png)

# Table of content
The main scripts are contained in *src* folder.
* Assessment of model performance on Her2+ and Visium datasets.
* Assessment of model generalisation capability on out-of-domain dataset.
* Assessment of Hist2ST variants performance on Her2+ and Visium dataset.
* Quantify the effect of color transformation on Hist2ST model performance.
* Comparison of different data augmentaion methods from different models based on Hist2ST model.
* Comparison of different gene expression preprocessing methods based on Hist2ST model.

# Summary of Image Encoders and Loss Functions from 6 Models

| **Model** | **Image Encoder**         | **Loss Function**                                            |
| --------- | ------------------------- | ------------------------------------------------------------ |
| ST-Net    | Densenet121 (Pre-trained) | Mean Squared Error                                           |
| HisToGene | MLP + Transformer         | Mean Squared Error                                           |
| Hist2ST   | CNN + Transformer + GCN   | Mean Squared Error (Regression) <br />Mean Squared Error (Self-distillation)<br />Negative log likelihood |
| STimage   | Resnet50 (Pre-trained)    | Negative log likelihood                                      |
| DeepSpaCE | VGG16 (Pre-trained)       | Mean Squared Error                                           |
| BLEEP     | Resnet50 (Pre-trained)    | Cross Entropy                                                |

# Benchmarked papers

| No. | Algorithm | Year | Datasets | Compared |
| --- | --- | --- | --- | --- |
| 1 | [ST-Net](https://www.nature.com/articles/s41551-020-0578-x) [[Code]](https://github.com/bryanhe/ST-Net) | 2020 | BC0 |  |
| 2 | [HisToGene](https://www.biorxiv.org/content/10.1101/2021.11.28.470212v1.abstract) [[Code]](https://github.com/maxpmx/HisToGene) | 2021 | (1) BC1 (2) Human cutaneous squamous cell carcinoma 10x Visium data (GSE144240) | ST-Net |
| 3 | [Hist2ST](https://academic.oup.com/bib/article-abstract/23/5/bbac297/6645485) [[Code]](https://github.com/biomed-AI/Hist2ST) | 2022 | (1) BC1 (2) Human cutaneous squamous cell carcinoma 10x Visium data (GSE144240) | ST-Net, HisToGene |
| 4 | [DeepSpaCE](https://www.nature.com/articles/s41598-022-07685-4) [[Code]](https://github.com/tmonjo/DeepSpaCE/tree/main) | 2022 | Six human breast cancer tissue sections (DNA Data Bank of Japan: accession number JGAS000202 and JGAS000290) | ST-Net |
| 5 | [BLEEP](https://arxiv.org/pdf/2306.01859.pdf) [[Code]](https://github.com/bowang-lab/BLEEP/tree/main) | 2023 | Human liver tissue dataset ([Link](https://figshare.com/projects/Human_Liver_SC_vs_SN_paper/98981)) | ST-Net, HisToGene |
| 6 | [STimage](https://www.biorxiv.org/content/10.1101/2023.05.14.540710v1) [[Code]](https://github.com/BiomedicalMachineLearning/STimage) | 2023 | (1) BC1 (2) BC2 + BC3 | ST-Net, HisToGene, Hist2ST |



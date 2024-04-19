# DeepHis2Exp

Assessment of deep learning models for predicting spatial gene expression profiles using histology images
![Benchmarking Overview](https://github.com/BiomedicalMachineLearning/DeepHis2Exp/blob/main/Figures/Cover.png)

# Summary of Image Encoders and Loss Functions from 7 Models

| **Model** | **Image Encoder**         | **Loss Function**                                            |
| --------- | ------------------------- | ------------------------------------------------------------ |
| ST-Net    | Densenet121 (Pre-trained) | Mean Squared Error                                           |
| HisToGene | MLP + Transformer         | Mean Squared Error                                           |
| Hist2ST   | CNN + Transformer + GCN   | Mean Squared Error (Regression) <br />Mean Squared Error (Self-distillation)<br />Negative log likelihood |
| STimage   | Resnet50 (Pre-trained)    | Negative log likelihood                                      |
| DeepSpaCE | VGG16 (Pre-trained)       | Mean Squared Error                                           |
| BLEEP     | Resnet50 (Pre-trained)    | Cross Entropy                                                |
| DeepPT    | Resnet50 (Pre-trained)+AutoEncoder | Mean Squared Error (Regression) <br /> Mean Squared Error(Reconstruct loss)|


# DEEPHIS2EXP Tutorial

## Overview

DEEPHIS2EXP is a pipeline that can benchmarking existing deep learning models, which predicting gene expression from H&E images only.

We conduct comprehensive and systematic benchmarking on the points below.

* Benchmarking 7 deep learning models on 6 spatial transcriptomic datasets using top1000 highly variable genes and functional genes.

* Identification of predictable genes from the In-Domain training results and corresponding gene enrichment analysis.

* Downstream task analysis to evaluate the useness of predicted gene expression.
  

Note: Benchmarking model performance on In-Domain datasets are using the default parameters from the original papers. All benchmarking models applied `reinhard` color normalization to reduce the variation in the color space. For fairly comparison on unseen dataset and saving computational resources, we apply the early stopping for all models.


## Prerequisites

List any prerequisites or dependencies that users need to have installed or set up before using the pipeline.

The `DeepHis2Exp.yml` contains the required dependencies for the conda environment.
If you can not install via `yml` file, you can run the commands below. All models were implemented by `pytorch`, so the you need to install pytorch according to your device.

```
pip install scanpy
pip install squidpy
pip install torchstain
pip install lightning
pip install easydl
pip install einops
pip install geopandas
pip install splot
pip install scikit-image
pip install torch_geometric
pip install opencv-python
pip install scprep
pip install hdf5plugin
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```


## Getting Started

### Clone the Repository

```bash
git clone https://github.com/BiomedicalMachineLearning/DeepHis2Exp.git
cd DeepHis2Exp
```

### Benchmarking models on In-Domain (ID) datasets

To reproduce the benchmarking results for 7 models, you can run the command below.

```
python ./Implementation/Baseline.py \
        --dataset_name BC_visium \
        --model_name stimage \
        --gene_list func \
        --colornorm reinhard \
        --exp_norm lognorm \
        --seed 42 \
        --fold 0 \
```
Note: The trained model weights are saved in `./Model_Weights/{dataset_name}`. The predicted gene expressions are saved in `./Results/dataset_name/`, format is `*.h5ad`.  The filename should be followed the format `{model_name}_{dataset_name}_{colornorm}_{Slide_name}_{gene_list}.ckpt`. The results include three components, predicted gene expression (Anndata), ground truth gene expression (Anndata), spatial location matrix (numpy array). They were saved at `./Results/{dataset_name}`.

### Benchmarking model generalizability (OOD) datasets
```
python ./Implementation/Generalizability/OOD_Inference.py --fold 0 \
                                                        --colornorm reinhard \
                                                        --source_dataset BC_visium \
                                                        --target_dataset BC_Her2ST \
                                                        --HPC wiener \
```
Note: `source_dataset` represents which dataset the model is trained on, `target_dataset` is which dataset is used for evaluating model performance. This is inference only, without finetuning.

### Benchmarking customized graph construction methods

```
python ./Implementation/New_graphbuild/Graphconstruction.py  --fold 0 \
                                                                --SLG \
                                                                --PAG \
                                                                --HSG \
```

Note: The abalation experiment can be conducted by removing one of parameters(`SLG`, `PAG`, `HSG`). If `SLG` is True, the weighted graph is constructed by spatial location. If `PAG` is True, the weighted graph is constructed by pathology annotations. If `HSG` is True, the weighted graph is constructed by histological image similarity(distance=cosine similarity). If there are two graph methods presented, the weighted graph is the summation of two graphs.

### Benchmarking message passing methods based on DeepPT with pathology annotation graph

```
python ./Implementation/New_graphbuild/Messagepass.py   --PAG \
                                                        --gnn gin \
                                                        --fold 0 \
```

Note: The abalation experiment can be conducted by changing the `gnn` to `gin`(graph isomorphism network), `gat`(graph attention network), `gcn`(graph convolutional network). 


### Benchmarking color transformation

```
python ./Implementation/Preprocess.py --augment random_grayscale --fold 0 
```
Note: The `augment` can be changed to `blur`, `random_grayscale`.

### Benchmarking gene expression preprocessing

```
python ./Implementation/Preprocess.py --exp_norm minmax --fold
```
Note: `exp_norm` can be changed, the options are `raw`, `minmax`, `log1p`, `lognorm`, and `norm`.

### Downstream task performance (spatial region detection)
If you are interested in the clustering performance of predicted gene expression from seven models, refer to [notebook](Figures/Figure6.ipynb).

We evaluated the performance of predicted gene expression on the clustering task. We regard the pathology annotations as ground truth, using the `ARI` and `NMI` as the metrics to evaluate the usefulness of predicted gene expressions.

### Metrics computation
If you have generated the Anndata results and saved them at `./Results/{dataset_name}`, you can run the scripts below to compute the metrics and save them into a dataframe.
```
python ./Implementation/Metrics.py \
        --dataset_name BC_visium \
        --gene_list func \
        --fold \
```
The summary table will be saved at `./Results/Summary`. The fold is the No. of slide, you can use for loop to enumerate all results.

### Gene expression visualization
We can visualize the predicted gene expression on the tissue, refer to [notebook](./Figures/Figure5.ipynb).

### Pathway analysis
After we got the top predictable genes, we can have a quick look that the pathway enrichment analysis for the set of predictable genes, you can refer to [notebook](./Figures/FigureS3.ipynb).

### Customized train/valid/test dataset splitting strategy
You can use this pipeline to train and test on new visium data. You can find the example at [notebook](./Development/Blake_visium_train_test_regression.ipynb)


# Acknowledgments
The benchmarking work is based on the original paper and codes, part of them were reimplemented due to the availabity and reproducibility of the code.
| No. | Algorithm | Year | Datasets | Compared |
| --- | --- | --- | --- | --- |
| 1 | [ST-Net](https://www.nature.com/articles/s41551-020-0578-x) [[Code]](https://github.com/bryanhe/ST-Net) | 2020 | BC0 |  |
| 2 | [HisToGene](https://www.biorxiv.org/content/10.1101/2021.11.28.470212v1.abstract) [[Code]](https://github.com/maxpmx/HisToGene) | 2021 | (1) BC1 (2) Human cutaneous squamous cell carcinoma 10x Visium data (GSE144240) | ST-Net |
| 3 | [Hist2ST](https://academic.oup.com/bib/article-abstract/23/5/bbac297/6645485) [[Code]](https://github.com/biomed-AI/Hist2ST) | 2022 | (1) BC1 (2) Human cutaneous squamous cell carcinoma 10x Visium data (GSE144240) | ST-Net, HisToGene |
| 4 | [DeepSpaCE](https://www.nature.com/articles/s41598-022-07685-4) [[Code]](https://github.com/tmonjo/DeepSpaCE/tree/main) | 2022 | Six human breast cancer tissue sections (DNA Data Bank of Japan: accession number JGAS000202 and JGAS000290) | ST-Net |
| 5 | [BLEEP](https://arxiv.org/pdf/2306.01859.pdf) [[Code]](https://github.com/bowang-lab/BLEEP/tree/main) | 2023 | Human liver tissue dataset ([Link](https://figshare.com/projects/Human_Liver_SC_vs_SN_paper/98981)) | ST-Net, HisToGene |
| 6 | [STimage](https://www.biorxiv.org/content/10.1101/2023.05.14.540710v1) [[Code]](https://github.com/BiomedicalMachineLearning/STimage) | 2023 | (1) BC1 (2) BC2 + BC3 | ST-Net, HisToGene, Hist2ST |
| 7 | [DeepPT](https://www.biorxiv.org/content/10.1101/2022.06.07.495219v3.full) [[Code]](https://github.com/PangeaResearch/enlight-deeppt-data.) | 2023 | None | None |


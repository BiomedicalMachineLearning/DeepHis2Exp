import sys
import shutil
import os
import gc
import cv2
import tqdm
import torch
import random
import glob
import pickle
import geopandas as gpd
import concurrent.futures
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Add relative path
sys.path.append("../../")

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from Models.Hist2ST.hist2st import *
from Models.STnet.stnet import *
from Models.HisToGene.histogene import *
from Models.DeepPT.deeppt import *
from Models.STimage.stimage import *
from Models.BLEEP.bleep import *
from Models.DeepSpaCE.deepspace import *
from Dataloader.Dataset_wiener import *

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse

import warnings
warnings.filterwarnings("ignore")

def inference():
    print("Start inference!")
    print("Hyperparameters are as follows:")
    print("Random seed:", seed)
    print("Model_name:", model_name)
    print("Color normalization method:", colornorm)
    print("gene_list:", gene_list)
    print("exp_norm:", exp_norm)
    print("source_dataset:", source_dataset)
    print("target_dataset:", target_dataset)

    # Modify the loading part to handle CPU-only machines
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

    # Define model 
    if model_name == "hist2st":
        tag = '5-7-2-8-4-16-32'
        kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))
        model = Hist2ST(
            depth1=depth1, depth2=depth2,
            n_genes=len(ID_target_gene_list), learning_rate=1e-5, label=None, 
            kernel_size=kernel, patch_size=patch, n_pos=256,
            heads=heads, channel=channel, dropout=0.2,
            zinb=0.25, nb=False,
            bake=5, lamb=0.5,)
    elif model_name == "stnet":
        model = STModel(n_genes=len(ID_target_gene_list), hidden_dim=1024, learning_rate=1e-6)
    elif model_name == "histogene":
        model = HisToGene(patch_size=224, n_layers=4, n_genes=len(ID_target_gene_list), dim=1024, 
                        learning_rate=1e-5, dropout=0.1, n_pos=256)
    elif model_name == "deeppt":
        model = DeepPT(n_genes=len(ID_target_gene_list), hidden_dim=512, learning_rate=1e-4,)
    elif model_name == "stimage":
        model = STimage(n_genes=len(ID_target_gene_list), ft=False, learning_rate=1e-5)
    elif model_name == "bleep":
        model = BLEEP(n_genes=len(ID_target_gene_list)) 
    elif model_name == "deepspace":
        model = DeepSpaCE(n_genes=len(ID_target_gene_list)) 

    # Find the common genes between the source and target gene list
    common_genes = list(set(ID_target_gene_list).intersection(set(OOD_target_gene_list)))

    # Ensemble the results 
    preds = []
    trainer = pl.Trainer(accelerator='auto', max_epochs=100, logger=False)
    for i in range(source_datasize): # the datasize of source dataset
        # Load pretrained model weights
        weights = glob.glob(f"{model_weight_path}/{source_dataset}/{model_name}_{source_dataset}_reinhard_*_func.ckpt")
        model.load_state_dict(torch.load(weights[i], map_location=device))
        if model_name == "bleep":
            # Load the pretrained dataset database
            if source_dataset == "Skin_Melanoma":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    full_train_dataset = Skin_Melanoma(train=True,fold=i,r=112, neighs=8, gene_list=ID_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=100)
                else:
                    full_train_dataset = Skin_Melanoma(train=True,fold=i,r=112, neighs=8, gene_list=ID_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15)

            elif source_dataset == "BC_visium":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    full_train_dataset = BC_visium(train=True,fold=i,r=112, neighs=8, 
                                                exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, 
                                                gene_list=ID_target_gene_list)
                else:
                    full_train_dataset = BC_visium(train=True,fold=i, r=112, neighs=8, color_norm=colornorm, 
                                                exp_norm=exp_norm, num_subsets=15, gene_list=ID_target_gene_list)
                    
            elif source_dataset == "Skin_cSCC":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    full_train_dataset = CSCC(train=True,fold=i,r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, gene_list=ID_target_gene_list)
                else:
                    full_train_dataset = CSCC(train=True,fold=i,r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, num_subsets=5, gene_list=ID_target_gene_list)

            elif source_dataset == "BC_Her2ST":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    full_train_dataset = Her2st(train=True,fold=i, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm=exp_norm, gene_list=ID_target_gene_list)
                else:
                    full_train_dataset = Her2st(train=True,fold=i, num_subsets=3, r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, gene_list=ID_target_gene_list)
                    
            elif source_dataset == "Liver_visium":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    full_train_dataset = Liver_visium(train=True,fold=i,r=112, neighs=8, gene_list=ID_target_gene_list, color_norm=colornorm, exp_norm="log1p", num_subsets=100)
                else:
                    full_train_dataset = Liver_visium(train=True,fold=i,r=112, neighs=8, gene_list=ID_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15)

            elif source_dataset == "Kidney_visium":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    full_train_dataset = Kidney_visium(train=True,fold=i,r=112, neighs=8, gene_list=ID_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=100)
                else:
                    full_train_dataset = Kidney_visium(train=True,fold=i,r=112, neighs=8, gene_list=ID_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15)
            tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
        for j in range(target_datasize):
            # Load the targeted test dataset
            if target_dataset == "Skin_Melanoma":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    test_dataset = Skin_Melanoma(train=False,fold=j, r=112, neighs=8, gene_list=OOD_target_gene_list, exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, shuffle=False)
                else:
                    test_dataset = Skin_Melanoma(train=False,fold=j, r=112, neighs=8, gene_list=OOD_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

            elif target_dataset == "BC_visium":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    test_dataset = BC_visium(train=False,fold=j,r=112, neighs=8, 
                                            exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, 
                                            gene_list=OOD_target_gene_list, shuffle=False)
                else:
                    test_dataset = BC_visium(train=False,fold=j,r=112, neighs=8, color_norm=colornorm, 
                                            exp_norm=exp_norm, num_subsets=15, gene_list=OOD_target_gene_list, shuffle=False)
                    
            elif target_dataset == "Skin_cSCC":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    test_dataset = CSCC(train=False,fold=j,r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, gene_list=OOD_target_gene_list, shuffle=False)
                else:
                    test_dataset = CSCC(train=False,fold=j,r=112, neighs=8, num_subsets=5, color_norm=colornorm, exp_norm=exp_norm, gene_list=OOD_target_gene_list, shuffle=False)

            elif target_dataset == "BC_Her2ST":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    test_dataset = Her2st(train=False,fold=j, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm=exp_norm, gene_list=OOD_target_gene_list, shuffle=False)
                else:
                    test_dataset = Her2st(train=False,fold=j,r=112, num_subsets=3, neighs=8, color_norm=colornorm, exp_norm=exp_norm, gene_list=OOD_target_gene_list, shuffle=False)
                    
            elif target_dataset == "Liver_visium":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    test_dataset = Liver_visium(train=False,fold=j, r=112, neighs=8, gene_list=OOD_target_gene_list, exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, shuffle=False)
                else:
                    test_dataset = Liver_visium(train=False,fold=j, r=112, neighs=8, gene_list=OOD_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

            elif target_dataset == "Kidney_visium":
                if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
                    test_dataset = Kidney_visium(train=False,fold=j, r=112, neighs=8, gene_list=OOD_target_gene_list, exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, shuffle=False)
                else:
                    test_dataset = Kidney_visium(train=False,fold=j, r=112, neighs=8, gene_list=OOD_target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

            te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            gc.collect()

            # Load the name of test sample 
            slide = list(test_dataset.names)[0].split("-")[0]
            
            # Open a folder if it does not exist. 
            if not os.path.isdir(f"{res_path}/Generalizability"):
                os.mkdir(f"{res_path}/Generalizability")
            if not os.path.isdir(f"{res_path}/Generalizability/{source_dataset}_{target_dataset}"):
                os.mkdir(f"{res_path}/Generalizability/{source_dataset}_{target_dataset}")
            # Save spatial location to numpy array
            spatial_loc = np.concatenate([test_dataset.meta_dict[key].obsm["spatial"] for key in list(test_dataset.meta_dict.keys())])
            np.save(f'{res_path}/Generalizability/{source_dataset}_{target_dataset}/spatial_loc_{model_name}_{source_dataset}_{target_dataset}_{test_dataset.te_names}_{gene_list}.npy', spatial_loc)
            gc.collect()

            if model_name == "bleep":
                # BLEEP Inference
                gt_exp, pred = bleep_inference(model, trainer, tr_loader, te_loader, method="average")
                preds.append(pred)
                gc.collect()
            else:
                # Inference
                out = trainer.predict(model, te_loader)
                pred = np.concatenate([out[i][0] for i in range(len(out))])
                gt_exp = np.concatenate([out[i][1] for i in range(len(out))])
            pred = ad.AnnData(pred)
            pred.var_names = ID_target_gene_list
            pred = pred[:, common_genes]
            pred.obsm["spatial"] = spatial_loc
            pred.write(f"{res_path}/Generalizability/{source_dataset}_{target_dataset}/pred_{model_name}_{source_dataset}_{target_dataset}_{slide}_{i}.h5ad")
        
            gt = sc.concat([test_dataset.meta_dict[sub_slide] for sub_slide in list(test_dataset.meta_dict.keys())])[:, OOD_target_gene_list].copy()
            gt.X = gt_exp
            gt = gt[:,common_genes]
            gc.collect()
            gt.write(f"{res_path}/Generalizability/{source_dataset}_{target_dataset}/gt_{model_name}_{source_dataset}_{target_dataset}_{slide}.h5ad")
            print(f"finish inference of {model_name}_{j}!")
            del test_dataset
        del full_train_dataset
    print("finish inference!")

"""
Start inference!
"""
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='a parameter to loop model names.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="reinhard", help='Color normalization methods.')
parser.add_argument('--source_dataset', type=str, default="Skin_Melanoma", help='The model weights from which dataset.')
parser.add_argument('--target_dataset', type=str, default="Skin_cSCC", help='The inferenced datasets.')
parser.add_argument('--gene_list', type=str, default="func", help='Gene list choice.')
parser.add_argument('--HPC', type=str, default="wiener", help='HPC cluster')
args = parser.parse_args()

fold = args.fold
seed = args.seed
colornorm = args.colornorm
source_dataset = args.source_dataset # "Skin_Melanoma", "BC_visium", "BC_Her2ST", "Liver_visium", "Kidney_visium", "Skin_cSCC"
target_dataset = args.target_dataset # "Skin_Melanoma", "BC_visium", "BC_Her2ST", "Liver_visium", "Kidney_visium", "Skin_cSCC"
gene_list = args.gene_list # "func", "hvg"
hpc = args.HPC

if hpc == "wiener":
    abs_path = "/afm03/Q2/Q2051" # The code should be located at here
    model_weight_path = "/afm03/Q2/Q2051/DeepHis2Exp/Model_Weights" # The model weights should be saved at here
    res_path = "/afm03/Q2/Q2051/DeepHis2Exp/Results" # The results should be saved at here
    data_path = "/afm03/Q2/Q2051/DeepHis2Exp/Dataset" # The dataset should be loaded at here
elif hpc == "vmgpu":
    abs_path = "/afm01/UQ/Q2051/DeepHis2Exp/Implementation"
    model_weight_path = "/afm01/UQ/Q2051/DeepHis2Exp/Model_Weights"
    res_path = "/afm01/UQ/Q2051/DeepHis2Exp/Results"
    data_path = "/afm01/UQ/Q2051/DeepHis2Exp/Dataset"
elif hpc == "bunya":
    abs_path = "/QRISdata/Q2051/DeepHis2Exp/Implementation"
    model_weight_path = "/QRISdata/Q2051/DeepHis2Exp/Model_Weights"
    res_path = "/QRISdata/Q2051/DeepHis2Exp/Results"
    data_path = "/QRISdata/Q2051/DeepHis2Exp/Dataset"
    
model_name = ["stnet", "deeppt", "stimage", "deepspace", "hist2st", "histogene", "bleep"][fold] # "stnet", "deeppt", "stimage", "bleep", "deepspace", "hist2st", "histogene"
model_exp_norm = {
                    "stnet": "log1p",
                    "deeppt": "log1p",
                    "stimage": "log1p",
                    "bleep": "log1p",
                    "deepspace": "log1p",
                    "hist2st": "lognorm",
                    "histogene": "lognorm",}

exp_norm = model_exp_norm.get(model_name, "lognorm")

dataset_sizes = {
    "Skin_Melanoma": 5,
    "BC_visium": 9,
    "BC_Her2ST": 36,
    "Liver_visium": 4,
    "Kidney_visium": 6,
    "Skin_cSCC": 12
}
source_datasize = dataset_sizes.get(source_dataset, None)
target_datasize = dataset_sizes.get(target_dataset, None)

# For reproducing the results
seed_everything(seed)

# Functional genes for visium dataset
ID_target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_func_{source_dataset}.npy', allow_pickle=True))
OOD_target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_func_{target_dataset}.npy', allow_pickle=True))

# Run the inference codes
inference()
import sys
import shutil
import os
import gc
import cv2
import tqdm
import torch
import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import anndata as ad
import argparse

# Add relative path
sys.path.append("../")

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from Models.Hist2ST.hist2st import *
from Models.STnet.stnet import *
from Models.HisToGene.histogene import *
from Models.DeepPT.deeppt import *
from Models.STimage.stimage import *
from Models.BLEEP.bleep import *
from Models.DeepSpaCE.deepspace import *
# from Dataloader.Dataset import *
from Dataloader.Dataset_wiener import *


# parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="reinhard", help='Color normalization methods.')
parser.add_argument('--dataset_name', type=str, default="SCC_Chenhao", help='Dataset choice.')
parser.add_argument('--gene_list', type=str, default="func", help='Gene list choice.')
parser.add_argument('--hpc', type=str, default="wiener", help='The usage of HPC ')
args = parser.parse_args()

seed = args.seed
colornorm = args.colornorm
dataset_name = args.dataset_name
gene_list = args.gene_list
exp_norm = args.exp_norm

hpc = "wiener"
if hpc == "wiener":
    abs_path = "/afm03/Q2/Q2051/DeepHis2Exp/Models/Benchmarking_main"
    model_weight_path = "/afm03/Q2/Q2051/DeepHis2Exp/Model_Weights"
    res_path = "/afm03/Q2/Q2051/DeepHis2Exp/Results"
    data_path = "/afm03/Q2/Q2051/DeepHis2Exp/Dataset"
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

# seed = 42
# colornorm = "reinhard"    # "reinhard", "raw"
# dataset_name = "Skin_Melanoma"   # "BC_visium", "BC_Her2ST", "Skin_Melanoma", "Skin_cSCC", "Liver_visium", "Kidney_visium"
# model_name = "deeppt"      # "hist2st", "histogene", "stnet", "deeppt", "stimage", "bleep", "deepspace"
# gene_list = "func"    # "hvg", "func"
# exp_norm = "log1p"
gnn = False
PAG = False
GAG = False
HSG = False

# The size of dataset
if dataset_name == "BC_Her2ST":
    datasize = 36
elif dataset_name == "BC_visium":
    datasize = 9
elif dataset_name == "Skin_cSCC":
    datasize = 12
elif dataset_name == "Skin_Melanoma":
    datasize = 5
elif dataset_name == "Liver_visium":
    datasize = 4
elif dataset_name == "Kidney_visium":
    datasize = 6
    
for model_name in ["stnet", "deeppt", "stimage", "histogene", "hist2st", "deepspace", "bleep"]:
    for fold in range(datasize): # enumerate all samples in the dataset
        print("Start predicting!")
        print("Hyperparameters are as follows:")
        print("Fold:", fold)
        print("Color normalization method:", colornorm)
        print("Dataset_name:", dataset_name)
        print("Model_name:", model_name)
        print("gene_list:", gene_list)
        print("exp_norm:", exp_norm)
        print("cluster:", hpc)

        # For reproducing the results
        seed_everything(seed)

        tag = '5-7-2-8-4-16-32'
        kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))

        # Load train and test dataset and wrap dataloader
        # Train dataset is sampled WSI dataset, test dataset is WSI dataset

        # Functional genes for visium dataset
        target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_{gene_list}_{dataset_name}.npy', allow_pickle=True))

        # Load sample names
        if dataset_name == "Skin_Melanoma":
            # For SCC_Chenhao dataset
            # data_path = "/afm03/Q2/Q2051/DeepHis2Exp/Dataset/Skin_Melanoma/"
            name_slides = ["Visium29_B1", "Visium29_C1", "Visium37_D1", "Visium38_B1", "Visium38_D1"]
            if model_name in ["stnet", "deeppt", "stimage", "deepspace"]:
                full_train_dataset, test_dataset = SCC_Chenhao_Visium(test_sample = name_slides[fold], sizes = 2700, 
                n_subsets = 50, exp_norm="log1p", n_neighbors=8, gene_list=target_gene_list, GAG=GAG, PAG=PAG, 
                                                                            color_norm=colornorm, model_name=model_name)
            else:
                full_train_dataset, test_dataset = SCC_Chenhao_Visium(test_sample = name_slides[fold], sizes = 2700, 
                n_subsets = 10, exp_norm="lognorm", n_neighbors=8, gene_list=target_gene_list, GAG=GAG, PAG=PAG, 
                                                                            color_norm=colornorm, model_name=model_name)

        elif dataset_name == "BC_visium":
            if model_name in ["stnet", "deeppt", "stimage", "deepspace"]:
                full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=8, 
                                               exp_norm="log1p", color_norm=colornorm, num_subsets=100, 
                                               gene_list=target_gene_list)
                test_dataset = BC_visium(train=False,fold=fold,r=112, neighs=8, 
                                         exp_norm="log1p", color_norm=colornorm, num_subsets=100, 
                                         gene_list=target_gene_list)
            else:
                full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, 
                                               exp_norm="lognorm", num_subsets=15, gene_list=target_gene_list)
                test_dataset = BC_visium(train=False,fold=fold,r=112, neighs=8, color_norm=colornorm, 
                                         exp_norm="lognorm", num_subsets=15, gene_list=target_gene_list)

        elif dataset_name == "Skin_cSCC":
            if model_name in ["stnet", "deeppt", "stimage", "deepspace"]:
                # when num_subsets=12, there is cuda memory issue. So modified to 15.
                full_train_dataset = CSCC(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm="log1p", num_subsets=15, gene_list=target_gene_list)
                test_dataset = CSCC(train=False,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm="log1p", num_subsets=15, gene_list=target_gene_list)
            else:
                # when num_subsets=3, there is cuda memory issue. So modified to 5.
                full_train_dataset = CSCC(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm="lognorm", num_subsets=5, gene_list=target_gene_list)
                test_dataset = CSCC(train=False,fold=fold,r=112, neighs=8, num_subsets=5, color_norm=colornorm, exp_norm="lognorm", gene_list=target_gene_list)

        elif dataset_name == "BC_Her2ST":
            if model_name in ["stnet", "deeppt", "stimage", "deepspace"]:
                full_train_dataset = Her2st(train=True,fold=fold, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm="log1p", gene_list=target_gene_list)
                test_dataset = Her2st(train=False,fold=fold, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm="log1p", gene_list=target_gene_list)
            else:
                full_train_dataset = Her2st(train=True,fold=fold, num_subsets=3, r=112, neighs=8, color_norm=colornorm, exp_norm="lognorm", gene_list=target_gene_list)
                test_dataset = Her2st(train=False,fold=fold,r=112, num_subsets=3, neighs=8, color_norm=colornorm, exp_norm="lognorm", gene_list=target_gene_list)

        elif dataset_name == "Liver_visium":
            if model_name in ["stnet", "deeppt", "stimage", "deepspace"]:
                full_train_dataset = Liver_visium(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="log1p", num_subsets=100)
                test_dataset = Liver_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, exp_norm="log1p", color_norm=colornorm, num_subsets=100)
            else:
                full_train_dataset = Liver_visium(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="lognorm", num_subsets=15)
                test_dataset = Liver_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="lognorm", num_subsets=15)

        elif dataset_name == "Kidney_visium":
            if model_name in ["stnet", "deeppt", "stimage", "deepspace"]:
                full_train_dataset = Kidney_visium(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="log1p", num_subsets=100)
                test_dataset = Kidney_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, exp_norm="log1p", color_norm=colornorm, num_subsets=100)
            else:
                full_train_dataset = Kidney_visium(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="lognorm", num_subsets=15)
                test_dataset = Kidney_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="lognorm", num_subsets=15)

        gc.collect()
        # For real dataset only
        te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Define model and train
        if model_name == "hist2st":
            model = Hist2ST(
                depth1=depth1, depth2=depth2,
                n_genes=len(test_dataset.gene_list), learning_rate=1e-5, label=None, 
                kernel_size=kernel, patch_size=patch, n_pos=256,
                heads=heads, channel=channel, dropout=0.2,
                zinb=0.25, nb=False,
                bake=5, lamb=0.5,)
        elif model_name == "stnet":
            model = STModel(n_genes=len(test_dataset.gene_list), hidden_dim=1024, learning_rate=1e-6)
        elif model_name == "histogene":
            model = HisToGene(patch_size=224, n_layers=4, n_genes=len(test_dataset.gene_list), dim=1024, 
                              learning_rate=1e-5, dropout=0.1, n_pos=256)
        elif model_name == "deeppt":
            model = DeepPT(n_genes=len(test_dataset.gene_list), hidden_dim=512, learning_rate=1e-4,)
        elif model_name == "stimage":
            model = STimage(n_genes=len(test_dataset.gene_list), ft=False, learning_rate=1e-5)
        elif model_name == "bleep":
            model = BLEEP(n_genes=len(test_dataset.gene_list)) 
        elif model_name == "deepspace":
            model = DeepSpaCE(n_genes=len(test_dataset.gene_list)) 

        # Empty cache of GPU
        torch.cuda.empty_cache()

        # Define trainer for each model
        if model_name == "hist2st":
            trainer = pl.Trainer(accelerator='auto', logger=True)
        elif model_name == "stnet":
            trainer = pl.Trainer(accelerator='auto', logger=True)
        elif model_name == "histogene":
            trainer = pl.Trainer(accelerator='auto', logger=True)
        elif model_name == "deeppt":
            trainer = pl.Trainer(accelerator='auto', logger=True)
        elif model_name == "stimage":
            trainer = pl.Trainer(accelerator='auto', logger=True)
        elif model_name == "bleep":
            trainer = pl.Trainer(accelerator='auto', logger=True)   
        elif model_name == "deepspace":
            trainer = pl.Trainer(accelerator='auto', logger=True)   

        # Loading the best model
        best_model_path = f"{model_weight_path}/{dataset_name}/{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.ckpt"
        model.load_state_dict(torch.load(best_model_path))

        # Inference
        if model_name == "bleep":
            if dataset_name == "BC_visium":
                full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=8, 
                                               exp_norm="lognorm", color_norm=colornorm, num_subsets=100, 
                                               gene_list=target_gene_list)
            elif dataset_name == "Skin_cSCC":
                # when num_subsets=12, there is cuda memory issue. So modified to 15.
                full_train_dataset = CSCC(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm="lognorm", num_subsets=15, gene_list=target_gene_list)

            elif dataset_name == "BC_Her2ST":
                full_train_dataset = Her2st(train=True,fold=fold, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm="lognorm", gene_list=target_gene_list)

            elif dataset_name == "Liver_visium":
                full_train_dataset = Liver_visium(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="lognorm", num_subsets=100)

            elif dataset_name == "Kidney_visium":
                full_train_dataset = Kidney_visium(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm="lognorm", num_subsets=100)
            tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=False)
            gc.collect()
            gt, pred = bleep_inference(model, trainer, tr_loader, te_loader, method="average")
            gt, pred = ad.AnnData(gt), ad.AnnData(pred)
        else:
            gc.collect()
            out = trainer.predict(model, te_loader)
            pred = ad.AnnData(np.concatenate([out[i][0] for i in range(len(out))]))
            gt = sc.concat([test_dataset.adata_dict[sub_slide] for sub_slide in list(test_dataset.adata_dict.keys())])[:,target_gene_list]

        # Add the gene list to AnnData    
        pred.var_names = test_dataset.gene_list

        # Save AnnData to H5AD file
        if not os.path.isdir(f"{res_path}/{dataset_name}/"):
            os.mkdir(f"{res_path}/{dataset_name}/")
        pred.write(f"{res_path}/{dataset_name}/pred_{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.h5ad")
        gt.write(f"{res_path}/{dataset_name}/gt_{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.h5ad")

        # Save spatial location to numpy array
        spatial_loc = np.concatenate([test_dataset.adata_dict[key].obsm["spatial"] for key in list(test_dataset.adata_dict.keys())])
        np.save(f'{res_path}/{dataset_name}/spatial_loc_{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.npy', spatial_loc)
        gc.collect()

        # Remove the log file after all models are trained
        # shutil.rmtree(f'{abs_path}/lightning_logs')
        del model, test_dataset
        print("Finish training!")


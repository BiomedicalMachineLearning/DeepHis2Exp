import sys
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
sys.path.append("../../")

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from Models.DeepPT.DeepPT_GNN import *
from Dataloader.Dataset_wiener import *


parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="raw", help='Color normalization methods.')
parser.add_argument('--dataset_name', type=str, default="BC_visium", help='Dataset choice.')
parser.add_argument('--model_name', type=str, default="DeepPT_GNN", help='Model choice.')
parser.add_argument('--gene_list', type=str, default="func", help='Gene list choice.')
parser.add_argument('--hpc', type=str, default="wiener", help='Clusters choice')
parser.add_argument('--exp_norm', type=str, default="lognorm", help='Expression preprocessing')
parser.add_argument('--PAG', default=False, action="store_true",)
parser.add_argument('--SLG', default=False, action="store_true")
parser.add_argument('--HSG', default=False, action="store_true")
args = parser.parse_args()

fold = args.fold
seed = args.seed
colornorm = args.colornorm
dataset_name = args.dataset_name
model_name = args.model_name
gene_list = args.gene_list
exp_norm = args.exp_norm
hpc = args.hpc
PAG = args.PAG
SLG = args.SLG
HSG = args.HSG

"""
Hyperparameters settings
"""
# hpc = "wiener"
# fold = 1
# seed = 42
# dataset_name = "BC_visium"
# colornorm = "raw"    # "reinhard", "raw"
# model_name = "DeepPT_GNN"  
# gene_list = "func"
# exp_norm = "lognorm"
# PAG = True
# SLG = False
# HSG = False

print("Start working!")
print("Hyperparameters are as follows:")
print("Fold:", fold)
print("Color normalization method:", colornorm)
print("Dataset_name:", dataset_name)
print("Model_name:", model_name)
print("gene_list:", gene_list)
print("exp_norm:", exp_norm)
print("cluster:", hpc)
print("HSG:", HSG)
print("SLG:", SLG)
print("PAG:", PAG)

if hpc == "wiener":
    abs_path = "/afm03/Q2/Q2051/DeepHis2Exp/Models/Benchmarking_main"
    model_weight_path = "/afm03/Q2/Q2051/DeepHis2Exp/Model_Weights"
   #  model_weight_path = "/scratch/imb/uqyjia11/Yuanhao/DeepHis2Exp/Model_Weights"
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

# For reproducing the results
seed_everything(seed)

# Load train and test dataset and wrap dataloader

# Functional genes for visium dataset
target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_{gene_list}_{dataset_name}.npy', allow_pickle=True))

# Load sample names
full_train_dataset = WeightedGraph_Anndata(fold=fold, gene_list=target_gene_list, num_subsets=50,
                    train=True, r=112, exp_norm='lognorm', SLG=SLG, HSG=HSG, PAG=PAG,
                    neighs=8, color_norm=colornorm, target=target, distance_mode="distance",)
tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
gc.collect()
test_dataset = WeightedGraph_Anndata(fold=fold, gene_list=target_gene_list, num_subsets=50,
                    train=False, r=112, exp_norm='lognorm', SLG=SLG, HSG=HSG, PAG=PAG,
                    neighs=8, color_norm=colornorm, target=target, distance_mode="distance",)
te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
gc.collect()


# Define model and train
model = CNN_AE_GNN(n_genes=len(target_gene_list), hidden_dim=512, learning_rate=1e-4)
    
# Empty cache of GPU
torch.cuda.empty_cache()

# Create folder to save model weights
if not os.path.isdir(f"{model_weight_path}/New_GraphBuild/"):
    os.mkdir(f"{model_weight_path}/New_GraphBuild/")
early_stop = pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=10)
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, dirpath=f"{model_weight_path}", 
                                                   filename=f"{model_name}_SLG_{str(SLG)}PAG{str(PAG)}_HSG{str(HSG)}_{test_dataset.te_names}_{gene_list}", 
                                                   monitor="train_loss", mode="min")

trainer = pl.Trainer(accelerator='auto', 
                    callbacks=[early_stop, checkpoint_callback], 
                    max_epochs=300, logger=True)

# Start training and save best model
trainer.fit(model, tr_loader)

# debug
# trainer.fit(model, te_loader)

print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
print(checkpoint_callback.best_model_score) # and prints it score
best_model = model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
torch.save(torch.load(checkpoint_callback.best_model_path)["state_dict"], f"{model_weight_path}/New_GraphBuild/{model_name}_SLG_{str(SLG)}PAG{str(PAG)}_HSG{str(HSG)}_{test_dataset.te_names}_{gene_list}.ckpt")
os.remove(checkpoint_callback.best_model_path)


# Inference

gc.collect()
target_gene_list = test_dataset.gene_set
out = trainer.predict(model, te_loader)
pred = ad.AnnData(np.concatenate([out[i][0] for i in range(len(out))]))
gt_exp = np.concatenate([out[i][1] for i in range(len(out))])
gt = sc.concat([test_dataset.meta_dict[sub_slide] for sub_slide in list(test_dataset.meta_dict.keys())])[:,target_gene_list]
gt.X = gt_exp
        
# Add the gene list to AnnData    
pred.var_names = target_gene_list
gt.var_names = target_gene_list

# Save AnnData to H5AD file
if not os.path.isdir(f"{res_path}/New_GraphBuild/"):
    os.mkdir(f"{res_path}/New_GraphBuild/")
pred.write(f"{res_path}/New_GraphBuild/pred_{model_name}_SLG_{str(SLG)}PAG{str(PAG)}_HSG{str(HSG)}_{test_dataset.te_names}_{gene_list}.h5ad")
gt.write(f"{res_path}/New_GraphBuild/gt_{model_name}_SLG_{str(SLG)}PAG{str(PAG)}_HSG{str(HSG)}_{test_dataset.te_names}_{gene_list}.h5ad")
gc.collect()

# Save spatial location to numpy array
spatial_loc = np.concatenate([test_dataset.meta_dict[key].obsm["spatial"] for key in list(test_dataset.meta_dict.keys())])
np.save(f'{res_path}/New_GraphBuild/spatial_loc_{model_name}_SLG_{str(SLG)}PAG{str(PAG)}_HSG{str(HSG)}_{test_dataset.te_names}_{gene_list}.npy', spatial_loc)
gc.collect()

print("Finish training!")
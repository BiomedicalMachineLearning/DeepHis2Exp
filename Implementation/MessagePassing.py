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
sys.path.append("../")
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from Models.DeepPT.CNN_GAE import *
from Dataloader.Dataset_wiener import *
# from Dataloader.Dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="reinhard", help='Color normalization methods.')
parser.add_argument('--dataset_name', type=str, default="weighted_graph", help='Dataset choice.')
parser.add_argument('--model_name', type=str, default="CNN_GAE", help='Model choice.')
parser.add_argument('--gene_list', type=str, default="func", help='Gene list choice.')
parser.add_argument('--gnn', type=str, default="GCN", help='Gene list choice.')
parser.add_argument('--GAG', action='store_true', default=False, help='Gene expression similarity graph.')
parser.add_argument('--PAG', action='store_true', default=False, help='Pathology annotation graph.')
parser.add_argument('--weighted_graph', action='store_true', default=False, help='weighted graph.')
parser.add_argument('--hpc', type=str, default="wiener", help='Clusters choice')
args = parser.parse_args()

fold = args.fold
seed = args.seed
colornorm = args.colornorm
dataset_name = args.dataset_name
model_name = args.model_name
gene_list = args.gene_list
weighted_graph = args.weighted_graph
gnn = args.gnn
pag = args.PAG
gag = args.GAG
hpc = args.hpc

exp_norm = "log1p"
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

print("Start training!")
print("Hyperparameters are as follows:")
print("Fold:", fold)
print("Color normalization method:", colornorm)
print("Dataset_name:", dataset_name)
print("Model_name:", model_name)
print("gene_list:", gene_list)
print("exp_norm:", exp_norm)
print("gnn:", gnn)
print("PAG:", pag)
print("GAG:", gag)
print("cluster:", hpc)


# For reproducing the results
seed_everything(seed)

# Load train and test dataset and wrap dataloader
target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_{gene_list}_BC_visium.npy', allow_pickle=True))

full_train_dataset = GAE_Anndata(
    fold=fold, gene_list=target_gene_list, num_subsets=60,
    train=True, r=112, exp_norm=exp_norm, GAG=gag, PAG=pag,
    neighs=6, color_norm=colornorm, target=target, weighted_graph=weighted_graph )
test_dataset = GAE_Anndata(
    fold=fold, gene_list=target_gene_list, num_subsets=60,
    train=False, r=112, exp_norm=exp_norm, GAG=False, PAG=False,
    neighs=6, color_norm=colornorm, target=target, weighted_graph=weighted_graph, )

# For real dataset only
tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
gc.collect()

# Define model and train
model = CNNGAE(gnn=gnn)

# Empty cache of GPU
torch.cuda.empty_cache()

# Create folder to save model weights
if not os.path.isdir(f"{model_weight_path}/{dataset_name}/"):
    os.mkdir(f"{model_weight_path}/{dataset_name}/")
early_stop = pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=10)
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, dirpath=f"{model_weight_path}", 
                                                   filename=f"{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}", 
                                                   monitor="train_loss", mode="min")

# Define trainer for each model
trainer = pl.Trainer(accelerator='auto', 
                    callbacks=[early_stop, checkpoint_callback], 
                    max_epochs=500, logger=True)


# Start training and save best model
trainer.fit(model, tr_loader)
print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
print(checkpoint_callback.best_model_score) # and prints it score
best_model = model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
torch.save(torch.load(checkpoint_callback.best_model_path)["state_dict"], f"{model_weight_path}/{dataset_name}/{gnn}_weighted_{weighted_graph}_GAG{gag}_PAG{pag}_{test_dataset.te_names}_{gene_list}.ckpt")
os.remove(checkpoint_callback.best_model_path)

# Inference
gc.collect()
out = trainer.predict(model, te_loader)
pred = ad.AnnData(np.concatenate([out[i][0] for i in range(len(out))]))
gt_exp = np.concatenate([out[i][1] for i in range(len(out))])
gt = sc.concat([test_dataset.meta_dict[sub_slide] for sub_slide in list(test_dataset.meta_dict.keys())])[:,target_gene_list]
gt.X = gt_exp
        
# Add the gene list to AnnData    
pred.var_names = target_gene_list
gt.var_names = target_gene_list

# Save AnnData to H5AD file
if not os.path.isdir(f"{res_path}/{dataset_name}/"):
    os.mkdir(f"{res_path}/{dataset_name}/")
pred.write(f"{res_path}/{dataset_name}/pred_{gnn}_weighted_{weighted_graph}_GAG{gag}_PAG{pag}_{test_dataset.te_names}_{gene_list}.h5ad")
gt.write(f"{res_path}/{dataset_name}/gt_{gnn}_weighted_{weighted_graph}_GAG{gag}_PAG{pag}_{test_dataset.te_names}_{gene_list}.h5ad")
gc.collect()

# Save spatial location to numpy array
spatial_loc = np.concatenate([test_dataset.meta_dict[key].obsm["spatial"] for key in list(test_dataset.meta_dict.keys())])
np.save(f'{res_path}/{dataset_name}/spatial_loc_{gnn}_weighted_{weighted_graph}_GAG{gag}_PAG{pag}_{test_dataset.te_names}_{gene_list}.npy', spatial_loc)
gc.collect()

# Remove the log file after all models are trained
print("Finish training!")

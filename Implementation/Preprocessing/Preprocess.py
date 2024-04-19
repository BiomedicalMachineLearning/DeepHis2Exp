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
from Models.DeepPT.preprocess_deeppt import *
from Dataloader.Dataset_wiener import *

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="raw", help='Color normalization methods.')
parser.add_argument('--dataset_name', type=str, default="Skin_cSCC", help='Dataset choice.')
parser.add_argument('--model_name', type=str, default="deeppt", help='Model choice.')
parser.add_argument('--gene_list', type=str, default="hvg", help='Gene list choice.')
parser.add_argument('--exp_norm', type=str, default="log1p", help='Gene expression preprocessing choice.')
parser.add_argument('--augment', type=str, default="none", help='Data augmentation choice.')
parser.add_argument('--hpc', type=str, default="wiener", help='Clusters choice')
args = parser.parse_args()

fold = args.fold
seed = args.seed
colornorm = args.colornorm
augment = args.augment
dataset_name = args.dataset_name
model_name = args.model_name
exp_norm = args.exp_norm
gene_list = args.gene_list
hpc = args.hpc

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

print("Start training!")
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
# Train dataset is windowed WSI dataset, test dataset is WSI dataset

# Functional genes for visium dataset
target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_{gene_list}_{dataset_name}.npy', allow_pickle=True))

# Load sample names
full_train_dataset = CSCC(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, gene_list=target_gene_list)
test_dataset = CSCC(train=False,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, gene_list=target_gene_list)
gc.collect()
tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define model and train
model = DeepPT(n_genes=len(test_dataset.gene_list), hidden_dim=512, learning_rate=1e-4,
               trans=augment)
    
# Empty cache of GPU
torch.cuda.empty_cache()

# Create folder to save model weights
if not os.path.isdir(f"{model_weight_path}/preprocessing/"):
    os.mkdir(f"{model_weight_path}/preprocessing/")
early_stop = pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=30)
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
torch.save(torch.load(checkpoint_callback.best_model_path)["state_dict"], f"{model_weight_path}/preprocessing/{model_name}_{augment}_{exp_norm}_{colornorm}_{test_dataset.te_names}_{gene_list}.ckpt")
os.remove(checkpoint_callback.best_model_path)

# Inference
gc.collect()
target_gene_list = test_dataset.gene_list
out = trainer.predict(model, te_loader)
pred = ad.AnnData(np.concatenate([out[i][0] for i in range(len(out))]))
gt_exp = np.concatenate([out[i][1] for i in range(len(out))])
gt = sc.concat([test_dataset.meta_dict[sub_slide] for sub_slide in list(test_dataset.meta_dict.keys())])[:,target_gene_list]
gt.X = gt_exp
        
# Add the gene list to AnnData    
pred.var_names = test_dataset.gene_list
gt.var_names = test_dataset.gene_list

# Save AnnData to H5AD file
if not os.path.isdir(f"{res_path}/preprocessing/"):
    os.mkdir(f"{res_path}/preprocessing/")
pred.write(f"{res_path}/preprocessing/pred_{model_name}_{augment}_{exp_norm}_{colornorm}_{test_dataset.te_names}_{gene_list}.h5ad")
gt.write(f"{res_path}/preprocessing/gt_{model_name}_{augment}_{exp_norm}_{colornorm}_{test_dataset.te_names}_{gene_list}.h5ad")
gc.collect()

# Save spatial location to numpy array
spatial_loc = np.concatenate([test_dataset.meta_dict[key].obsm["spatial"] for key in list(test_dataset.meta_dict.keys())])
np.save(f'{res_path}/preprocessing/spatial_loc_{model_name}_{augment}_{exp_norm}_{colornorm}_{test_dataset.te_names}_{gene_list}.npy', spatial_loc)
gc.collect()

# Remove the log file after all models are trained
# shutil.rmtree(f'{abs_path}/lightning_logs')
print("Finish training!")

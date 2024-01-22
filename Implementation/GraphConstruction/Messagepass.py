
# %%
import sys
sys.path.append("../../")
import argparse
import os
import gc
import torch
import random
import cv2
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import anndata as ad
import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from Dataloader.Dataset_wiener import *
from HIST2ST_GNN import *

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--gnn', type=str, default="GIN", help='Graph neural network.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="reinhard", help='Color normalization methods.')
parser.add_argument('--dataset_name', type=str, default="BC_visium", help='Dataset choice.')
parser.add_argument('--gene_list', type=str, default="func", help='Gene list choice.')
parser.add_argument('--hpc', type=str, default="wiener", help='Clusters choice')
args = parser.parse_args()

fold = args.fold
seed = args.seed
gnn = args.gnn
dataset_name = args.dataset_name
colornorm = args.colornorm
gene_list = args.gene_list
hpc = args.hpc

exp_norm = "lognorm"

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

tag = '5-7-2-8-4-16-32'
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))

# Load train and test dataset and wrap dataloader
# Train dataset is windowed WSI dataset, test dataset is WSI dataset

# Functional genes for visium dataset
target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_{gene_list}_{dataset_name}.npy', allow_pickle=True))

# For Alex and 10x dataset
full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, 
                                       exp_norm=exp_norm, num_subsets=10, gene_list=target_gene_list, shuffle=False,)
test_dataset = BC_visium(train=False,fold=fold,r=112, neighs=8, color_norm=colornorm, 
                                 exp_norm=exp_norm, num_subsets=10, gene_list=target_gene_list, shuffle=False,)
gc.collect()
    
tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define model and train
model = Hist2ST_GNN(
    depth1=depth1, depth2=depth2,
    n_genes=len(target_gene_list), learning_rate=1e-5, label=None, 
    kernel_size=kernel, patch_size=patch, n_pos=256,
    heads=heads, channel=channel, dropout=0.2,
    zinb=0.25, nb=False, gnn=gnn,
    bake=5, lamb=0.5,
)

# Empty cache of GPU
torch.cuda.empty_cache()

# Create folder to save model weights
if not os.path.isdir(f"{model_weight_path}/Messagepass/"):
    os.mkdir(f"{model_weight_path}/Messagepass/")
early_stop = pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=5)
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, dirpath=f"{model_weight_path}", 
                                                   filename=f"{gnn}_{test_dataset.te_names}", 
                                                   monitor="train_loss", mode="min")
trainer = pl.Trainer(accelerator='auto', 
                        callbacks=[early_stop, checkpoint_callback], 
                        max_epochs=350, logger=True)

print("debug")

# Start training and save best model
trainer.fit(model, tr_loader)
print(checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
print(checkpoint_callback.best_model_score) # and prints it score
best_model = model.load_state_dict(torch.load(checkpoint_callback.best_model_path)["state_dict"])
torch.save(torch.load(checkpoint_callback.best_model_path)["state_dict"], f"{model_weight_path}/Messagepass/Hist2ST_BC_visium_{gnn}_{test_dataset.te_names}.ckpt")
os.remove(checkpoint_callback.best_model_path)


# %%
gc.collect()

out = trainer.predict(model, te_loader)
pred = ad.AnnData(np.concatenate([out[i][0] for i in range(len(out))]))
gt_exp = np.concatenate([out[i][1] for i in range(len(out))])
gt = sc.concat([test_dataset.meta_dict[sub_slide] for sub_slide in list(test_dataset.meta_dict.keys())])[:,target_gene_list]
gt.X = gt_exp
        
# Add the gene list to AnnData    
pred.var_names = test_dataset.gene_list
gt.var_names = test_dataset.gene_list

# Save AnnData to H5AD file
if not os.path.isdir(f"{res_path}/Messagepass/"):
    os.mkdir(f"{res_path}/Messagepass/")
pred.write(f"{res_path}/Messagepass/pred_{gnn}_{test_dataset.te_names}.h5ad")
gt.write(f"{res_path}/Messagepass/gt_{gnn}_{test_dataset.te_names}.h5ad")
gc.collect()

# Save spatial location to numpy array
spatial_loc = np.concatenate([test_dataset.meta_dict[key].obsm["spatial"] for key in list(test_dataset.meta_dict.keys())])
np.save(f'{res_path}/Messagepass/spatial_loc_{gnn}_{test_dataset.te_names}.npy', spatial_loc)
gc.collect()

# Remove the log file after all models are trained
print("Finish training!")
gc.collect()






# %%
import sys
sys.path.append("../../")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
# parser.add_argument('--seed', type=int, default=42, help='random seed.')
# parser.add_argument('--gnn', type=str, default="GIN", help='Graph neural network.')
parser.add_argument('--pag', type=bool, default=False, help='Pathology annotation graph.')
parser.add_argument('--gag', type=bool, default=False, help='Gene association graph.')
parser.add_argument('--hsg', type=bool, default=False, help='Histology similarity graph.')
# parser.add_argument('--colornorm', type=str, default="reinhard", help='Color normalization methods.')
# parser.add_argument('--dataset_name', type=str, default="SCC_Chenhao", help='Dataset choice.')
# parser.add_argument('--model_name', type=str, default="hist2st", help='Model choice.')


args = parser.parse_args()

fold = args.fold
# seed = args.seed
# gnn = args.gnn
PAG = args.pag
GAG = args.gag
HSG = args.hsg
print(f"PAG:{PAG}, GAG:{GAG}")
# colornorm = args.colornorm
# dataset_name = args.dataset_name
# model_name = args.model_name

# fold = 0
seed = 42
colornorm = "reinhard"    # "reinhard", "raw"
dataset_name = "BC_visium"    # "BC_visium", "SCC_Chenhao"
model_name = "hist2st_GraphBuild"      # "hist2st", "stnet", "histogene"
gnn = "GCN"     # "GCN", "GIN", "GAT"
# PAG = True
# GAG = True
# HSG = False

print("Fold:", fold)
print("Color normalization method:", colornorm)
print("Model_name:", model_name)
print("Dataset_name:", dataset_name)
print(f"PAG:{PAG}, GAG:{GAG}")
# %%
import os
import gc
import torch
import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from Models.Hist2ST.hist2st import *
# from Models.STnet.stnet import *
# from Models.HistoGene.histogene import *
from HIST2ST_dgl import *
from Dataloader.Dataset import *


# For reproducing the results
seed_everything(seed)

tag = '5-7-2-8-4-16-32'
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))

# Load train and test dataset and wrap dataloader
# Train dataset is windowed WSI dataset, test dataset is WSI dataset
# Load gene list
gene_list = pd.read_excel('../../Predictable_genes/Marker_list/Intersection_marker_genes.xlsx')["gene_name"].to_list()

# Load sample names
if dataset_name == "SCC_Chenhao":
    # For SCC_Chenhao dataset
    data_path = "../../Dataset/SCC_Chenhao/"
    name_slides = os.listdir(data_path)
    full_train_dataset, test_dataset = SCC_Chenhao_Visium(test_sample=name_slides[fold], sizes = 2700, subset=28, n_subsets = 5, exp_norm="lognorm", 
                                                        GAG=False, PAG=False, n_neighbors=8, data_path=data_path, gene_list=gene_list)

elif dataset_name == "BC_visium":
    # For Alex and 10x dataset
    samps1 = ["1142243F", "1160920F", "CID4290", "CID4465", "CID44971", "CID4535", ]
    samps2 = ["block1", "block2", "FFPE"]
    name_slides = samps1+samps2
    if model_name == "hist2st_GraphBuild":
        full_train_dataset, test_dataset = Breast_Visium(test_sample = name_slides[fold], sizes = 2700, subset=9, n_subsets = 10, 
            exp_norm="lognorm", n_neighbors=8, gene_list=gene_list, GAG=GAG, PAG=PAG, color_norm=colornorm, distance_mode="distance",
            GraphConstruction=True)
    else:
        full_train_dataset, test_dataset = Breast_Visium(test_sample = name_slides[fold], sizes = 2700, subset=9, n_subsets = 10, 
        exp_norm="lognorm", n_neighbors=8, gene_list=gene_list, GAG=GAG, PAG=PAG, color_norm=colornorm, distance_mode="distance",
        GraphConstruction=False)

gc.collect()
    
# For real dataset only
# train_size = int(0.7 * len(full_train_dataset))
# valid_size = len(full_train_dataset) - train_size
# train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, valid_size])

tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
# tr_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define model and train
if model_name == "hist2st":
    model = Hist2ST(
        depth1=depth1, depth2=depth2,
        n_genes=len(gene_list), learning_rate=1e-5, label=None, 
        kernel_size=kernel, patch_size=patch, n_pos=256,
        heads=heads, channel=channel, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5,)
elif model_name == "hist2st_GraphBuild":
    model = Hist2ST_dgl(
        depth1=depth1, depth2=depth2,
        n_genes=len(gene_list), learning_rate=1e-5, label=None, 
        kernel_size=kernel, patch_size=patch, n_pos=256,
        heads=heads, channel=channel, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5,)
elif model_name == "stnet":
    model = STModel(n_genes=len(gene_list), hidden_dim=1024, learning_rate=1e-6)
elif model_name == "histogene":
    model = HisToGene(patch_size=224, n_layers=4, n_genes=len(gene_list), dim=1024, 
                      learning_rate=1e-5, dropout=0.1, n_pos=256)

if not os.path.isdir("./model/"):
    os.mkdir("./model/")
if not os.path.isdir("./logs/"):
    os.mkdir("./logs/")
logger = pl.loggers.CSVLogger("./logs", name=f"Hist2ST_{dataset_name}_{name_slides[fold]}_PAG_{PAG}_GAG_{GAG}")

torch.cuda.empty_cache()
trainer = pl.Trainer(accelerator='auto', 
                     callbacks=[EarlyStopping(monitor='train_loss', mode='min', patience=30)], 
                     max_epochs=350, logger=logger)

trainer.fit(model, tr_loader)
torch.save(model.state_dict(), f"./model/BC_visium_{colornorm}_{name_slides[fold]}_PAG_{PAG}_GAG_{GAG}.ckpt")


# %%
import anndata as ad
import gc

gc.collect()
out = trainer.predict(model, te_loader)
pred = ad.AnnData(out[0][0])
pred.var_names = gene_list
gt = ad.AnnData(out[0][1])
gt.var_names = gene_list

# # Save AnnData to H5AD file
if not os.path.isdir("./Results/"):
    os.mkdir("./Results/")
if not os.path.isdir(f"./Results/{dataset_name}"):
    os.mkdir(f"./Results/{dataset_name}")
pred.write(f"./Results/{dataset_name}/pred_{dataset_name}_{name_slides[fold]}_PAG_{PAG}_GAG_{GAG}.h5ad")
gt.write(f"./Results/{dataset_name}/gt_{dataset_name}_{name_slides[fold]}_PAG_{PAG}_GAG_{GAG}.h5ad")
gc.collect()


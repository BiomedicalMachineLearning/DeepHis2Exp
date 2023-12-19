
# %%
import sys
sys.path.append("../../")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--gnn', type=str, default="GIN", help='Graph neural network.')
parser.add_argument('--pag', type=bool, default=False, help='Pathology annotation graph.')
parser.add_argument('--gag', type=bool, default=False, help='Gene association graph.')
parser.add_argument('--hsg', type=bool, default=False, help='Histology similarity graph.')

args = parser.parse_args()


fold = args.fold
seed = args.seed
gnn = args.gnn
PAG = args.pag
GAG = args.gag
HSG = args.hsg
if PAG:
    graph = "PAG"
if GAG:
    graph = "GAG"
if HSG:
    graph = "HSG"
print(f"PAG:{PAG}, GAG:{GAG}, HSG:{HSG}")
print(graph)

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

from Dataloader.Dataset import *
from Models.GraphConstruction.HIST2ST import *

# For reproducing the results
seed_everything(seed)

tag = '5-7-2-8-4-16-32'
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))

# Load train and test dataset and wrap dataloader
# Train dataset is windowed WSI dataset, test dataset is WSI dataset
# Load gene list
gene_list = pd.read_excel('/afm03/Q2/Q2051/DeepHis2Exp/Predictable_genes/Marker_list/Intersection_marker_genes.xlsx')["gene_name"].to_list()

# Load sample names
# # For SCC_Chenhao dataset
# data_path = "../../Dataset/SCC_Chenhao/"
# name_slides = os.listdir(data_path)

# For Alex and 10x dataset
samps1 = ["1142243F", "1160920F", "CID4290", "CID4465", "CID44971", "CID4535", ]
samps2 = ["block1", "block2", "FFPE"]
name_slides = samps1+samps2

# full_train_dataset, test_dataset = SCC_Chenhao_Visium(test_sample=name_slides[0], sizes = 2700, n_subset=2, n_subsets = 5, exp_norm="lognorm", 
#                                                       GAG=False, PAG=False, n_neighbors=0, data_path=data_path, gene_list=gene_list)
full_train_dataset, test_dataset = Breast_Visium(test_sample = name_slides[fold], sizes = 2700, subset=9, n_subsets = 10, exp_norm="lognorm", n_neighbors=10, gene_list=gene_list, GAG=GAG, PAG=PAG, color_norm="raw")
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
# Benchmark_MessagePass
model = Hist2ST(
    depth1=depth1, depth2=depth2,
    n_genes=len(gene_list), learning_rate=1e-5, label=None, 
    kernel_size=kernel, patch_size=patch, n_pos=256,
    heads=heads, channel=channel, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5,
)

if not os.path.isdir("./model/"):
    os.mkdir("./model/")
if not os.path.isdir("./logs/"):
    os.mkdir("./logs/")
logger = pl.loggers.CSVLogger("./logs", name=f"Hist2ST_BC_visium_{graph}_{name_slides[fold]}")

torch.cuda.empty_cache()
trainer = pl.Trainer(accelerator='auto', 
#                      callbacks=[EarlyStopping(monitor='train_loss', mode='min', patience=30)], 
                     max_epochs=350, logger=logger)

trainer.fit(model, tr_loader)
torch.save(model.state_dict(),f"./model/Hist2ST_BC_visium_{graph}_{name_slides[fold]}.ckpt")


# %%
import anndata as ad
import gc

gc.collect()
out = trainer.predict(model, te_loader)
pred = ad.AnnData(out[0][0])
pred.var_names = gene_list
gt = ad.AnnData(out[0][1])
gt.var_names = gene_list

# Save AnnData to H5AD file
pred.write(f"./Results/BC_visium/pred_Hist2st_BC_visium_{graph}_{name_slides[fold]}.h5ad")
gt.write(f"./Results/BC_visium/gt_Hist2st_BC_visium_{graph}_{name_slides[fold]}.h5ad")
gc.collect()




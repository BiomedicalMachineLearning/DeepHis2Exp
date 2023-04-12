import sys
sys.path.append("./")
# sys.path.append("../../scripts/")

import time

from window_adata import *
from read_stimage_genes import read_gene_set_hvg,intersect_section_genes

import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict import *
from HIST2ST import *
from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
import pickle
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
import h5py
import json
from matplotlib.image import imread
import matplotlib.pyplot as plt

import anndata
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,
)
from anndata import read as read_h5ad
from anndata import read_h5ad
import scanpy as sc
from scanpy import read_visium, read_10x_mtx

import glob
import torch
import torchvision
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None



def calculate_correlation(attr_1, attr_2):
    r = spearmanr(attr_1, 
                       attr_2)[0]
    return r


data_dir1 = "../../data/pfizer/"

samps1 = ["VLP79_D","VLP82_A","VLP79_A","VLP80_D","VLP83_A","VLP80_A","VLP81_A","VLP82_D","VLP83_D","VLP78_A"]

samples1 = {i:data_dir1 + i for i in samps1}

adata_dict1 = {name: read_visium(path, library_id = name, source_image_path = path + f"/image.tif") for name,path in samples1.items()}

adata_dict0 = {**adata_dict1}

for k,v in adata_dict0.items():
    v.obsm["spatial"] = v.obsm["spatial"].astype(np.int64)
    v.obs[['in_tissue','array_row','array_col']] = v.obs[['in_tissue','array_row','array_col']].astype(np.int64)


sizes = [3000 for i in range(len(adata_dict0))]

adata_dict = window_adata(adata_dict0, sizes)

gene_list = read_gene_set_hvg("../../data/pfizer/") # train_adata.h5ad
# gene_list = ['CD4', 'TRAC', 'CXCR4']
# gene_list = set(gene_list)

gene_list = intersect_section_genes(gene_list, adata_dict)
n_genes = len(gene_list)
print("number of genes: ", n_genes)
with open('../../data/hist2st_hvg1000.pickle', 'wb') as f:
    pickle.dump(gene_list, f)



from data_vit import ViT_Anndata


device='cuda'
tag='5-7-2-8-4-16-32'
k,p,d1,d2,d3,h,c=map(lambda x:int(x),tag.split('-'))
dropout=0.2
random.seed(12000)
np.random.seed(12000)
torch.manual_seed(12000)
torch.cuda.manual_seed(12000)
torch.cuda.manual_seed_all(12000)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

df = pd.DataFrame()
i = int(sys.argv[1])
# i=1
fold = i
test_sample = samps1[fold]
fold2name = dict(enumerate(samps1))



# gene_list = ["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
#            "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
# genes = len(gene_list)

train_set = list(set(list(adata_dict.keys())) - set([i for i in list(adata_dict.keys()) if test_sample in i]))

trainset = ViT_Anndata(adata_dict = adata_dict, train_set = train_set, gene_list = gene_list,
            train=True,flatten=False,adj=True,ori=True,prune='NA',neighs=4, 
        )

print("LOADED TRAINSET")


train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

model=Hist2ST(
    depth1=d1, depth2=d2,depth3=d3,n_genes=n_genes,
    kernel_size=k, patch_size=p,
    heads=h, channel=c, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, n_pos=128,
)
model.load_state_dict(torch.load(f'./model/31-Hist2ST.ckpt'))
logger=None
trainer = pl.Trainer(
    gpus=[0], max_epochs=100, ########################### Changed from 350
    logger=logger,
)

start_train = time.perf_counter()
trainer.fit(model, train_loader)

end_train = time.perf_counter()
import os
if not os.path.isdir("../../trained_models/"):
    os.mkdir("../../trained_models/")

torch.save(model.state_dict(),f"../../trained_models/{test_sample}-Hist2ST_pretrained.ckpt")

# Some local variable referencing error when model is inside function
##########################
# def evall(test_sample, gene_list):
    
#     train_set = list(set(list(adata_dict.keys())) - set(test_sample))
#     testset = ViT_Anndata(adata_dict = adata_dict, train_set = train_set, gene_list = gene_list,
#                 train=False,flatten=False,adj=True,ori=True,prune='NA',neighs=4, 
#             )
#     test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

#     adata_pred, adata_truth = test(model, test_loader,'cuda')

#     adata_pred.var_names = gene_list
#     adata_truth.var_names = gene_list

#     pred_adata = adata_pred.copy()
#     test_dataset = adata_truth.copy()
    
#     test_sample = ','.join(list(test_sample))
    
#     with open(f"../../results/pf/hist2st_preds_{test_sample}_{i}.pkl", 'wb') as f:
#         pickle.dump([pred_adata,test_dataset], f)

#     for gene in pred_adata.var_names:
#         cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
#         df = df.append(pd.Series([gene, cor_val, test_sample, "Hist2ST"], 
#                              index=["Gene", "Pearson correlation", "Slide", "Method"]),
#                   ignore_index=True)

#     del model
#     torch.cuda.empty_cache()

#     df.to_csv("../../results/pf/hist2st_cor_{}_{i}.csv".format(test_sample, i))

#     with open("../../results/pf/hist2st_times.txt", 'a') as f:
#         f.write(f"{i} {test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")



# evall([i for i in list(adata_dict.keys()) if 'VLP78_A' in i], gene_list)

def test(model,test,device='cuda'):
    model=model.to(device)
    model.eval()
    preds=None
    ct=None
    gt=None
    loss=0
    adatas,adata_gts = [],[]
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(test):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
            adata = ad.AnnData(preds)
            adata.obsm['spatial'] = ct
            adata_gt = ad.AnnData(gt)
            adata_gt.obsm['spatial'] = ct
           
            adatas.append(adata)
            adata_gts.append(adata_gt)
    adata = ad.concat(adatas)
    adata_gt = ad.concat(adata_gts)
    return adata,adata_gt



test_sample = [i for i in list(adata_dict.keys()) if test_sample in i]
train_set = list(set(list(adata_dict.keys())) - set(test_sample))
testset = ViT_Anndata(adata_dict = adata_dict, train_set = train_set, gene_list = gene_list,
            train=False,flatten=False,adj=True,ori=True,prune='NA',neighs=4, 
        )
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)


adata_pred, adata_truth = test(model, test_loader,'cuda')

adata_pred.var_names = gene_list
adata_truth.var_names = gene_list

pred_adata = adata_pred.copy()
test_dataset = adata_truth.copy()

test_sample = ','.join(list(test_sample))

with open(f"../../results/pf_cv/hist2st_preds_{test_sample}_pretrained.pkl", 'wb') as f:
    pickle.dump([pred_adata,test_dataset], f)

for gene in pred_adata.var_names:
    pred = pred_adata.to_df().loc[:,gene]
    pred = pred.fillna(0)
    cor_val = calculate_correlation(pred, test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, test_sample, "Hist2ST"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
              ignore_index=True)

del model
torch.cuda.empty_cache()

df.to_csv("../../results/pf_cv/hist2st_cor_{}_pretrained.csv".format(test_sample))

with open("../../results/pf_cv/hist2st_times_pretrained.txt", 'a') as f:
    f.write(f"{i} {test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")


# gene_list = ["COX6C","TTLL12", "HSP90AB1", 
#            "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
# evall("CytAssist_FFPE", gene_list)

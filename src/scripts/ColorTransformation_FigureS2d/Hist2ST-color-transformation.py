#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ImgProcess', type=str, default="HE-Intensity", help='[HE-Intensity, RandStainNA, Reinhard]')


# In[2]:


import os
import torch
import random
import argparse
import pickle as pk
import pytorch_lightning as pl
from utils import *
from HIST2ST import Hist2ST
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append("./")

import time
from window_adata import *
import os
import sys
import numpy as np
import pandas as pd 
import h5py
import json
import pickle
import anndata
import scanpy as sc
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import glob
import torchvision
import scprep as scp
import anndata as ad
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from scipy.stats import pearsonr,spearmanr
from scipy import stats
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
from matplotlib.image import imread
from scanpy import read_visium, read_10x_mtx
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from anndata import read as read_h5ad
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
from matplotlib.image import imread
import pickle

from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,)



# In[3]:


""" Integrate two visium datasets """
data_dir1 = "./Alex_NatGen_6BreastCancer/"
data_dir2 = "./breast_cancer_10x_visium/"

samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
samps2 = ["block1", "block2", "FFPE"]

sampsall = samps1 + samps2
samples1 = {i:data_dir1 + i for i in samps1}
samples2 = {i:data_dir2 + i for i in samps2}

# Marker gene list
gene_list = ["COX6C","TTLL12", "HSP90AB1", "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

# Load windowed dataset
import pickle
with open('/scratch/imb/uqyjia11/Yuanhao/10X_visium/CNN_GAT/10x_visium_dataset_without_window.pickle', 'rb') as f:
    adata_dict0 = pickle.load(f)
    
# Define the gridding size
sizes = [4000 for i in range(len(adata_dict0))]

# Split tiles into smaller patches according to gridding size
adata_dict = window_adata(adata_dict0, sizes)

# Specify the train and test sample name
train_sample = samps1
test_sample = samps2

# Specify the windowed samples name
tr_name = list(set([i for i in list(adata_dict.keys()) for tr in train_sample if tr in i]))
te_name = list(set([i for i in list(adata_dict.keys()) for te in test_sample if te in i]))



# In[4]:


# Set random seed
def setup_seed(seed=12000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
# Compute pearson correlation at gene level
def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

# For OOD dataset training
from data_vit import ViT_Anndata

def dataset_wrap(processed_img=None, dataloader= True):
    train_sample = list(set(samps1)-set(["1160920F","CID4290"])) # Alex visium samples
    tr_name = list(set([i for i in list(adata_dict.keys()) for tr in train_sample if tr in i]))
    trainset = ViT_Anndata(adata_dict = adata_dict, train_set = tr_name, gene_list = gene_list, processed_img=processed_img, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    print("LOADED TRAINSET")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    if dataloader==True:
        return train_loader
    else:
        return trainset


# # Data Augmentation Comparison

# In[5]:


# processed_img=None
# AugNorm="aug"
# ImgProcess="HE-Intensity"
# ImgProcess="RandStainNA"
# ImgProcess="Reinhard"


# In[6]:


import pickle
source_path = "/scratch/imb/uqyjia11/Yuanhao/10X_visium/Augmentation_comparison/Processed_tiles/"
# Specify the path to the trainset pickle file
file_path = source_path + f'Train-aug-{args.ImgProcess}-samples.pkl'

# Load the dictionary from the pickle file
with open("/scratch/imb/uqyjia11/Yuanhao/10X_visium/Augmentation_comparison/Processed_tiles/Train-aug-RandStainNA-samples.pkl", 'rb') as f:
    dict1 = pickle.load(f)

processed_img = dict1


# In[7]:


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import *

"""For Hist2ST OOD training only (tr4 val2 te3)"""
import gc
from data_vit import ViT_Anndata
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

"""Training loops"""
seed=12000
epochs=350
fold=0

"""Load dataset"""
train_loader = dataset_wrap(processed_img=processed_img, dataloader=True)



# In[ ]:


"""Define model"""
model = Hist2ST(
    depth1=2, depth2=8, depth3=4,n_pos=128,
    n_genes=len(gene_list), learning_rate=1e-5,
    kernel_size=5, patch_size=7, fig_size=112,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
    policy='mean', )

setup_seed(seed)

start_time = time.time()
"""Setup trainer"""
trainer = pl.Trainer(accelerator='auto', callbacks=[EarlyStopping(monitor='Train_loss',mode='min')], max_epochs=epochs, logger=False)
trainer.fit(model, train_loader)

"""Save model and clean memory"""
torch.save(model.state_dict(),f"./model/Hist2st-{args.ImgProcess}-OOD.ckpt")
gc.collect()
# del train_loader, model
    
end_time = time.time()
execution_time = end_time - start_time
print("Training time: ", execution_time/3600, " hours")


# In[ ]:

# For testing
from data_vit import ViT_Anndata

def dataset_wrap(fold = 0, dataloader= True):
    test_sample = sampsall[fold]
    print(f"Test sample: {test_sample}")
    te_name = list(set([i for i in list(adata_dict0.keys()) if test_sample in i]))
    adata_dict = {}
    for k in adata_dict0.keys():
        v = adata_dict0[k]
        v.var_names_make_unique()
        adata_dict[k] = v

    testset = ViT_Anndata(adata_dict = adata_dict, train_set = te_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    if dataloader==True:
        return test_loader
    else:
        return testset
    
def test(name, model, test_loader):
    model.eval()
    device = 'cpu'
    for patch, center, exp, adj, label, oris, sfs, *_ in test_loader:
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        ct = center.squeeze().cpu().numpy()
        gts = exp.squeeze().cpu().numpy()

        """Model inference"""
        gene_exp,*_,h = model(patch, center, adj)

        """Put prediction into adata"""
        # Marker gene list
        gene_list = ["COX6C","TTLL12", "HSP90AB1", "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

        pred = ad.AnnData(gene_exp.cpu().detach().numpy())
        
        pred.var_names = gene_list
        pred.obsm['spatial'] = ct

        """Regression loss"""
        mse_loss = nn.MSELoss(reduction="none")(gene_exp, exp).mean(1)
        pred.obs['MSE'] = mse_loss.cpu().detach().numpy()
        
        """Attention weights"""
        norm = nn.LayerNorm(1024)
        multihead_attn = nn.MultiheadAttention(embed_dim=1024, num_heads=1)
        _, attn_output_weights = multihead_attn(query=h, key=h, value=h)
        attention_weight = attn_output_weights.sum(axis=0) / attn_output_weights.sum()
        pred.obs['attention weights'] = attention_weight.cpu().detach().numpy()
        
        """Pearson correlation coeficient"""
        adata1 = ad.AnnData(gene_exp.cpu().detach().numpy())
        adata2 = ad.AnnData(exp.cpu().detach().numpy())
        R=get_R(adata1,adata2)[0]
        mean_pcc=np.nanmean(R)
        pred.var["Pearson correlation"]=R
        print(f"Mean_PCC:{mean_pcc}")
    return pred

"""For OOD test only"""
import gc
from HIST2ST import *
from data_vit import ViT_Anndata
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

start_time = time.time()
"""Training loops"""
for fold in range(9):
    """print sample name"""
    name = sampsall[fold]
    
    """Reproducibility"""
    setup_seed(12000)
    
    """Load dataset"""
    test_loader = dataset_wrap(fold=fold, dataloader= True)
    
    """Define model"""
    model = Hist2ST(
    depth1=2, depth2=8, depth3=4,n_pos=128,
    n_genes=len(gene_list), learning_rate=1e-5,
    kernel_size=5, patch_size=7, fig_size=112,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
    policy='mean', )
    model.load_state_dict(torch.load(f"./model/Hist2st-{args.ImgProcess}-OOD.ckpt"))

    """Test model performance"""
    pred = test(name, model, test_loader)
    df = pred.var
    df["Gene"] = df.index
    df["Slide"] = name
    df["Method"] = f"Hist2ST({args.ImgProcess})"
    import os
    directory = f"./Results/PCC_table"
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    df.to_csv(f"./Results/PCC_table/{name}-Hist2st-{args.ImgProcess}-OOD.csv")
    gc.collect()
    del test_loader, model
end_time = time.time()
execution_time = end_time - start_time
print("Inference time: ", execution_time/3600, " hours")

# For OOD performance (Train 4 samples, validate 2 samples, test 3 samples)
df1 = []
for name in ["1160920F","CID4290"]:
    df = pd.read_csv(f"./Results/PCC_table/{name}-Hist2st-{args.ImgProcess}-OOD.csv", index_col=0)
    df1.append(df)
df_train = pd.concat(df1).reset_index()
df_train["Kind"] = "Training dataset (6 samples)"

df1 = []
for name in samps2:
    df = pd.read_csv(f"./Results/PCC_table/{name}-Hist2st-{args.ImgProcess}-OOD.csv", index_col=0)
    df1.append(df)
df_test = pd.concat(df1).reset_index()
df_test["Kind"] = "Test dataset (3 samples)"

df_all = pd.concat([df_train, df_test]).reset_index()
df_all.to_csv(f"./Results/PCC_table/Hist2st-{args.ImgProcess}-OOD.csv")
df_all


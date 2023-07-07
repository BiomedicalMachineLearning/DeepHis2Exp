#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from tqdm import tqdm
import scipy.io as sio

import torch
from torch import nn
import torch.distributed as dist
import torch.utils.data.distributed

import config as CFG
from dataset import CLIPDataset
# from models import CLIPModel, CLIPModel_ViT, CLIPModel_ViT_L, CLIPModel_CLIP, CLIPModel_resnet101, CLIPModel_resnet152
# from utils import AvgMeter
# from torch.utils.data import DataLoader

import scanpy as sc
from scanpy import read_visium, read_10x_mtx
import argparse
import pickle
import pandas as pd


#only need to run once to save hvg_matrix.npy
#filter expression matrices to only include HVGs shared across all datasets

def intersect_section_genes(adata_list):
    shared = set.intersection(*[set(adata.var_names) for adata in adata_list])
    return list(shared)

def her2_hvg_selection_and_pooling(adata_list, n_top_genes = 1000):
    shared = intersect_section_genes(adata_list)
    
    hvg_bools = []

    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        print(adata.shape)
        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        #save hvgs
        hvg = adata.var['highly_variable']
        hvg_bools.append(hvg)
        
    hvg_union = hvg_bools[0]
    hvg_intersection = hvg_bools[0] 
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]
        print(sum(hvg_intersection), sum(hvg_bools[i]))
        hvg_intersection = hvg_intersection & hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())
    print("Number of HVGs (intersection): ", hvg_intersection.sum())
    
    with open('her2_hvgs_intersection.pickle', 'wb') as handle:
        pickle.dump(hvg_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('her2_hvgs_union.pickle', 'wb') as handle:
        pickle.dump(hvg_union, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Add all the HVGs
    gene_list_path = "../../scripts/gene_list.pkl"
    with open(gene_list_path, 'rb') as f:
        gene_list = pickle.load(f)

    hvg_union[gene_list] = True

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to shared genes
        adata = adata[:, shared]
        filtered_exp_mtxs.append(adata[:, hvg_union].X.T.toarray())
    return filtered_exp_mtxs

import os
names = os.listdir("data/her2st/ST-cnts")
names.sort()
names = [i[:2] for i in names]
print(names)
print(len(names))

adata_list = [sc.AnnData(pd.read_csv(f"data/her2st/ST-cnts/{name}.tsv",sep='\t',index_col=0)) for name in names]

filtered_mtx = her2_hvg_selection_and_pooling(adata_list)

for i in range(len(filtered_mtx)):
    pathset = f"data/filtered_expression_matrices/her2/{names[i]}"
    if not(os.path.exists(pathset)):
        os.makedirs(pathset)

    np.save(f"data/filtered_expression_matrices/her2/{names[i]}/hvg_matrix_plusmarkers.npy", filtered_mtx[i])


# In[6]:


def her2_pool_gene_list(adata_list, n_top_genes = 1000):
    shared = intersect_section_genes(adata_list)
    
    hvg_bools = []

    # Add all the HVGs
    gene_list_path = "../../scripts/gene_list.pkl"
    with open(gene_list_path, 'rb') as f:
        gene_list = pickle.load(f)

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to genes
        filtered_exp_mtxs.append(adata[:, gene_list].X.T.toarray())
    return filtered_exp_mtxs


# In[7]:


adata_list = [sc.AnnData(pd.read_csv(f"data/her2st/ST-cnts/{name}.tsv",sep='\t',index_col=0)) for name in names]

filtered_mtx = her2_pool_gene_list(adata_list)

for i in range(len(filtered_mtx)):
    pathset = f"data/filtered_expression_matrices/her2_subset/{names[i]}"
    if not(os.path.exists(pathset)):
        os.makedirs(pathset)

    np.save(f"data/filtered_expression_matrices/her2_subset/{names[i]}/hvg_matrix_plusmarkers.npy", filtered_mtx[i])



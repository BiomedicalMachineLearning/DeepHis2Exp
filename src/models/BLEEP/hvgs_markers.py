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


def hvg_selection_and_pooling(exp_paths, samp_names, n_top_genes = 1000):
    #input n expression matrices paths, output n expression matrices with only the union of the HVGs

    #read adata and find hvgs
    hvg_bools = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        print(adata.shape)
        adata = sc.AnnData(X=adata.T, dtype=adata.dtype)

        # Preprocess the data
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        #save hvgs
        hvg = adata.var['highly_variable']

        hvg_bools.append(hvg)

    #find union of hvgs
    hvg_union = hvg_bools[0]
    for i in range(1, len(hvg_bools)):
        print(sum(hvg_union), sum(hvg_bools[i]))
        hvg_union = hvg_union | hvg_bools[i]

    print("Number of HVGs: ", hvg_union.sum())

    #filter expression matrices
    filtered_exp_mtxs = []
    for d in exp_paths:
        adata = sio.mmread(d)
        adata = adata.toarray()
        adata = adata[hvg_union]
        filtered_exp_mtxs.append(adata)

    return filtered_exp_mtxs

from alex import read_visium_alex

def intersect_section_genes(adata_list):
    shared = set.intersection(*[set(adata.var_names) for adata in adata_list])
    return list(shared)

def adata_hvg_selection_and_pooling(adata_list, n_top_genes = 1000):
    shared = intersect_section_genes(adata_list)
    
    hvg_bools = []

    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to the spots in barcodes.tsv
        if adata.uns['library_id'] in samps1:
            index = pd.read_csv(f"data/Alex_NatGen/{adata.uns['library_id']}/filtered_count_matrix/barcodes.tsv.gz", sep="\t", header = None)[0].tolist()
        elif adata.uns['library_id'] in samps2:
            index = pd.read_csv(f"data/Breast_Cancer_10x/{adata.uns['library_id']}/filtered_feature_bc_matrix/barcodes.tsv.gz", sep="\t", header = None)[0].tolist()
        adata = adata[index].copy()
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
    
    with open('hvgs_intersection.pickle', 'wb') as handle:
        pickle.dump(hvg_intersection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hvgs_union.pickle', 'wb') as handle:
        pickle.dump(hvg_union, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Add all the markers
    gene_list = ["COX6C","TTLL12", "HSP90AB1", 
           "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
    hvg_union[gene_list] = True

    filtered_exp_mtxs = []
    for adata in adata_list:
        adata.var_names_make_unique()
        # Subset to the spots in barcodes.tsv
        if adata.uns['library_id'] in samps1:
            index = pd.read_csv(f"data/Alex_NatGen/{adata.uns['library_id']}/filtered_count_matrix/barcodes.tsv.gz", sep="\t", header = None)[0].tolist()
        elif adata.uns['library_id'] in samps2:
            index = pd.read_csv(f"data/Breast_Cancer_10x/{adata.uns['library_id']}/filtered_feature_bc_matrix/barcodes.tsv.gz", sep="\t", header = None)[0].tolist()
        adata = adata[index].copy()
        # Subset to shared genes
        adata = adata[:, shared]
        filtered_exp_mtxs.append(adata[:, hvg_union].X.T.toarray())
    return filtered_exp_mtxs


# exp_paths = ["data/filtered_expression_matrices/1/matrix.mtx",
#             "data/filtered_expression_matrices/2/matrix.mtx",
#             "data/filtered_expression_matrices/3/matrix.mtx",
#             "data/filtered_expression_matrices/4/matrix.mtx"]

# samp_names = [1,2,3,4]


samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
# samps2 = ["block1", "block2", "FFPE", "1168993F"]
samps2 = ["block1", "block2", "FFPE"]

# exp_paths1 = [f"data/Alex_NatGen/{samp}/filtered_count_matrix/matrix.mtx.gz"
#             for samp in samps1]
samps = samps1 + samps2
paths1 =  [f"data/Alex_NatGen/{samp}"
            for samp in samps1]
adata_list1 = [read_visium_alex(path) for path in paths1]
adata_list2 = [read_visium(f"data/Breast_Cancer_10x/{samp}") for samp in samps2]
adata_list = adata_list1 + adata_list2
for i,adata in enumerate(adata_list):
    adata.uns['library_id'] = samps[i]

filtered_mtx = adata_hvg_selection_and_pooling(adata_list)

for i in range(len(filtered_mtx)):
    pathset = f"data/filtered_expression_matrices/{samps[i]}"
    if not(os.path.exists(pathset)):
        os.makedirs(pathset)

    np.save(f"data/filtered_expression_matrices/{samps[i]}/hvg_matrix_plusmarkers.npy", filtered_mtx[i])

import sys
import os

# Add the folder paths to sys.path
sys.path.append(os.path.abspath('/scratch/user/s4647285/DeepHis2Exp/Models/BLEEP'))
sys.path.append(os.path.abspath('/scratch/user/s4647285/DeepHis2Exp/Dataloader'))
sys.path.append(os.path.abspath('/scratch/user/s4647285/DeepHis2Exp/Models'))
sys.path.append(os.path.abspath('/scratch/user/s4647285/DeepHis2Exp'))

# Now you can import your scripts
import bleep as bl
import Dataset as ds
import scanpy as sc
import stlearn as st
from matplotlib.image import imread
from scanpy import read_visium, read_10x_mtx
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
from scipy.spatial import distance_matrix, minkowski_distance, distance
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict as dfd
from squidpy.im import ImageContainer
from torchvision import transforms
import gc
import cv2
import tqdm
import dgl
import glob
import torchvision
import numpy as np
import pandas as pd
import scprep as scp
import warnings
import torch
import torch.nn as nn
import anndata as ad
import json
import seaborn as sns
import scanpy as sc
import squidpy as sq
import PIL.Image as Image
import hdf5plugin
import torchstain
from Models.Hist2ST.utils import read_tiff, get_data
from Models.Hist2ST.graph_construction import calcADJ
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch_lightning as pl
import bleep as bl
from torch.utils.data import DataLoader, Subset



"""
    This script is adopted from the Liver Visium dataloader to load the Pfizer data
    Run this script for individual data via Pfizer_dataset(gene_list=gene_list, num_subsets=20, r = 112, fold = 1, sample_dir = sample)
"""

# Color normalization
target = cv2.cvtColor(
    cv2.imread(
        f"/QRISdata/Q2051/DeepHis2Exp/Dataset/Reference_Normalization/ref_HE.png"),
    cv2.COLOR_BGR2RGB)

def color_normalization(img, target, method="macenko"):
    """
    img: numpy array, RGB image
    method: str, "raw", "macenko" or "reinhard"
    target: numpy array, RGB image
    """

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Lambda(lambda x: x * 255)
    ])
    if method == "macenko":
        Normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        Normalizer.fit(T(target))
        img, Macenko_H, Macenko_E = Normalizer.normalize(I=T(img), stains=True)
    elif method == "reinhard":
        Normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
        Normalizer.fit(T(target))
        img = Normalizer.normalize(I=T(img))
    elif method == "raw":
        img = img
    return img


def subset_adatas(adata_dict, num_subsets=4, shuffle=True):
    """
    Subset the AnnData object into several subsets.
    adata_dict: dictionary, key is the name of slide, value is the adata object.
    num_subsets: int, number of subsets you want to split.
    """
    import numpy as np

    subset_adatas = {}
    for name, adata in adata_dict.items():

        
        # Get the number of observations in the full AnnData
        num_observations = adata.shape[0]

        # Set the size of each subset
        subset_size = num_observations // num_subsets
        print("subset_size:", subset_size)

        # Create an array of indices for random sampling without replacement
        random_indices = np.random.choice(
            num_observations, num_observations, replace=False)

        # Iterate through the subsets and create new AnnData objects
        for i in range(num_subsets):
            # Extract indices for the current subset
            start_index = i * subset_size
            end_index = (i + 1) * subset_size
            if shuffle:
                subset_indices = random_indices[start_index:end_index]
            else:
                subset_indices = np.arange(start_index, end_index)
            # Create the subset AnnData
            subset_anndata = adata[subset_indices, :].copy()

            # Append the subset AnnData to the list
            subset_adatas[f"{name}-{i}"] = subset_anndata
    return subset_adatas

class Pfizer_dataset(torch.utils.data.Dataset):
    """
    Some Information about Liver_visium
    patch, center, exp, adj, oris, sfs = batch
    r is half of the patch height and width
    gene_list is the list of interest of the gene
    sample_dir is where the samples are stored
    """

    def __init__(
            self,
            # train=True,
            fold=0,
            r=112,
            flatten=False,
            ori=True,
            adj=True,
            prune='Grid',
            neighs=8,
            color_norm="reinhard",
            num_subsets=100,
            target=target,
            exp_norm="log1p",
            shuffle=True,
            gene_list=None,
            sample_dir = 'VLP'
            ):
        super(Pfizer_dataset, self).__init__()

        self.data_dir = f'/QRISdata/Q1851/Wilson/Pfizer/PROCESSED_DATA/{sample_dir}' #trying for one sample first
        self.r = r
        # print(glob.glob((self.data_dir + '/*filtered.h5ad')))
        self.gene_list = gene_list
        self.names = [
            name.split('/')[-1].split(".")[0]
            for name in glob.glob(self.data_dir + '/*filtered.h5ad')] #create list of files that end with h5ad 
        self.sample_names = [name[:7] for name in self.names]
        # print("The name is", self.names, len(self.names))
        self.names.sort()
        # self.train = train
        self.ori = ori
        self.adj = adj
        samples = self.names
        # print(f"Datasize:{len(samples)}")

            

        print('Loading imgs...')
        self.img_dict = {
            sample_name: torch.Tensor(np.array(self.get_img(sample_name)))
            for sample_name in tqdm.tqdm(self.sample_names)}


        print('Loading metadata...')
        adata_dict = {
            name[:7]: sc.read_h5ad(f"{self.data_dir}/{name}.h5ad")  #path = f"{self.data_dir[:-3]}/{name}/{name}.tif" #to get '/QRISdata/Q1851/Wilson/Pfizer/PROCESSED_DATA'
            for name in tqdm.tqdm(self.names)}

        for i, m in adata_dict.items():
            adata_dict[i].var_names_make_unique()
        self.meta_dict = subset_adatas(
            adata_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = list(gene_list)
        
        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm == "raw":
            self.exp_dict = {
                i: m.to_df()[self.gene_set].values.astype(np.float64)
                for i, m in self.meta_dict.items()
            }
        elif self.exp_norm == "lognorm":
            self.exp_dict = {
                i: scp.transform.log(
                    scp.normalize.library_size_normalize(
                        m.to_df()[self.gene_set].values)).astype(np.float64)
                for i, m in self.meta_dict.items()}
        elif self.exp_norm == "norm":
            self.exp_dict = {
                i: scp.normalize.library_size_normalize(
                    m.to_df()[
                        self.gene_set].values).astype(
                    np.float64) for i,
                m in self.meta_dict.items()}
        elif self.exp_norm == "log1p":
            self.exp_dict = {
                i: scp.transform.log(
                    m.to_df()[
                        self.gene_set].values).astype(
                    np.float64)
                # i:scp.transform.log(m.to_df().values).astype(np.float64)
                for i, m in self.meta_dict.items()
            }
        elif self.exp_norm == "minmax":
            self.exp_dict = {
                i: MinMaxScaler().fit_transform(
                    m.to_df()[
                        self.gene_set].values).astype(
                    np.float64) for i,
                m in self.meta_dict.items()}
        elif self.exp_norm == "Pearson residuals":
            self.exp_dict = {}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)
                
        print("Loading gene expression completed....")
        if self.ori:
            self.ori_dict = {
                i: m.to_df()[
                    self.gene_set].values for i,
                m in self.meta_dict.items()}
            self.counts_dict = {}
            for i, m in self.ori_dict.items():
                n_counts = m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i] = sf
        self.center_dict = {
            i: np.floor(m.obsm["spatial"]).astype(int)
            for i, m in self.meta_dict.items()
        }
        print("The key values are:", self.center_dict.keys())
        self.loc_dict = {
            i: m.obs[['array_row', 'array_row']].values for i,
            m in self.meta_dict.items()}
        self.adj_dict = {
            i: torch.Tensor(
                kneighbors_graph(
                    m,
                    n_neighbors=neighs,
                    metric="euclidean",
                    mode='connectivity',
                    include_self=True).toarray()) for i,
            m in self.loc_dict.items()}
        self.patch_dict = dfd(lambda: None)
        self.patch_centers_dict = dfd(lambda: None)
        self.id2name = dict(enumerate(self.names))
        self.flatten = flatten
        self.color_norm = color_norm
        self.target = target
        self.weird_patches_dict = {}

        for ID in self.names:
            print("The ID is: ", ID)
            im = self.img_dict[ID.split("-")[0]].permute(1, 0, 2) #H W C-> W H C
            # print("The dimension of im is", im.shape)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            patch_centers = []

            if self.flatten:
                patches = torch.zeros((n_patches, patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2 * self.r, 2 * self.r))

            print("Croping WSI into patches...")
            for i in tqdm.tqdm(range(n_patches)):
                counts = 0 #just for checking
                center = centers[i]
                x, y = center
                patch_centers.append(center)
                
                
                patch = im[(x - self.r):(x + self.r),
                           (y - self.r):(y + self.r), :] #W H C
                
                if patch.shape == torch.Size([224, 224, 3]): #W H C
                    if self.flatten:
                        patches[i] = patch.flatten()
                    else:
                        patches[i] = patch.permute(2, 1, 0) # W H C -> C H W (2,1,0) because we want to have tensor input of C H W 
                        
                    self.patch_dict[ID] = patches
                
                else:
                    
                    ##fix the dimensions things tomorrow!!!
                    size = patch.shape
                    if size[1] != 224: #y is stored in the index of 1 W H C
                        print("yyy exists!!!")
                        # print(patch.shape)
                        gap = 224 - size[1]
                        padding_top = 0
                        padding_bot = gap
                    else:
                        padding_top = 0
                        padding_bot = 0
                    if size[0] != 224: #W 
                        print("xxx exists!!!")
                        # print(patch.shape)
                        gap = 224 - size[0]
                        padding_left = gap // 2
                        padding_right = gap - padding_left
                    else:
                        padding_left = 0
                        padding_right = 0

                    pad = nn.ConstantPad2d((padding_left, padding_right, padding_top, padding_bot), 0)  # Padding with 0 (or any constant value
                    # Permute the patch before padding
                    patch_permuted = patch.permute(2, 1, 0) # want to input (C, H, W) fo the next function, the current dimension is (W, H, C)
                    pad_patch = pad(patch_permuted).permute(2, 1, 0)  # Permute back to original format(C, H, W) -> (W,H,C)
                    # print(padding_left, padding_right, padding_top, padding_bot)

                    if self.flatten:
                        patches[i] = pad_patch.flatten()
                    else:
                        patches[i] = pad_patch.permute(2, 1, 0) # W H C -> C H W
                        

                    self.patch_dict[ID] = patches[i]
            self.patch_centers_dict[ID] = patch_centers


                    
        
            # self.patch_dict[ID] = patches

    def __getitem__(self, index):
        ID = self.id2name[index]
        
                # Debug information
        # print(f"Index: {index}, ID: {ID}")
        # print(f"Available IDs in patch_centers: {list(self.patch_centers_dict.keys())}")
        # print(f"Available IDs in patch_dict: {list(self.patch_dict.keys())}")

        
        
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        # print(f"Index: {index}, ID: {ID}, self.patch_centers[ID]: {patch_centers.keys()}")
        centers = self.patch_centers_dict[ID]
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        name = self.sample_names
        data = [patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data += [torch.Tensor(oris), torch.Tensor(sfs)]
        data.append(centers)
        data.append(name)
        return data

    def __len__(self):
        return len(self.meta_dict)

    def get_img(self, name):
        path = f"{self.data_dir}/{name}.tif" #to get '/QRISdata/Q1851/Wilson/Pfizer/PROCESSED_DATA'
        im = Image.open(path).convert('RGB')
        # Convert the image to a numpy array
        width = im.width
        height = im.height
        im_np = np.array(im)
        # Print the shape of the image
        # print(name)
        # print("The shape of the image is:", im_np.shape, "with width = ", width, "and height = ", height)
        return im
    
# vis_sample_list = ["VLP78_A",  "VLP78_D",  "VLP79_A",  "VLP79_D",  "VLP80_A",  "VLP80_D",  "VLP81_A",  "VLP82_A",  "VLP82_D",  "VLP83_A",  "VLP83_D"]

def BLEEP_Pfizer_data_loader(sample_list, desired_genes, batch_size = 1, shuffle = True, subset_size = 10, random_seed = 2):
    datasets = []
    #making patches for each sample
    for sample in sample_list:
        dataset = Pfizer_dataset(gene_list=desired_genes, num_subsets=subset_size, r = 112, fold = 1, sample_dir = sample)
        datasets.append(dataset)
    
    #load the data set and splitting the data into train, test, val
    train_size = int(0.8 * len(datasets))
    test_size = len(datasets) - train_size
    other_dataset, test_dataset = torch.utils.data.random_split(datasets, [train_size, test_size], generator=torch.Generator().manual_seed(random_seed))
    other_size = int(0.9* len(other_dataset))
    val_size = len(other_dataset) - other_size
    train_dataset, val_dataset = torch.utils.data.random_split(other_dataset, [other_size, val_size], generator=torch.Generator().manual_seed(random_seed))
    print(len(train_dataset), len(test_dataset),len(val_dataset))
    
    #concat
    train_dataset_cat = torch.utils.data.ConcatDataset(train_dataset)
    test_dataset_cat = torch.utils.data.ConcatDataset(test_dataset)
    val_dataset_cat = torch.utils.data.ConcatDataset(val_dataset)
    # test_dataset, val_dataset = torch.utils.data.random_split(other_dataset, [int(0.5*len(other_dataset)), int(0.5*len(other_dataset))], generator=torch.Generator().manual_seed(42))
    
    #batch problems with subsets need this function when loading the data into the model
    def cust_collate(batch_list):
        return torch.squeeze(torch.stack([i[0] for i in batch_list]), dim=0), torch.squeeze(torch.stack([i[1] for i in batch_list]), dim=0), torch.squeeze(torch.stack([i[2] for i in batch_list]), dim=0), torch.squeeze(torch.stack([i[3] for i in batch_list]), dim=0), torch.squeeze(torch.stack([i[4] for i in batch_list]), dim=0), torch.squeeze(torch.stack([i[5] for i in batch_list]), dim=0), [i[-2] for i in batch_list] ,[i[-1] for i in batch_list]
    # #put them into data loader with batches
    
    train_dataloader = DataLoader(train_dataset_cat, batch_size=batch_size, collate_fn=cust_collate, shuffle=True)
    test_dataloader = DataLoader(test_dataset_cat, batch_size=batch_size, collate_fn=cust_collate, shuffle=True)
    val_dataloader = DataLoader(val_dataset_cat, batch_size=batch_size, collate_fn=cust_collate, shuffle=True)
    
    name_array = []
    for batch in test_dataloader:
        name_array.append(batch[-1])
        
    print("The samples in the test set:", np.unique(name_array))
    
    
    return train_dataloader, test_dataloader, val_dataloader
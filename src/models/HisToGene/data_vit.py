import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def read_tiff(path):
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    imarray = np.array(im)
    # I = plt.imread(path)
    return im

class ViT_Anndata(torch.utils.data.Dataset):
    """Some Information about ViT_SKIN"""
    def __init__(self, adata_dict, train_set, gene_list, train=True,r=4,norm=False,flatten=True,ori=True,adj=True,prune='NA',neighs=4):
        super(ViT_Anndata, self).__init__()

        self.r = 224//r

        names = list(adata_dict.keys())

        self.ori = ori
        self.adj = adj
        self.norm = norm
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        tr_names = train_set
        te_names = list(set(samples)-set(tr_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print("Eval set: ", te_names)
        
        self.adata_dict = {k: v for k, v in adata_dict.items() if k in self.names}
    
        print('Loading imgs...')
#        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        path_img_dict = {}
        self.img_dict = {}
        for name in self.names:
            name_orig = list(self.adata_dict[name].uns['spatial'])[0]
            path = self.adata_dict[name].uns["spatial"][name_orig]["metadata"]["source_image_path"]

            if path in path_img_dict:
                self.img_dict[name] =  path_img_dict[path]
            else:
                path_img_dict[path] = torch.Tensor(np.array(read_tiff(path)))
                self.img_dict[name] = path_img_dict[path]

            # self.img_dict[name] = torch.Tensor(np.array(self.img_dict[name]))
            self.img_dict[name] = self.img_dict[name]

        del path_img_dict


        self.gene_set = list(gene_list)
        if self.norm:
            self.exp_dict = {
                i:sc.pp.scale(scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values))).astype(np.float64)
                for i,m in self.adata_dict.items()
            }
        else:
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64) 
                for i,m in self.adata_dict.items()
            }
        if self.ori:
            self.ori_dict = {i:m.to_df()[self.gene_set].values.astype(np.float64) for i,m in self.adata_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf.astype(np.float64)
        self.center_dict = {
            i:np.floor(m.obsm["spatial"].astype(np.int64)).astype(int)
            for i,m in self.adata_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_col', 'array_row']].values for i,m in self.adata_dict.items()}
        self.patch_dict=dfd(lambda :None)
        self.lengths = [i.n_obs for i in self.adata_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))


    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID].permute(1,0,2)

        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))

        # return data
        if self.train:
            return patches, positions, exps
        else:
            return patches, positions, exps, torch.Tensor(centers)        
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        name_orig = list(self.adata_dict[name].uns['spatial'])[0]
        path = self.adata_dict[name].uns["spatial"][name_orig]["metadata"]["source_image_path"]
        im = read_tiff(path)
        return im


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
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,mode="train",fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4):
        super(ViT_HER2ST, self).__init__()
        
        self.cnt_dir = '/QRISdata/Q2051/Yuanhao/Hist2ST/data/her2st/data/ST-cnts'
        self.img_dir = '/QRISdata/Q2051/Yuanhao/Hist2ST/data/her2st/data/ST-imgs'
        self.pos_dir = '/QRISdata/Q2051/Yuanhao/Hist2ST/data/her2st/data/ST-spotfiles'
        self.lbl_dir = '/QRISdata/Q2051/Yuanhao/Hist2ST/data/her2st/data/ST-pat/lbl'
        self.r = 224//r

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('/QRISdata/Q2051/Yuanhao/Hist2ST/data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.mode = mode
        self.ori = ori
        self.adj = adj
       # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names[0:33]

        te_names = [samples[fold]]
        print("test sample:",te_names)
        tr_names = list(set(samples)-set(te_names))
        
        val_names = tr_names[:3]
        print("validation samples:",val_names)
        tr_names = list(set(tr_names)-set(val_names))
        

        if self.mode=="train":
            self.names = tr_names
        elif self.mode=="test":
            self.names = te_names
        elif self.mode=="val":
            self.names = val_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':6
        }
        
        for i in self.names:
            if i in ['A1','B1','C1','D1','E1','F1','G2','H1']:
                idx=self.meta_dict[i].index
                lbl=self.get_lbl(i)
                lbl=lbl.loc[idx,:]['label'].values
                lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                self.label[i]=lbl
            else:
                idx=self.meta_dict[i].index
                # Encode the unknow cell type as 6
                self.label[i]=torch.full((len(idx),),6)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i,m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
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
        data=[patches, positions, exps, label]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df

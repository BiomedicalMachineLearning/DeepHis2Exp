import time
from pathlib import Path
import sys
# sys.path.append("../models/HisToGene")

sys.path.append("./")

from window_adata import *
from read_stimage_genes import *
from data_vit import ViT_Anndata
from read_stimage_genes import read_gene_set_hvg,intersect_section_genes


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import *
from dataset import ViT_HER2ST, ViT_SKIN

import pickle

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from scanpy import read_visium, read_10x_mtx

BCELL = ['CD19', 'CD79A', 'CD79B', 'MS4A1']
TUMOR = ['FASN']
CD4T = ['CD4']
CD8T = ['CD8A', 'CD8B']
DC = ['CLIC2', 'CLEC10A', 'CD1B', 'CD1A', 'CD1E']
MDC = ['LAMP3']
CMM = ['BRAF', 'KRAS']
IG = {'B_cell':BCELL, 'Tumor':TUMOR, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T, 'Dendritic_cells':DC, 
        'Mature_dendritic_cells':MDC, 'Cutaneous_Malignant_Melanoma':CMM}
MARKERS = []
for i in IG.values():
    MARKERS+=i
LYM = {'B_cell':BCELL, 'CD4+T_cell':CD4T, 'CD8+T_cell':CD8T}

class STDataset(torch.utils.data.Dataset):
    """Some Information about STDataset"""
    def __init__(self, adata, img_path, diameter=177.5, train=True):
        super(STDataset, self).__init__()

        self.exp = adata.X.toarray()
        self.im = read_tiff(img_path)
        self.r = np.ceil(diameter/2).astype(int)
        self.train = train
        # self.d_spot = self.d_spot if self.d_spot%2==0 else self.d_spot+1
        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
        self.centers = adata.obsm['spatial']
        self.pos = adata.obsm['position_norm']
    def __getitem__(self, index):
        exp = self.exp[index]
        center = self.centers[index]
        x, y = center
        patch = self.im.crop((x-self.r, y-self.r, x+self.r, y+self.r))
        exp = torch.Tensor(exp)
        mask = exp!=0
        mask = mask.float()
        if self.train:
            patch = self.transforms(patch)
        pos = torch.Tensor(self.pos[index])
        return patch, pos, exp, mask

    def __len__(self):
        return len(self.centers)



class HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,fold=0):
        super(HER2ST, self).__init__()
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//2
        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
  
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names
        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))
        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names
        print('Loading imgs...')
        self.img_dict = {i:self.get_img(i) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}

        # self.gene_set = self.get_overlap(self.meta_dict,gene_list)
        # print(len(self.gene_set))
        # np.save('data/her_hvg',self.gene_set)
        # quit()
        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        i = 0
        while index>=self.cumlen[i]:
            i += 1
        idx = index
        if i > 0:
            idx = index - self.cumlen[i-1]
        
        exp = self.exp_dict[self.id2name[i]][idx]
        center = self.center_dict[self.id2name[i]][idx]
        loc = self.loc_dict[self.id2name[i]][idx]

        # if self.cls or self.train==False:

        exp = torch.Tensor(exp)
        loc = torch.Tensor(loc)

        x, y = center
        patch = self.img_dict[self.id2name[i]].crop((x-self.r, y-self.r, x+self.r, y+self.r))
        if self.train:
            patch = self.transforms(patch)
        else:
            patch = transforms.ToTensor()(patch)

        if self.train:
            return patch, loc, exp
        else: 
            return patch, loc, exp, torch.Tensor(center)

    def __len__(self):
        return self.cumlen[-1]

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

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))
        self.max_x = 0
        self.max_y = 0
        loc = meta[['x','y']].values
        self.max_x = max(self.max_x, loc[:,0].max())
        self.max_y = max(self.max_y, loc[:,1].max())
        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(ViT_HER2ST, self).__init__()

        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//4

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        # gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
        #    "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
        gene_list_path = "../../scripts/gene_list.pkl"
        with open(gene_list_path, 'rb') as f:
            gene_list = pickle.load(f)
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.sr = sr
        
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names

        te_names = [samples[fold]]
        tr_names = list(set(samples)-set(te_names))

        if train:
            # names = names[1:33]
            # names = names[1:33] if self.cls==False else ['A1','B1','C1','D1','E1','F1','G2']
            names = tr_names
        else:
            # names = [names[33]]
            # names = ['A1']
            # names = [ds] if ds else ['H1']
            names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in names}


        self.gene_set = list(gene_list)
        self.exp_dict = {i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) for i,m in self.meta_dict.items()}
        self.center_dict = {i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) for i,m in self.meta_dict.items()}
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}


        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[self.id2name[i]]
        centers = self.center_dict[self.id2name[i]]
        loc = self.loc_dict[self.id2name[i]]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, positions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            if self.train:
                return patches, positions, exps
            else: 
                return patches, positions, exps, torch.Tensor(centers)
        
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

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)


import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor, self).__init__()
        backbone = torchvision.models.resnet101(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        return x

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, backbone='resnet50', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        backbone = torchvision.models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)
        # self.valid_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.feature_extractor(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        return parser


class STModel(pl.LightningModule):
    def __init__(self, feature_model=None, n_genes=1000, hidden_dim=2048, learning_rate=1e-5, use_mask=False, use_pos=False, cls=False):
        super().__init__()
        self.save_hyperparameters()
        # self.feature_model = None
        if feature_model:
            # self.feature_model = ImageClassifier.load_from_checkpoint(feature_model)
            # self.feature_model.freeze()
            self.feature_extractor = ImageClassifier.load_from_checkpoint(feature_model)
        else:
            self.feature_extractor = FeatureExtractor()
        # self.pos_embed = nn.Linear(2, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, center):
        feature = self.feature_extractor(patch).flatten(1)
        h = feature
        pred = self.pred_head(F.relu(h))
        return pred

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('valid_loss', loss)
        
    def test_step(self, batch, batch_idx):
        patch, center, exp, mask, label = batch
        if self.use_mask:
            pred, mask_pred = self(patch, center)
        else:
            pred = self(patch, center)

        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


class HisToGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):        
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



from scipy import stats

def plot_correlation(df, attr_1, attr_2):
    r = stats.pearsonr(df[attr_1], 
                       df[attr_2])[0] **2

    g = sns.lmplot(data=df,
        x=attr_1, y=attr_2,
        height=5, legend=True
    )
    # g.set(ylim=(0, 360), xlim=(0,360))

    g.set_axis_labels(attr_1, attr_2)
    plt.annotate(r'$R^2:{0:.2f}$'.format(r),
                (max(df[attr_1])*0.9, max(df[attr_2])*0.9))
    return g


def calculate_correlation(attr_1, attr_2):
    r = stats.pearsonr(attr_1, 
                       attr_2)[0]
    return r

def calculate_correlation_2(attr_1, attr_2):
    r = stats.spearmanr(attr_1, 
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
with open('../../data/histogene_hvg1000.pickle', 'wb') as f:
    pickle.dump(gene_list, f)






# gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
#           "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
# gene_list_path = "../../scripts/gene_list.pkl"
# with open(gene_list_path, 'rb') as f:
#     gene_list = pickle.load(f)
# n_genes = len(gene_list)

# cnt_dir = 'data/her2st/data/ST-cnts'
# names = os.listdir(cnt_dir)
# names.sort()
# names = [i[:2] for i in names]

df = pd.DataFrame()
i = int(sys.argv[1])
fold = i
test_sample = samps1[fold]
test_sample_orig = samps1[fold]

# test_sample = names[fold]
# print("process {}".format(test_sample))
tag = 'pf'+test_sample
train_set = list(set(list(adata_dict.keys())) - set([i for i in list(adata_dict.keys()) if test_sample in i]))


trainset = ViT_Anndata(adata_dict = adata_dict, train_set = train_set, gene_list = gene_list,
            train=True,flatten=True,adj=True,ori=True,prune='NA',neighs=4, 
        )

train_loader = DataLoader(trainset, batch_size=1, num_workers=4, shuffle=True)
model = HisToGene(n_layers=8, n_genes=n_genes, learning_rate=1e-5, n_pos=128)
trainer = pl.Trainer(gpus=1, max_epochs=100)

start_train = time.perf_counter()

trainer.fit(model, train_loader)
trainer.save_checkpoint("../../trained_models/last_train_"+tag+'_'+str(fold)+".ckpt")

#     model = HisToGene.load_from_checkpoint("model/last_train_"+tag+'_'+str(fold)+".ckpt",n_layers=8, n_genes=n_genes, learning_rate=1e-5)

end_train = time.perf_counter()

device = torch.device('cuda')

test_sample = [i for i in list(adata_dict.keys()) if test_sample in i]
test_set = list(set(list(adata_dict.keys())) - set(test_sample))    
testset = ViT_Anndata(adata_dict = adata_dict, train_set = test_set, gene_list = gene_list,
            train=False,flatten=True,adj=True,ori=True,prune='NA',neighs=4, 
        )
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

from tqdm import tqdm

def model_predict(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    adatas,adata_gts = [],[]
    with torch.no_grad():
        for patch, position, exp, center in tqdm(test_loader):

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch, position)


            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp
            else:
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)
                
            preds = preds.cpu().squeeze().numpy()
            ct = ct.cpu().squeeze().numpy()
            gt = gt.cpu().squeeze().numpy()
            adata = ann.AnnData(preds)
            adata.obsm['spatial'] = ct

            adata_gt = ann.AnnData(gt)
            adata_gt.obsm['spatial'] = ct

            adatas.append(adata)
            adata_gts.append(adata_gt)

    adata = ad.concat(adatas)
    adata_gt = ad.concat(adata_gts)

    return adata, adata_gt


adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device = device)

adata_pred.var_names = gene_list
adata_truth.var_names = gene_list

pred_adata = adata_pred.copy()
test_dataset = adata_truth.copy()

# test_sample = ','.join(list(test_sample))

with open(f"../../results/pf_cv/histogene_preds_{test_sample_orig}_{i}.pkl", 'wb') as f:
    pickle.dump([pred_adata,test_dataset], f)

for gene in pred_adata.var_names:
    pred = pred_adata.to_df().loc[:,gene]
    pred = pred.fillna(0)
    cor_val = calculate_correlation_2(pred, test_dataset.to_df().loc[:,gene])
    cor_pearson = calculate_correlation(pred, test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val,cor_pearson, test_sample_orig, "HisToGene"], 
                         index=["Gene", "Spearman correlation", "Pearson correlation","Slide", "Method"]),
              ignore_index=True)

del model
torch.cuda.empty_cache()

df.to_csv("../../results/pf_cv/histogene_cor_{}_{i}.csv".format(test_sample_orig, i))

with open("../../results/pf_cv/histogene_times.txt", 'a') as f:
    f.write(f"{i} {test_sample_orig} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")



# evall([i for i in list(adata_dict.keys()) if 'VLP78_A' in i], gene_list)





# dataset = ViT_HER2ST(train=False,sr=False,fold=fold)
# test_loader = DataLoader(dataset, batch_size=1, num_workers=4)

# start_pred = time.perf_counter()
# adata_pred, adata_truth = model_predict(model, test_loader, attention=False, device = device)

# end_pred = time.perf_counter()
# adata_pred.var_names = gene_list
# adata_truth.var_names = gene_list

# pred_adata = adata_pred.copy()
# test_dataset = adata_truth.copy()

# for gene in pred_adata.var_names:
#     cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
#     df = df.append(pd.Series([gene, cor_val, test_sample, "His2gene"], 
#                          index=["Gene", "Pearson correlation", "Slide", "Method"]),
#               ignore_index=True)
# del model
# torch.cuda.empty_cache()

# df.to_csv("../../results/ffpe/histogene_cor_{}.csv".format(test_sample))

# with open("../../results/ffpe/histogene_times.txt", 'a') as f:
#     f.write(f"{test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")

# with open(f"../../results/ffpe/histogene_preds_{test_sample}.pkl", 'wb') as f:
#     pickle.dump([pred_adata,test_dataset], f)


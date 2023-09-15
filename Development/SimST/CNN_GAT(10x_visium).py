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
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,)

""" Integrate two visium datasets """
data_dir1 = "./Alex_NatGen_6BreastCancer/"
data_dir2 = "./breast_cancer_10x_visium/"

samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
samps2 = ["block1", "block2", "FFPE", "1168993F"]

sampsall = samps1 + samps2
samples1 = {i:data_dir1 + i for i in samps1}
samples2 = {i:data_dir2 + i for i in samps2}

# Marker gene list
gene_list = ["COX6C","TTLL12", "HSP90AB1", "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

# # Load windowed dataset
import pickle
with open('10x_visium_dataset_without_window.pickle', 'rb') as f:
    adata_dict0 = pickle.load(f)
    
# Define the gridding size
sizes = [3000 for i in range(len(adata_dict0))]

# Split tiles into smaller patches according to gridding size
adata_dict = window_adata(adata_dict0, sizes)
del adata_dict0

# For training
from data_vit import ViT_Anndata

def dataset_wrap(fold = 0, dataloader= True):
    test_sample = sampsall[fold] # Split one sample as test sample
    test_sample_orig = sampsall[fold] 
    val_sample = list(set(sampsall)-set(sampsall[fold]))[:3] # Split 3 samples as validation samples
    train_sample = list(set(sampsall)-set(test_sample)-set(val_sample)) # Other samples are training samples

    tr_name = list(set([i for i in list(adata_dict.keys()) for tr in train_sample if tr in i]))
    val_name = list(set([i for i in list(adata_dict.keys()) for val in val_sample if val in i]))
    te_name = list(set([i for i in list(adata_dict.keys()) if test_sample in i]))

    trainset = ViT_Anndata(adata_dict = adata_dict, train_set = tr_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    valset = ViT_Anndata(adata_dict = adata_dict, train_set = val_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
#     testset = ViT_Anndata(adata_dict = adata_dict, train_set = te_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )

    print("LOADED TRAINSET")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1, num_workers=0, shuffle=True)
#     test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    if dataloader==True:
        return train_loader, val_loader
#         return train_loader, val_loader, test_loader
    else:
        return trainset, valset, testset
    
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
import torch.nn as nn

from gcn import *
from transformer import *
from NB_module import *
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from torchvision import models, transforms

# set random seed
def setup_seed(seed=12000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
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
    
def ft_extra(name="resnet"):
    # Fine tune the pretrained model
    if name=="resnet":
        model_ft = torchvision.models.resnet50(weights = models.ResNet50_Weights)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.fc = nn.Sequential(nn.Identity())
        dim=2048
    elif name=="efficient":
        model_ft = torchvision.models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.classifier = nn.Sequential(nn.Identity())
        dim=1280
    elif name=='swin':
        model_ft = torchvision.models.swin_s(weights = models.Swin_S_Weights)
        for param in model_ft.parameters():
            param.requires_grad = False
        model_ft.head = nn.Sequential(nn.Identity())
        dim=768
    return model_ft, dim

class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, num_heads=8, concat_heads=True, alpha=0.2):
        """
        Args:
            c_in: Dimensionality of input features
            c_out: Dimensionality of output features
            num_heads: Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads: If True, the output of the different heads is concatenated instead of averaged.
            alpha: Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Args:
            node_feats: Input features of the node. Shape: [batch_size, c_in]
            adj_matrix: Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs: If True, the attention weights are printed during the forward pass
                               (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat(
            [
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )  # Index select returns a tensor with node_feats_flat being indexed at the desired positions

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        atten = attn_probs.permute(0, 3, 1, 2)
        node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)
        return node_feats, atten

class Feature_extractor(nn.Module):
    def __init__(self, name="resnet", dim=1024,
                 num_layer=4,
                ):
        super().__init__()
        self.ft_extractor,ft_dim = ft_extra(name=name)
        self.projection = nn.Sequential(nn.Linear(ft_dim, dim))
        self.GATLayer=GATLayer(c_in=dim, c_out=dim)
        self.GAT=nn.ModuleList([self.GATLayer for _ in range(num_layer)])
        self.jknet=nn.Sequential(
            nn.LSTM(dim,dim,2),
            SelectItem(0),
        )
        self.dropout = nn.Dropout(0.2)
        self.tf = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),])

    def forward(self,patch,ct,adj):
        # Resize the tiles
        x = self.tf(patch.squeeze())
        # Extract the embedding
        x = self.ft_extractor(x)
        x = self.projection(x)
        x = self.dropout(x).unsqueeze(0)
        # GAT with layer-aggregation
        jk=[]
        for layer in self.GAT:
            x, attn=layer(x,adj)
            jk.append(x)
        x=torch.cat(jk,0)
        # Jumping knowledge-LSTM
        x=self.jknet(x).mean(0)
        return x, attn

class CNN_GAT(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, name="resnet", dim=1024, n_pos=128, n_genes=12, 
                 num_layer=4, 
                 zinb=0.25, nb=False, policy='mean', bake=5, lamb=0.5, 
                ):
        super().__init__()
#          self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.nb=nb
        self.zinb=zinb
        self.bake=bake
        self.lamb=lamb
        
        """ Position Embedding """
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        
        """ Feature Extractor """
        self.vit = Feature_extractor(name=name, num_layer=num_layer)
        
        self.n_genes=n_genes
        """ ZINB Loss """
        if self.zinb>0:
            if self.nb:
                self.hr=nn.Linear(dim, n_genes)
                self.hp=nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())

        """Data augmentation"""
        self.coef=nn.Sequential(
                nn.Linear(dim,dim),
                nn.ReLU(),
                nn.Linear(dim,1),
            )
        self.imaug=transforms.Compose([
            transforms.RandomGrayscale(0.1),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(0.2),
        ])
        """ Regression Module """
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )
        
    def forward(self, patch, centers, adj):
        
        """ Spatial location for Transformer """
        centers_x = self.x_embed(centers[:,:,0].long())
        centers_y = self.y_embed(centers[:,:,1].long())
        ct=centers_x + centers_y
        
        """ Feature Extraction """
        h, attn = self.vit(patch,ct,adj)
        
        """ Gene expression prediction """
        x = self.gene_head(h)
        
        """ZINB Distribution"""
        extra=None
        if self.zinb>0:
            if self.nb:
                r=self.hr(h)
                p=self.hp(h)
                extra=(r,p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra=(m,d,p) 
        
        h=self.coef(h)
        return x,extra,h,attn
    
    def aug(self,patch,center,adj):
        bake_x=[]
        # generate 5 additional image patches
        for i in range(self.bake):
            new_patch=self.imaug(patch.squeeze(0)).unsqueeze(0)
            x,_,h,_=self(new_patch,center,adj)
            bake_x.append((x.unsqueeze(0),h.unsqueeze(0)))
        return bake_x
    
    def distillation(self,bake_x):
        new_x,coef=zip(*bake_x)
        coef=torch.cat(coef,0)
        new_x=torch.cat(new_x,0)
        coef=F.softmax(coef,dim=0)
        new_x=(new_x*coef).sum(0)
        return new_x
    

    def training_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
#         adj=adj.squeeze(0)
        exp=exp.squeeze(0)

        """ Model inference """
        pred,extra,h,attn = self(patch, center, adj)

        """ Regression Loss """
        mse_loss = F.mse_loss(pred, exp)
        self.log('mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)

        """ ZINB Loss """
        zinb_loss=0
        if self.zinb>0:
            if self.nb:
                r,p=extra
                zinb_loss = NB_loss(oris.squeeze(0),r,p)
            else:
                m,d,p=extra
                zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
        self.log('zinb_loss', zinb_loss,on_epoch=True, prog_bar=True, logger=True)
        
        """ Self-distillation loss """
        bake_loss=0
        bake_x=self.aug(patch,center,adj)
        new_pred=self.distillation(bake_x)
        bake_loss+=F.mse_loss(new_pred,pred)
        self.log('bake_loss', bake_loss,on_epoch=True, prog_bar=True, logger=True)

        """ Total Loss """
        loss = mse_loss + self.zinb*zinb_loss+self.lamb*bake_loss
        self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
#         adj=adj.squeeze(0)
        exp=exp.squeeze(0)

        """ Model Inference """
        pred,extra,h,_ = self(patch, center, adj)

        """ Regression Loss """
        mse_loss = F.mse_loss(pred, exp)
        self.log('val_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)

        return mse_loss

    
    def test_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs, *_ = batch
#         adj=adj.squeeze(0)
        exp=exp.squeeze(0)

        """ Model Inference """
        gene_exp,extra,h,attn = self(patch, center, adj)

        """Pearson correlation coeficient"""
        adata1 = ad.AnnData(gene_exp.cpu().detach().numpy())
        adata2 = ad.AnnData(exp.cpu().detach().numpy())
        R=get_R(adata1,adata2)[0]
        mean_pcc=np.nanmean(R)
        self.log('test_mean_PCC', mean_pcc, on_epoch=True, prog_bar=True, logger=True)

        

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optim_dict = {'optimizer': optim}
        return optim_dict

"""For training only"""
import gc
from data_vit import ViT_Anndata
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

start_time = time.time()
"""Training loops"""
seed=12000
epochs=100
dim=1024
name = "resnet"   # ["swim_s", "efficientnet", "resnet"]
for fold in range(10):
    """Load dataset"""
    train_loader, val_loader = dataset_wrap(fold = fold, dataloader= True)
    del adata_dict0
    """Define model"""
    model = CNN_GAT(name=name, dim=dim)
    setup_seed(seed)

    """Setup trainer"""
    logger = pl.loggers.CSVLogger("logs", name=f"./{name}_fold{fold}")
    trainer = pl.Trainer(accelerator='auto',  callbacks=[EarlyStopping(monitor='train_loss',mode='min')], max_epochs=epochs,logger=logger)
    trainer.fit(model, train_loader, val_loader)

    """Save model and clean memory"""
    torch.save(model.state_dict(),f"./model/{name}-seed{seed}-epochs{epochs}-sampleIndex{fold}.ckpt")
    gc.collect()
    del train_loader, val_loader, test_loader, model
end_time = time.time()
execution_time = end_time - start_time
print("Training time: ", execution_time/3600, " hours")

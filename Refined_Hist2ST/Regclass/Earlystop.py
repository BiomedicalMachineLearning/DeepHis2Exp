import gc
import os
import time
import random
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as tf
import warnings
warnings.filterwarnings("ignore")


from torch.nn import init
from easydl import *
from anndata import AnnData
from torch import nn, einsum
from scipy.stats import pearsonr
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd.variable import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from collections import defaultdict
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


from HIST2ST import *
from tqdm import tqdm
from predict import *
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    
def MLP(n_layers, n_units):
    """
    num_layers=[2,3,4]
    num_units=[200,400,600,800,1000]
    Dropout_rate=0.2
    """
    layers = []
    in_size =1024
    for i in range(n_layers):
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.2))
        in_size = n_units
    layers.append(nn.Linear(in_size, 7))
    return nn.Sequential(*layers)
    
    
def Stage1(fold=0, freeze=False):        
    model=Hist2ST(
    depth1=2, depth2=8,depth3=4,n_genes=785, 
    kernel_size=5, patch_size=7,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5,)
    model.load_state_dict(torch.load(f'/scratch/imb/uqxjin3/Yuanhao/Hist2ST/model/Basic_Hist2ST/{fold}-Hist2ST.ckpt'))
    def set_parameter_requires_grad(model, freeze):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
    set_parameter_requires_grad(model,freeze=freeze)
    return model


class Stage2(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, fig_size=112,label=None,
                 dropout=0.2, n_pos=64, kernel_size=5, patch_size=7, 
                 fold=0,freeze=True, seed=12000, unbiased=False,
                 n_layer=2, n_units=200, 
                ):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.seed=seed
        self.unbiased=unbiased
        self.label=label
        self.Hist2ST=Stage1(fold,freeze)
        self.classifier = MLP(n_layer,n_units)

    def forward(self, patches, centers, adj):
        x,extra,h = self.Hist2ST(patches, centers, adj)
        cell_type_prob = self.classifier(h)
        return cell_type_prob

    def training_step(self, batch, batch_idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        setup_seed(self.seed)
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        cell_type_prob = self(patch, center, adj)
        """
        Classification loss
        """
        if torch.all(label.eq(torch.full((label.shape), 6).to(device))):
            cross_entropy = 0.0
            cross_entropy = torch.tensor(cross_entropy,requires_grad=True).to(device)
        else:
            cell_type = cell_type_prob.argmax(dim=1, keepdim=True)
            cell_type, label = cell_type.view(-1), label.view(-1)
            if self.unbiased:
                cls_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.98,35.89,13.81,25.22,5.64,11.8,0]).to(device))
            else:
                cls_loss = nn.CrossEntropyLoss()
            cross_entropy = cls_loss(cell_type_prob,label.long())
            accuracy = cell_type.eq(label.view_as(cell_type)).float().mean()
            self.log('accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
#             print(f"CE:{cross_entropy}")
        self.log('cross_entropy', cross_entropy, on_epoch=True, prog_bar=True, logger=True)

        loss = cross_entropy
        self.log('Loss', loss, on_epoch=True, prog_bar=True, logger=True) 

        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict

def Pretrain_Stage2(fold=0, freeze=False, seed=12000, epochs=200):
    n_layer=4
    n_units=800
    model=Stage2(fold=fold, freeze=freeze,n_layer=4, n_units=800)
    model.load_state_dict(torch.load(f"./model/Exp0_ft/Stage2-seed{seed}-epochs{epochs}-n_layer{n_layer}-n_units{n_units}-sampleIndex{fold}.ckpt"))
    def set_parameter_requires_grad(model, freeze):
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
    set_parameter_requires_grad(model,freeze=freeze)
    return model

class Stage3(pl.LightningModule):
    def __init__(self, fold=0, zinb=0.25, nb=False, bake=5, lamb=0.5, gamma=0,
                 learning_rate=1e-5, seed=12000, unbiased=False, label=None, 
                ):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.zinb=zinb
        self.nb=nb
        self.bake=bake
        self.lamb=lamb
        self.gamma=gamma

        
        self.seed=seed
        self.unbiased=unbiased
        self.label=label
        self.Stage2 = Pretrain_Stage2(fold=fold, freeze=False, seed=12000, epochs=200)

    def forward(self, patches, centers, adj):
        gene_exp,extra,ft = self.Stage2.Hist2ST(patches, centers, adj)
        cell_type_prob = self.Stage2.classifier(ft)
        return gene_exp, extra, ft, cell_type_prob
    
    """Helper function"""
    def aug(self,patch,center,adj):
        bake_x=[]
        for i in range(self.bake):
            new_patch=self.Stage2.Hist2ST.tf(patch.squeeze(0)).unsqueeze(0)
            x,_,h=self.Stage2.Hist2ST(new_patch,center,adj,True)
            bake_x.append((x.unsqueeze(0),h.unsqueeze(0)))
        return bake_x

    def distillation(self,bake_x):
        new_x,coef=zip(*bake_x)
        coef=torch.cat(coef,0)
        new_x=torch.cat(new_x,0)
        coef=F.softmax(coef,dim=0)
        new_x=(new_x*coef).sum(0)
        return new_x
    
    """Training steps"""
    def training_step(self, batch, batch_idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        setup_seed(self.seed)
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        
        """Model inference"""
        gene_exp, extra, ft, cell_type_prob = self(patch, center, adj)
        
        """Regression loss"""
        mse_loss = F.mse_loss(gene_exp, exp)
        self.log('mse_loss', mse_loss, on_epoch=True, prog_bar=True, logger=True)
        
        """Classification loss"""
        if torch.all(label.eq(torch.full((label.shape), 6).to(device))):
            cross_entropy = 0.0
            cross_entropy = torch.tensor(cross_entropy,requires_grad=True).to(device)
        else:
            cell_type = cell_type_prob.argmax(dim=1, keepdim=True)
            cell_type, label = cell_type.view(-1), label.view(-1)
            if self.unbiased:
                cls_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.98,35.89,13.81,25.22,5.64,11.8,0]).to(device))
            else:
                cls_loss = nn.CrossEntropyLoss()
            cross_entropy = cls_loss(cell_type_prob,label.long())
            accuracy = cell_type.eq(label.view_as(cell_type)).float().mean()
            self.log('accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('cross_entropy', cross_entropy, on_epoch=True, prog_bar=True, logger=True)
        
        """Data augmentation loss"""
        bake_loss=0
        if self.bake>0:
            bake_x=self.aug(patch,center,adj)
            new_pred=self.distillation(bake_x)
            bake_loss+=F.mse_loss(new_pred,gene_exp)
            self.log('bake_loss', bake_loss, on_epoch=True, prog_bar=True, logger=True)
        
        """Distribution loss"""
        zinb_loss=0
        if self.zinb>0:
            if self.nb:
                r,p=extra
                zinb_loss = NB_loss(oris.squeeze(0),r,p)
            else:
                m,d,p=extra
                zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
            self.log('zinb_loss', zinb_loss,on_epoch=True, prog_bar=True, logger=True)

        """Total loss"""
        loss = mse_loss + self.zinb*zinb_loss + self.lamb*bake_loss + self.gamma*cross_entropy
        self.log('Loss', loss, on_epoch=True, prog_bar=True, logger=True) 

        return loss
    
    def validation_step(self, batch, batch_idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        setup_seed(self.seed)
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        
        """Model inference"""
        gene_exp, extra, ft, cell_type_prob = self(patch, center, adj)
        
        """Regression loss"""
        mse_loss = F.mse_loss(gene_exp, exp)
        self.log('val_loss', mse_loss, on_epoch=True, prog_bar=True, logger=True)
        
        """Pearson correlation coeficient"""
        adata1 = ad.AnnData(gene_exp.cpu().detach().numpy())
        adata2 = ad.AnnData(exp.cpu().detach().numpy())
        R=get_R(adata1,adata2)[0]
        mean_pcc=np.nanmean(R)
        self.log('mean_PCC', mean_pcc, on_epoch=True, prog_bar=True, logger=True)
        
        """Classification loss"""
        if torch.all(label.eq(torch.full((label.shape), 6).to(device))):
            cross_entropy = 0.0
            cross_entropy = torch.tensor(cross_entropy,requires_grad=True).to(device)
        else:
            cell_type = cell_type_prob.argmax(dim=1, keepdim=True)
            cell_type, label = cell_type.view(-1), label.view(-1)
            if self.unbiased:
                cls_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.98,35.89,13.81,25.22,5.64,11.8,0]).to(device))
            else:
                cls_loss = nn.CrossEntropyLoss()
            cross_entropy = cls_loss(cell_type_prob,label.long())
            accuracy = cell_type.eq(label.view_as(cell_type)).float().mean()
            self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_cross_entropy', cross_entropy, on_epoch=True, prog_bar=True, logger=True)
    def test_step(self, batch, batch_idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        setup_seed(self.seed)
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        
        """Model inference"""
        gene_exp, extra, ft, cell_type_prob = self(patch, center, adj)
        
        """Regression loss"""
        mse_loss = F.mse_loss(gene_exp, exp)
        self.log('test_loss', mse_loss, on_epoch=True, prog_bar=True, logger=True)
        
        """Pearson correlation coeficient"""
        adata1 = ad.AnnData(gene_exp.cpu().detach().numpy())
        adata2 = ad.AnnData(exp.cpu().detach().numpy())
        R=get_R(adata1,adata2)[0]
        mean_pcc=np.nanmean(R)
        self.log('test_mean_PCC', mean_pcc, on_epoch=True, prog_bar=True, logger=True)
        
        """Classification loss"""
        if torch.all(label.eq(torch.full((label.shape), 6).to(device))):
            cross_entropy = 0.0
            cross_entropy = torch.tensor(cross_entropy,requires_grad=True).to(device)
        else:
            cell_type = cell_type_prob.argmax(dim=1, keepdim=True)
            cell_type, label = cell_type.view(-1), label.view(-1)
            accuracy = cell_type.eq(label.view_as(cell_type)).float().mean()
            self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
    
"""Parameters Setting"""
seeds=[12000]
gammas=[0, 0.1, 1, 10]
epochs=100
unbiased=True

# folds=[0, 6, 12, 18, 24, 27, 31]
for seed in seeds:
    for gamma in gammas:
        """Training loops"""
        for fold in range(29,33):
            """Load dataset"""
            seed_everything(seed)
            trainset = pk_load(mode='train',fold=fold, )
            validset = pk_load(mode='val',fold=fold, )
            testset = pk_load(mode='test',fold=fold, )
            
            train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
            val_loader = DataLoader(validset, batch_size=1, num_workers=0, shuffle=False)
            test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

            """Define model"""
            model = model=Stage3(fold=fold, gamma=gamma)
            setup_seed(seed)

            """Setup trainer"""
            print("gamma:",gamma)
            logger = pl.loggers.CSVLogger("logs", name=f"Earlystop_gamma{gamma}_{fold}")
            trainer = pl.Trainer(accelerator='auto',callbacks=[EarlyStopping(monitor='val_loss',mode='min')], max_epochs=epochs,logger=logger)
            trainer.fit(model, train_loader, val_loader)
            trainer.validate(model, test_loader)
            
            """Save model and clean memory"""
            torch.save(model.state_dict(),f"./model/Earlystop/Stage3-unbiased-seed{seed}-epochs{epochs}-gamma{gamma}-sampleIndex{fold}.ckpt")
            gc.collect()
       

import warnings
warnings.filterwarnings("ignore")
import torch
import random
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
import torch.nn as nn

from gcn import *
from NB_module import *
from transformer import *
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

# set random seed
def setup_seed(seed=12000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

class ViT(nn.Module):
    def __init__(self, dim=768,
                 depth1=2, depth2=8, depth3=4, 
                 heads=8, dim_head=64, mlp_dim=1024,
                 policy='mean',gcn=True
                ):
        super().__init__()
        
        self.transformer=nn.Sequential(*[attn_block(dim,heads,dim_head,mlp_dim,0.2) for i in range(depth2)])
        self.GCN=nn.ModuleList([gs_block(dim,dim,policy,gcn) for i in range(depth3)])
        self.jknet=nn.Sequential(
            nn.LSTM(dim,dim,2),
            SelectItem(0),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self,ft,ct,adj):
        x = self.dropout(ft.squeeze(0)).unsqueeze(0)
        x=self.transformer(x+ct).squeeze(0)
        jk=[]
        for layer in self.GCN:
            x=layer(x,adj.squeeze(0))
            jk.append(x.unsqueeze(0))
        x=torch.cat(jk,0)
        x=self.jknet(x).mean(0)
        return x

class CNN_ST(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, dim=768, n_pos=64, n_genes=785, 
                 depth1=2, depth2=8, depth3=4, heads=16, 
                 zinb=0.25, nb=False, policy='mean', 
                ):
        super().__init__()
#          self.save_hyperparameters()

#         dim=(fig_size//patch_size)**2*channel//8

        self.learning_rate = learning_rate
        self.nb=nb
        self.zinb=zinb
        
        """ Position Embedding """
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        
        """ Feature Extractor """
        self.vit = ViT(
            heads=heads,
            dim=dim, depth1=depth1,depth2=depth2, depth3=depth3, 
            mlp_dim=dim, policy=policy, gcn=True, )
        
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

         
        """ Regression Module """
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )
        
    def forward(self, ft, centers, adj ):
        
        """ Spatial location for Transformer """
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        ct=centers_x + centers_y
        
        """ Feature Extraction """
        h = self.vit(ft,ct,adj)
        
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
            
        return x,extra,h
    

    def training_step(self, batch, batch_idx):
            setup_seed()
            patch, center, exp, label, ft, adj, oris, sfs, *_ = batch
            adj=adj.squeeze(0)
            exp=exp.squeeze(0)
            
            """ Model inference """
            pred,extra,h = self(ft, center, adj)

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

            """ Total Loss """
            loss=mse_loss+self.zinb*zinb_loss
            self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)
            return loss

    def validation_step(self, batch, batch_idx):
            setup_seed()
            patch, center, exp, label, ft, adj, oris, sfs, *_ = batch
            adj=adj.squeeze(0)
            exp=exp.squeeze(0)
            
            """ Model Inference """
            pred,extra,h = self(ft, center, adj)
            
            """ Regression Loss """
            mse_loss = F.mse_loss(pred, exp)
            self.log('val_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
            
            return mse_loss

    
    def test_step(self, batch, batch_idx):
            setup_seed()
            patch, center, exp, label, ft, adj, oris, sfs, *_ = batch
            adj=adj.squeeze(0)
            exp=exp.squeeze(0)
            
            """ Model Inference """
            gene_exp,extra,h = self(ft, center, adj)
            
            """ Regression Loss """
            mse_loss = F.mse_loss(gene_exp, exp)
            self.log('test_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
            
            """Pearson correlation coeficient"""
            adata1 = ad.AnnData(gene_exp.cpu().detach().numpy())
            adata2 = ad.AnnData(exp.cpu().detach().numpy())
            R=get_R(adata1,adata2)[0]
            mean_pcc=np.nanmean(R)
            self.log('test_mean_PCC', mean_pcc, on_epoch=True, prog_bar=True, logger=True)

            return mse_loss

    def configure_optimizers(self):
            # self.hparams available because we called self.save_hyperparameters()
            optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            optim_dict = {'optimizer': optim}
            return optim_dict


import gc
from dataset import *
from predict import *
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

"""Training loops"""
seed=12000
epochs=100

for pretrained_model in ["swim_s", "efficientnet", "resnet"]:
        for fold in range(33):
            
                """ Load dataset """
                seed_everything(seed)
                if pretrained_model== "swim_s":
                    dim=768
                elif pretrained_model== "efficientnet":
                    dim=1280
                elif pretrained_model == "resnet":
                    dim=2048

                trainset = ViT_HER2ST(mode='train',fold=fold, pretrained_model=pretrained_model)
                validset = ViT_HER2ST(mode='val',fold=fold, pretrained_model=pretrained_model)
                testset = ViT_HER2ST(mode='test',fold=fold, pretrained_model=pretrained_model)

                train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
                val_loader = DataLoader(validset, batch_size=1, num_workers=0, shuffle=False)
                test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

                """Define model"""
                model = CNN_ST(dim=dim)
                setup_seed(seed)
                
                """Setup trainer"""
                logger = pl.loggers.CSVLogger("logs", name=f"Exp1_{pretrained_model}_fold{fold}")
                trainer = pl.Trainer(accelerator='auto',  callbacks=[EarlyStopping(monitor='val_loss',mode='min')], max_epochs=epochs,logger=logger)
                trainer.fit(model, train_loader, val_loader)
                trainer.test(model, test_loader)
                
                """Save model and clean memory"""
                torch.save(model.state_dict(),f"./model/Earlystop/{pretrained_model}-seed{seed}-epochs{epochs}-sampleIndex{fold}.ckpt")
               
                gc.collect()
                del trainset, train_loader


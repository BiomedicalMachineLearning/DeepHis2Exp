import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from scipy.stats import pearsonr
from torch.nn import functional as F

from GNN import *
from NB_module import *
from transformer import *


class convmixer_block(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.dw=nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.BatchNorm2d(dim),
                nn.GELU(),
        )
        self.pw=nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
    def forward(self,x):
        x=self.dw(x)+x
        x=self.pw(x)
        return x
class mixer_transformer(nn.Module):
    def __init__(self,channel=32, kernel_size=5, dim=1024,
                 depth1=2, depth2=8, 
                 heads=8, dim_head=64, mlp_dim=1024, dropout = 0.,
                 gnn="GIN", co_conv=False,
                ):
        super().__init__()
        self.co_conv=co_conv
        self.layer1=nn.Sequential(
            *[convmixer_block(channel,kernel_size) for i in range(depth1)],
        )
        self.layer2=nn.Sequential(*[attn_block(dim,heads,dim_head,mlp_dim,dropout) for i in range(depth2)])
        self.layer3=GNN(num_feature=1024, nhid=1024, residual=True, gnn=gnn,
                avgpool=True, maxpool=True, channel=True, 
                lstm=True, dropout=0.2,)
        if co_conv:
            self.layer4=GIN_HSG(num_feature=1024, nhid=1024, residual=True, gnn=gnn,
                avgpool=True, maxpool=True, channel=True, 
                lstm=True, dropout=0.2,)

        self.down=nn.Sequential(
            nn.Conv2d(channel,channel//8,1,1),
            nn.Flatten(),
            nn.Linear(channel//8*dim, dim),
        )
    def forward(self,x,ct,adj):
        x1=self.down(self.layer1(x)).unsqueeze(0)
        x2=self.layer2(x1+ct).squeeze(0)
        x3=self.layer3(x2, adj)
        if self.co_conv:
            x4=self.layer4(x2)
            x_out=(x3+x4)/2
            return x_out
        return x3
class ViT(nn.Module):
    def __init__(self, channel=32,kernel_size=5,dim=1024, 
                 depth1=2, depth2=8, 
                 heads=8, mlp_dim=1024, dim_head = 64, dropout = 0.,
                 gnn="GIN", co_conv=False
                ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transformer = mixer_transformer(
            channel, kernel_size, dim, 
            depth1, depth2, 
            heads, dim_head, mlp_dim, dropout,
            gnn, co_conv
        )

    def forward(self,x,ct,adj):
        x = self.dropout(x)
        x = self.transformer(x,ct,adj)
        return x

class Hist2ST_GNN(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, fig_size=112, label=None, 
                 dropout=0.2, n_pos=256, kernel_size=5, patch_size=7, n_genes=785, 
                 depth1=2, depth2=8, heads=16, channel=32, dim=1024,
                 zinb=0, nb=False, bake=0, lamb=0, policy='mean', gnn="GIN",
                 co_conv=False
                ):
        super().__init__()
        # self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        
        self.nb=nb
        self.zinb=zinb
        
        self.bake=bake
        self.lamb=lamb
        
        self.label=label
        self.patch_embedding = nn.Conv2d(3,channel,patch_size,patch_size)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(
            channel=channel, kernel_size=kernel_size, heads=heads,
            dim=dim, depth1=depth1,depth2=depth2,
            mlp_dim=dim, dropout = dropout, gnn=gnn,
            co_conv=co_conv,
        )
        self.channel=channel
        self.patch_size=patch_size
        self.n_genes=n_genes
        if self.zinb>0:
            if self.nb:
                self.hr=nn.Linear(dim, n_genes)
                self.hp=nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        if self.bake>0:
            self.coef=nn.Sequential(
                nn.Linear(dim,dim),
                nn.ReLU(),
                nn.Linear(dim,1),
            )
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )
        self.tf=tf.Compose([
            tf.RandomGrayscale(0.1),
            tf.RandomRotation(90),
            tf.RandomHorizontalFlip(0.2),
        ])
    def forward(self, patches, centers, adj, aug=False):
        B,N,C,H,W=patches.shape
        patches=patches.reshape(B*N,C,H,W)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        ct=centers_x + centers_y
        h = self.vit(patches,ct,adj)
        x = self.gene_head(h)
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
        if aug:
            h=self.coef(h)
        return x,extra,h
    def aug(self,patch,center,adj):
        bake_x=[]
        for i in range(self.bake):
            new_patch=self.tf(patch.squeeze(0)).unsqueeze(0)
            x,_,h=self(new_patch,center,adj,True)
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
        patch, center, exp, adj, oris, sfs = batch
        adj=adj.squeeze(0)
        exp=exp.squeeze(0)
        pred,extra,h = self(patch, center, adj)
        mse_loss = F.mse_loss(pred, exp)
        self.log('mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
        bake_loss=0
        if self.bake>0:
            bake_x=self.aug(patch,center,adj)
            new_pred=self.distillation(bake_x)
            bake_loss+=F.mse_loss(new_pred,pred)
            self.log('bake_loss', bake_loss,on_epoch=True, prog_bar=True, logger=True)
        zinb_loss=0
        if self.zinb>0:
            if self.nb:
                r,p=extra
                zinb_loss = NB_loss(oris.squeeze(0),r,p)
            else:
                m,d,p=extra
                zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
            self.log('zinb_loss', zinb_loss,on_epoch=True, prog_bar=True, logger=True)
            
        loss=mse_loss+self.zinb*zinb_loss+self.lamb*bake_loss
        self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs = batch 
        adj=adj.squeeze(0)
        pred, *_ = self(patch, center, adj)
        loss = F.mse_loss(pred.squeeze(0), exp.squeeze(0))
        self.log('valid_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        
        pred=pred.squeeze(0).cpu().numpy()
        exp=exp.squeeze(0).cpu().numpy()
        def get_R(data1,data2,dim=1,func=pearsonr):
            return np.array([func(data1[:,g],data2[:,g], alternative='greater')[0] for g in range(data1.shape[dim])])
        r = get_R(pred,exp)
        R = torch.Tensor(r).mean()
        self.log('R', R, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        patch, center, exp, adj, oris, sfs = batch 
        pred, *_ = self(patch, center, adj.squeeze(0))
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return pred, exp
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        StepLR = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        optim_dict = {'optimizer': optim, 'lr_scheduler': StepLR}
        return optim_dict
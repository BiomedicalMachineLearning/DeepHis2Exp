import torch
import random
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import *
from predict import *
from Modules import *
from NB_module import *

# set random seed
def setup_seed(seed=12000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



class Multiview_ST(pl.LightningModule):
    def __init__(self, learning_rate=1e-5, dim=1024, n_genes=785, nb=False, zinb=0.5, 
                ):
        super().__init__()
        """Hyper-parameter setting"""
        self.learning_rate = learning_rate       
        self.n_genes=n_genes
        self.nb=nb
        self.zinb=zinb
        
        """ Encoders """
        self.encoder1 = encoder1() # Resnet50 from VICReg
        self.encoder2 = encoder2() # Swin Transformer
        
        """ Fine tune the last block """
        for param in self.encoder1.model.layer4.parameters():
            param.requires_grad = True
        for param in self.encoder2.model[0][7].parameters():
            param.requires_grad = True
        for param in self.encoder2.model[1].parameters():
            param.requires_grad = True
        for param in self.encoder2.model[2].parameters():
            param.requires_grad = True
        for param in self.encoder2.model[3].parameters():
            param.requires_grad = True
        for param in self.encoder2.model[4].parameters():
            param.requires_grad = True

        
        """ Map the encoder ft to 1024 dim"""
        self.map_reg1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, dim),
        )
        self.map_reg2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, dim),
        )
        
        """ Attention cat """
        self.atten_cat = Attention(in_dim = 1024)
        
        """ GAT """
        self.GAT = GAT(dim = 1024, num_layer = 3)
        
        """ Regression Module"""
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.2),
            nn.Linear(dim, n_genes),
        )
        
#         """ Classification Module"""
#         self.cls = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Linear(dim, 7),
#             nn.Softmax(dim=1)
#         )
        

#         """ ZINB Distribution"""
#         if self.zinb>0:
#             if self.nb:
#                 self.hr=nn.Linear(dim, n_genes)
#                 self.hp=nn.Linear(dim, n_genes)
#             else:
#                 self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
#                 self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
#                 self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        
    def forward(self, patch, adj):
        
        """ Encoding """
        ft1 = self.encoder1(patch)
        ft2 = self.encoder2(patch)
        
        ft3 = self.map_reg1(ft1)
        ft4 = self.map_reg2(ft2)
        
        """ Attention Cat """
        cat_ft, atten1 = self.atten_cat(ft3, ft4)
        
        """ GAT """
        ft, atten2 = self.GAT(cat_ft, adj)
        
        """ Regression (prediction gene expression) """
        pred_exp = self.gene_head(ft)
        
        return pred_exp, ft, atten1, atten2
    
    def training_step(self, batch, batch_idx):
        setup_seed()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        """ Load data """
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        patch, adj, exp = patch.squeeze(0).to(device), adj.squeeze(0).to(device), exp.squeeze(0).to(device)
        
        """ Prediction"""
        pred_exp, *_ = self(patch, adj)
    
        """ Regression Loss"""
        mse_loss = F.mse_loss(pred_exp, exp)
        self.log('mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
        
#         """ ZINB Loss """
#         zinb_loss=0
#         if self.zinb>0:
#             if self.nb:
#                 r,p=extra
#                 zinb_loss = NB_loss(oris.squeeze(0),r,p)
#             else:
#                 m,d,p=extra
#                 zinb_loss = ZINB_loss(oris.squeeze(0),m,d,p,sfs.squeeze(0))
#             self.log('zinb_loss', zinb_loss,on_epoch=True, prog_bar=True, logger=True)


        
#         """ Classification Loss"""
#         if torch.all(label.eq(torch.full((label.shape), 6).to(device))):
#             cross_entropy = 0.0
#             cross_entropy = torch.tensor(cross_entropy,requires_grad=True).to(device)
#         else:
#             cell_type = cls_prob.argmax(dim=1, keepdim=True)
#             cell_type, label = cell_type.view(-1), label.view(-1)
#             cls_loss = nn.CrossEntropyLoss()
#             cross_entropy = cls_loss(cls_prob,label.long())
#             accuracy = cell_type.eq(label.view_as(cell_type)).float().mean()
#             self.log('accuracy', accuracy, on_epoch=True, prog_bar=True, logger=True)
#             self.log('cross_entropy', cross_entropy, on_epoch=True, prog_bar=True, logger=True)

        loss = mse_loss
        self.log('Training_Loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        setup_seed()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        patch, adj, exp = patch.squeeze(0).to(device), adj.squeeze(0).to(device), exp.squeeze(0).to(device)
        
        pred_exp, *_ = self(patch, adj)
        
        """ Regression Loss"""
        mse_loss = F.mse_loss(pred_exp, exp)
        self.log('valid_mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
        
        loss = mse_loss
        self.log('Validation_Loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        setup_seed()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        patch, center, exp, label, adj, oris, sfs, *_ = batch
        patch, adj, exp = patch.squeeze(0).to(device), adj.squeeze(0).to(device), exp.squeeze(0).to(device)
        
        pred_exp, *_ = self(patch, adj)
        
        """ Regression Loss"""
        mse_loss = F.mse_loss(pred_exp, exp)
        self.log('test_mse_loss', mse_loss,on_epoch=True, prog_bar=True, logger=True)
        
        loss = mse_loss
        self.log('Test_Loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optim_dict = {'optimizer': optim}
        return optim_dict

if __name__ == "__main__":
    """ parameter setting """
    lr = 1e-3
    fold = 0
    epochs = 20
    model = Multiview_ST(learning_rate=lr)
#     pred, *_ = model(img,adj)
    
    """ Load dataset """
    trainset = pk_load(fold,'train', dataset='her2st',flatten=False,adj=True,ori=True,prune='Grid')
    validset = pk_load(fold,'val', dataset='her2st',flatten=False,adj=True,ori=True,prune='Grid')
    testset = pk_load(fold,'test', dataset='her2st',flatten=False,adj=True,ori=True,prune='Grid')
    train_loader = DataLoader(trainset, batch_size=1, num_workers=8, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=1, num_workers=8, shuffle=False)
    test_loader = DataLoader(testset, batch_size=1, num_workers=8, shuffle=False)
    
    """ Setup trainer """
    logger = pl.loggers.CSVLogger("logs", name=f"Multiview_ST_{fold}")
    trainer = pl.Trainer(accelerator='auto', callbacks=[EarlyStopping(monitor='Validation_Loss',mode='min')], 
                         min_epochs=epochs, logger=logger, )
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)
    """Save model and clean memory"""
    torch.save(model.state_dict(),f"./model/{pretrained_model}/Stage2-seed{seed}-epochs{epochs}-layer{num_layer}-sampleIndex{fold}.ckpt")
    
    import gc
    gc.collect()
    print("finished")

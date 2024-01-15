# %%
import os
import gc
import pandas as pd
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# %%
class FeatureExtractor(nn.Module):
    """
    Some Information about FeatureExtractor
    """
    def __init__(self, backbone='resnet50'):
        super(FeatureExtractor, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class NegativeBinomialLayer(nn.Module):
    def forward(self, x):
        # Get the number of dimensions of the input
        num_dims = x.dim()

        # Separate the parameters
        n, p = torch.unbind(x, dim=-1)

        # Add one dimension to make the right shape
        n = n.unsqueeze(-1)
        p = p.unsqueeze(-1)

        # Apply a softplus to make positive
        n = F.softplus(n)

        # Apply a sigmoid activation to bound between 0 and 1
        p = torch.sigmoid(p)

        # Join back together again
        out_tensor = torch.cat((n, p), dim=num_dims - 1)

        return out_tensor

class STimage(pl.LightningModule):
    def __init__(self, n_genes=10, learning_rate=1e-5, ft=False):
        super(STimage, self).__init__()
        self.learning_rate = learning_rate
        self.n_genes = n_genes
        self.ft_extr = FeatureExtractor()
        self.fc_base = nn.Linear(in_features=2048, out_features=512)

        if not ft:
            for param in self.ft_extr.parameters():
                param.requires_grad = True

        self.output_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 2), 
                          nn.Dropout(0.5),
                          NegativeBinomialLayer()) for _ in range(n_genes)
        ])

    def forward(self, x):
        x = self.ft_extr(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_base(x))
        
        param = torch.stack([output_layer(x) for output_layer in self.output_layers], dim=1)
        
        return param
    
    def predict_exp(self,x):
        # Extract r,p
        param = self(x)
        
        # Seperate 2 parameters
        r, p = torch.unbind(param, dim=-1)

        # Mean and variance
        mean = torch.mul(r, (1 - p) / p)
        var = torch.mul(r, (1 - p) / (p ** 2))
#         print(f"Mean:{mean}, \n Variance:{var}")
        
        return mean, var
    
    def negative_binomial_loss(self, y_true, y_pred):
        # Separate the parameters
        n, p = torch.unbind(y_pred, dim=-1)

#         # Add one dimension to make the right shape
#         n = n.unsqueeze(-1)
#         p = p.unsqueeze(-1)
#         print("n, y", n.shape, y_true.shape)

        # Calculate the negative log likelihood
        nll = (
                torch.lgamma(n)
                + torch.lgamma(y_true + 1)
                - torch.lgamma(n + y_true)
                - n * torch.log(p)
                - y_true * torch.log(1 - p)
        )

        return nll.mean()
    
    def training_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        param = self(patch)
        loss = self.negative_binomial_loss(exp, param)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        param = self(patch)
        loss = self.negative_binomial_loss(exp, param)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, _ = self.predict_exp(patch)
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return  pred, exp
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    
    

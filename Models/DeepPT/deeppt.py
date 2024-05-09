# %%
import os
import gc
import torch
import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet50'):
        super(FeatureExtractor, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        return x
   
class Autoencoder(nn.Module):
    # Auto encoder
    def __init__(self, hidden_dim=512, input_dim=2048):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        h = self.encoder(x)
        h = nn.Dropout(0.2)(h)
        x = self.decoder(h)
        return x, h
    

class DeepPT(pl.LightningModule):
    def __init__(self, n_genes=1630, hidden_dim=512, learning_rate=1e-4,):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.AE = Autoencoder(hidden_dim=512, input_dim=2048)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch,):
        AE_in = self.feature_extractor(patch)
        AE_out, h = self.AE(AE_in)
        h = nn.Dropout(0.2)(h)
        pred = self.pred_head(F.relu(h))
        return pred, AE_in, AE_out

    def training_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, AE_in, AE_out = self(patch)
        loss = F.mse_loss(pred, exp) + F.mse_loss(AE_in, AE_out)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, AE_in, AE_out = self(patch)
        loss = F.mse_loss(pred, exp) + F.mse_loss(AE_in, AE_out)
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, AE_in, AE_out = self(patch)
        loss = F.mse_loss(pred, exp) + F.mse_loss(AE_in, AE_out)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, *_ = self(patch)
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return  pred, exp
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

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
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='vgg16'):
        super(FeatureExtractor, self).__init__()
        self.backbone = torchvision.models.vgg16(pretrained=True)
        self.backbone.classifier[6] = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        return x

class DeepSpaCE(pl.LightningModule):
    def __init__(self, n_genes=1630, hidden_dim=4096, learning_rate=1e-4,):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch,):
        feature = self.feature_extractor(patch)
        h = feature
        pred = self.pred_head(F.relu(h))
        return pred

    def training_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred = self(patch)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred = self(patch)
        loss = F.mse_loss(pred, exp)
        self.log('val_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred = self(patch)
        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred = self(patch)
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return  pred, exp
    
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

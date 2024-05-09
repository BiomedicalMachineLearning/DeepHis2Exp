# %%
import os
import gc
import cv2
import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import torchvision.transforms as tf
import torchvision.transforms.functional as TF

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
    

class CNN_DAE(pl.LightningModule):
    def __init__(self, n_genes=1630, hidden_dim=512, learning_rate=1e-4, trans='blur'):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.AE = Autoencoder(hidden_dim=512, input_dim=2048)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.learning_rate = learning_rate
        self.n_genes = n_genes
        self.trans = trans

    def forward(self, patch,):
        aug_patch = self.aug(patch)
        ori_ft = self.feature_extractor(patch)
        AE_in = self.feature_extractor(aug_patch)
        AE_out, h = self.AE(AE_in)
        h = nn.Dropout(0.2)(h)
        pred = self.pred_head(F.relu(h))
        return pred, ori_ft, AE_out
    
    def aug(self, image, trans):
        # Define the transformations
        trans = ["blur", "random_grayscale", "random_rotation", "none"]

        # Randomly select one augmentation
        selected_augmentation = trans[random.randint(0, 4)]

        # Apply the selected augmentation
        if selected_augmentation == "blur":
            # Gassian blur
            image = tf.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))(image)

        elif selected_augmentation == "random_grayscale":
            # Random grayscale
            image = tf.RandomGrayscale(0.1)(image)

        elif selected_augmentation == "random_rotation":
            # Random flipping and rotations
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            if random.random() > 0.5:
                image = TF.rotate(image, random.choice([180, 90, 0, -90]))

        elif selected_augmentation == "none":
            # No augmentation
            pass
        return image
    
    def training_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, ori_ft, AE_out = self(patch)
        loss = F.mse_loss(pred, exp) + F.mse_loss(ori_ft, AE_out)
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

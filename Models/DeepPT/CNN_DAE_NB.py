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

class CNN_DAE_NB(pl.LightningModule):
    def __init__(self, n_genes=1630, hidden_dim=512, learning_rate=1e-4, paralle=False):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.AE = Autoencoder(hidden_dim=512, input_dim=2048)
        self.output_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 2), 
                          nn.Dropout(0.5),
                          NegativeBinomialLayer()) for _ in range(n_genes)
        ])
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.output_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(512, 2), 
                          nn.Dropout(0.5),
                          NegativeBinomialLayer()) for _ in range(n_genes)
        ])
        self.fc_base = nn.Linear(in_features=2048, out_features=512)
        self.learning_rate = learning_rate
        self.n_genes = n_genes
        self.paralle = paralle

    def forward(self, patch,):
        # Augmentation
        aug_patch = self.aug(patch)

        ori_ft = self.feature_extractor(patch)

        # Denoisy Autoencoder
        AE_in = self.feature_extractor(aug_patch)
        AE_out, h = self.AE(AE_in)

        # Latent space from DAE
        h = nn.Dropout(0.2)(h)

        # Prediction from Negative Binomial
        if self.paralle:
            param = self.return_param(ori_ft)
            pred_NB, var = self.sample_exp(param)
            pred_DAE = self.pred_head(F.relu(h))
            pred = (pred_DAE + pred_NB) / 2 # Combine 2 predictions
        else:
            param = self.return_param(h)
            pred_NB, var = self.sample_exp(param)
            pred = pred_NB
        return pred, ori_ft, AE_out, param
    
    def return_param(self, x):
        # x is the extracted features from CNN
        x = x.view(x.size(0), -1)
        if self.paralle:
            x = F.relu(self.fc_base(x))
        else:
            x = F.relu(x)
        param = torch.stack([output_layer(x) for output_layer in self.output_layers], dim=1)
        return param
    
    def sample_exp(self, param):
        # param is the parameters (r, p) for Negative Binomial distribution
        
        # Seperate 2 parameters
        r, p = torch.unbind(param, dim=-1)

        # Mean and variance
        mean = torch.mul(r, (1 - p) / p)
        var = torch.mul(r, (1 - p) / (p ** 2))
        return mean, var
    
    def negative_binomial_loss(self, y_true, y_pred):
        # Separate the parameters
        n, p = torch.unbind(y_pred, dim=-1)

        # Calculate the negative log likelihood
        nll = (
                torch.lgamma(n)
                + torch.lgamma(y_true + 1)
                - torch.lgamma(n + y_true)
                - n * torch.log(p)
                - y_true * torch.log(1 - p)
        )
        return nll.mean()
    
    def recon_loss(self, ori_ft, AE_out):
        # ori_ft is the feature extracted from original patch
        # AE_out is the reconstructed feature from augmented patch
        return F.mse_loss(ori_ft, AE_out)
    
    def aug(self, image):
        # Define the transformations
        trans = ["blur", "random_grayscale", "random_rotation", "none"]

        # Randomly select one augmentation
        selected_augmentation = trans[random.randint(0, 3)]

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
        pred, ori_ft, AE_out, param = self(patch)
        if self.paralle:
            loss = F.mse_loss(pred, exp) + self.recon_loss(ori_ft, AE_out) + self.negative_binomial_loss(exp, param)
        else:
            loss = self.recon_loss(ori_ft, AE_out) + self.negative_binomial_loss(exp, param)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('NB_loss', self.negative_binomial_loss(exp, param), on_epoch=True, prog_bar=True, logger=True)
        self.log('recon_loss', self.recon_loss(ori_ft, AE_out), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, ori_ft, AE_out, param = self(patch)
        if self.paralle:
            loss = F.mse_loss(pred, exp) + self.recon_loss(ori_ft, AE_out) + self.negative_binomial_loss(exp, param)
        else:
            loss = self.recon_loss(ori_ft, AE_out) + self.negative_binomial_loss(exp, param)
        self.log('valid_loss', loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch = patch.squeeze(0)
        pred, ori_ft, AE_out, param = self(patch)
        if self.paralle:
            loss = F.mse_loss(pred, exp) + self.recon_loss(ori_ft, AE_out) + self.negative_binomial_loss(exp, param)
        else:
            loss = self.recon_loss(ori_ft, AE_out) + self.negative_binomial_loss(exp, param)
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

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
import gc
import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import cv2
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class ImageEncoder(nn.Module):
    """
    Some Information about image encoder
    """
    def __init__(self, backbone='resnet50'):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        return x
    
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    
class BLEEP(pl.LightningModule):
    def __init__(
        self,
        temperature=1.0,
        image_embedding=2048,
        n_genes=1000,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #Latent space, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=n_genes)  
        self.temperature = temperature

    def forward(self, patch, exp):
        # Getting Image and spot Features
        image_features = self.image_encoder(patch)
        spot_features = exp
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = self.cross_entropy(logits, targets, reduction='none')
        images_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()    
    
    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        
    def aug(self, image):
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        image = TF.rotate(image, random.choice([180, 90, 0, -90]))
        return image

    def training_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch, exp = patch.squeeze(0), exp.squeeze(0)
        patch = self.aug(patch)
        loss = self(patch, exp)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch, exp = patch.squeeze(0), exp.squeeze(0)
        loss = self(patch, exp)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch, exp = patch.squeeze(0), exp.squeeze(0)
        loss = self(patch, exp)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.AdamW(
        self.parameters(), lr=1e-3, weight_decay=1e-3)
        return optimizer  
    
    def predict_step(self, batch, batch_idx):
        patch, _, exp, *_ = batch
        patch, exp = patch.squeeze(0), exp.squeeze(0)
        image_embeddings = self.image_projection(self.image_encoder(patch))
        image_embeddings = image_embeddings.cpu().numpy()
        if self.spot_projection.projection.in_features == exp.shape[1]:
            spot_embeddings = self.spot_projection(exp)
            spot_embeddings = spot_embeddings.cpu().numpy()
            return  image_embeddings, spot_embeddings, exp
        else:
            # output format: img_emb_(num_spots, 256) exp_emb_(num_spots, 256) exp_(num_spots, n_genes)
            return  image_embeddings, _, exp    
    
    
def find_matches(spot_embeddings, query_embeddings, top_k=1):
    #find the closest matches 
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265
#     print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    return indices.cpu().numpy()

def bleep_inference(model=None, trainer=None, tr_loader=None, te_loader=None, method="average", ):
    # Build database from train dataset
    tr_out = trainer.predict(model, tr_loader) # tr_loader
    tr_image_embeddings = np.concatenate([tr_out[i][0] for i in range(len(tr_out))])
    tr_spot_embeddings = np.concatenate([tr_out[i][1] for i in range(len(tr_out))])
    tr_exp = np.concatenate([tr_out[i][2] for i in range(len(tr_out))])

    # Build query database from test dataset
    te_out = trainer.predict(model, te_loader) # te_loader
    te_image_embeddings = np.concatenate([te_out[i][0] for i in range(len(te_out))])
#     te_spot_embeddings = np.concatenate([te_out[i][1] for i in range(len(te_out))])
    te_exp = np.concatenate([te_out[i][2] for i in range(len(te_out))])
    if method == "simple":
        indices = find_matches(tr_spot_embeddings, te_image_embeddings, top_k=1)
        matched_spot_embeddings_pred = tr_spot_embeddings[indices[:,0],:]
        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        matched_spot_expression_pred = tr_exp[indices[:,0],:]
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "average":
        print("finding matches, using average of top 50 expressions")
        indices = find_matches(tr_spot_embeddings, te_image_embeddings, top_k=50)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], tr_spot_embeddings.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], tr_exp.shape[1]))
        for i in range(indices.shape[0]):
            matched_spot_embeddings_pred[i,:] = np.average(tr_spot_embeddings[indices[i,:],:], axis=0)
            matched_spot_expression_pred[i,:] = np.average(tr_exp[indices[i,:],:], axis=0)

        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "weighted_average":
        print("finding matches, using weighted average of top 50 expressions")
        indices = find_matches(tr_spot_embeddings, te_image_embeddings, top_k=50)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], tr_spot_embeddings.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], tr_exp.shape[1]))
        for i in range(indices.shape[0]):
            a = np.sum((tr_spot_embeddings[indices[i,0],:] - te_image_embeddings[i,:])**2) #the smallest MSE
            weights = np.exp(-(np.sum((tr_spot_embeddings[indices[i,:],:] - te_image_embeddings[i,:])**2, axis=1)-a+1))
            if i == 0:
                print("weights: ", weights)
            matched_spot_embeddings_pred[i,:] = np.average(tr_spot_embeddings[indices[i,:],:], axis=0, weights=weights)
            matched_spot_expression_pred[i,:] = np.average(tr_exp[indices[i,:],:], axis=0, weights=weights)

        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)
        
    return te_exp, matched_spot_expression_pred

    
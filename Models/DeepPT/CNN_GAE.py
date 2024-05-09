# %%
import torch
import pytorch_lightning as pl
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.nn import GCNConv, GINConv, GATConv, BatchNorm
# from torch_geometric.nn.models import GAE


class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet50'):
        super(FeatureExtractor, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        return x

class GraphAE(nn.Module):
    def __init__(self, gnn="GIN", in_channels=2048, hidden_dim=512, ):
        super().__init__()
        self.gnn = gnn
        if gnn == "GIN":
            self.encoder = GINConv(nn.Linear(in_channels, hidden_dim),eps=1e-5)
            self.decoder = GINConv(nn.Linear(hidden_dim, in_channels), eps=1e-5)
        elif gnn == "GCN":
            self.encoder = GCNConv(in_channels, hidden_dim)  
            self.decoder = GCNConv(hidden_dim, in_channels)
        elif gnn == "GAT":
            self.encoder = GATConv(in_channels, hidden_dim, heads=2, concat=False)  
            self.decoder = GATConv(hidden_dim, in_channels, heads=2, concat=False)
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(in_channels)

    def forward(self, x, edge_index, edge_weights):
        if self.gnn == "GIN":
            x = self.encoder(x, edge_index)
        else:
            x = self.encoder(x, edge_index, edge_weights)
        x = self.norm1(x)
        z = nn.ReLU()(x)
        if self.gnn == "GIN":
            x = self.decoder(x, edge_index)
        else:
            x = self.decoder(x, edge_index, edge_weights)
        x = self.norm2(x)
        AE_out = nn.ReLU()(x)
        return z, AE_out
    
class CNNGAE(pl.LightningModule):
    def __init__(self, gnn="GIN", n_genes=1630, hidden_dim=512, learning_rate=1e-4,):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = gnn
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.AE = GraphAE(gnn=gnn, hidden_dim=hidden_dim, in_channels=2048)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, edge_index, edge_weights):
        AE_in = self.feature_extractor(patch)
        z, AE_out = self.AE(AE_in, edge_index, edge_weights)
        pred = self.pred_head(F.relu(z))
        recon_loss = F.mse_loss(AE_in, AE_out)
        return pred, z, recon_loss

    def training_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch = patch.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_weights = edge_weights.squeeze(0)
        pred, _, recon_loss = self(patch, edge_index, edge_weights)
        loss = F.mse_loss(pred, exp) + recon_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch = patch.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_weights = edge_weights.squeeze(0)
        pred, _, recon_loss = self(patch, edge_index, edge_weights)
        loss = F.mse_loss(pred, exp) + recon_loss
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch = patch.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_weights = edge_weights.squeeze(0)
        pred, _, recon_loss = self(patch, edge_index, edge_weights)
        loss = F.mse_loss(pred, exp) + recon_loss
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch = patch.squeeze(0)
        edge_index = edge_index.squeeze(0)
        edge_weights = edge_weights.squeeze(0)
        pred, *_ = self(patch, edge_index, edge_weights)
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return  pred, exp
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

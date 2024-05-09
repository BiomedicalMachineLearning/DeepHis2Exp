# %%
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch_geometric.nn import GCNConv


class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet50'):
        super(FeatureExtractor, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
    def forward(self, x):
        x = self.backbone(x)
        return x

class GNN(nn.Module):
    def __init__(self, in_channels=512, out_channels=512,):
        super().__init__()
        self.conv = nn.ModuleList([GCNConv(in_channels, out_channels) for i in range(3)])
        self.lstm = nn.LSTM(out_channels, out_channels, 2)

    def forward(self, x, edge_index, edge_weights):
        jk = [gcn(x, edge_index, edge_weights).unsqueeze(0) for gcn in self.conv]
        x = torch.cat(jk,0).to(torch.float32)
        x, _ = self.lstm(x)
        x = x.mean(0)
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
        out = self.decoder(h)
        return out, h, F.mse_loss(x, out)
    
class CNN_AE_GNN(pl.LightningModule):
    def __init__(self, n_genes=1630, hidden_dim=512, learning_rate=1e-4,):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.AE = Autoencoder(hidden_dim=512, input_dim=2048)
        self.GNN = GNN(in_channels=hidden_dim, out_channels=hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, edge_index, edge_weights):
        """CNN+AE+GNN"""
        x = self.feature_extractor(patch)
        recon, h, recon_loss  = self.AE(x)
        h = self.GNN(h, edge_index, edge_weights)
        pred = self.pred_head(F.relu(h))
        return pred, h, recon_loss

    def training_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
        pred, _, recon_loss = self(patch, edge_index, edge_weights)
        loss = F.mse_loss(pred, exp) + 10*recon_loss
        self.log('train_loss', loss, logger=True)
        self.log('mse', F.mse_loss(pred, exp), logger=True)
        self.log('recon_loss', recon_loss, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
        pred, _, recon_loss = self(patch, edge_index, edge_weights)
        loss = F.mse_loss(pred, exp) + recon_loss
        self.log('val_loss', loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
        pred, _, recon_loss = self(patch, edge_index, edge_weights)
        loss = F.mse_loss(pred, exp) + recon_loss
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        patch, _, exp, *_, edge_index, edge_weights = batch
        patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
        pred, *_ = self(patch, edge_index, edge_weights)
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return  pred, exp
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

class CNN_GNN_AE(CNN_AE_GNN):
    def __init__(self, n_genes=1630, hidden_dim=512, learning_rate=1e-4,):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = FeatureExtractor()
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        self.AE = Autoencoder(hidden_dim, input_dim=1024)
        self.GNN = GNN(in_channels=2048, out_channels=1024)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, edge_index, edge_weights):
        """CNN+GNN+AE"""
        h = self.feature_extractor(patch)
        h = self.GNN(h, edge_index, edge_weights)
        recon, h, recon_loss  = self.AE(h)
        pred = self.pred_head(F.relu(h))
        return pred, h, recon_loss

    # def training_step(self, batch, batch_idx):
    #     patch, _, exp, *_, edge_index, edge_weights = batch
    #     patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
    #     pred, _, recon_loss = self(patch, edge_index, edge_weights)
    #     loss = F.mse_loss(pred, exp) + 10*recon_loss
    #     self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
    #     self.log('mse', F.mse_loss(pred, exp), on_epoch=True, prog_bar=True, logger=True)
    #     self.log('recon_loss', recon_loss, on_epoch=True, prog_bar=True, logger=True)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     patch, _, exp, *_, edge_index, edge_weights = batch
    #     patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
    #     pred, _, recon_loss = self(patch, edge_index, edge_weights)
    #     loss = F.mse_loss(pred, exp) + recon_loss
    #     self.log('val_loss', loss)
    #     return loss
        
    # def test_step(self, batch, batch_idx):
    #     patch, _, exp, *_, edge_index, edge_weights = batch
    #     patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
    #     pred, _, recon_loss = self(patch, edge_index, edge_weights)
    #     loss = F.mse_loss(pred, exp) + recon_loss
    #     self.log('test_loss', loss)
    #     return loss

    # def predict_step(self, batch, batch_idx):
    #     patch, _, exp, *_, edge_index, edge_weights = batch
    #     patch, edge_index, edge_weights = patch.squeeze(0), edge_index.squeeze(0), edge_weights.squeeze(0)
    #     pred, *_ = self(patch, edge_index, edge_weights)
    #     pred = pred.squeeze(0).cpu().numpy()
    #     exp = exp.squeeze(0).cpu().numpy()
    #     return  pred, exp
    
    # def configure_optimizers(self):
    #     # self.hparams available because we called self.save_hyperparameters()
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer

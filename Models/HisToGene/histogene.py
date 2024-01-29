import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from Models.HisToGene.Transformer import ViT

class HisToGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):   
        patch, center, exp, *_ = batch
        patch = patch.flatten(2)
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('train_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp, *_ = batch
        patch = patch.flatten(2)
        pred = self(patch, center)
        loss = F.mse_loss(pred.view_as(exp), exp)
        self.log('val_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        patch, center, exp, *_ = batch
        patch = patch.flatten(2)
        patch = patch.squeeze(0)
        pred = self(patch, center)
        pred = pred.squeeze(0).cpu().numpy()
        exp = exp.squeeze(0).cpu().numpy()
        return  pred, exp

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

# if __name__ == '__main__':
#     a = torch.rand(1,4000,3*112*112)
#     p = torch.ones(1,4000,2).long()
#     model = HisToGene()
#     print(count_parameters(model))
#     x = model(a,p)
#     print(x.shape)
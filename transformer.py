import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn import GATConv
from dgl import LaplacianPE
from dgl.nn import LaplacianPosEnc

import squidpy as sq
from squidpy.im import ImageContainer
import numpy as np
import scanpy as sc
import anndata as ad

from utilss import *

from performer_pytorch import Performer
from torchinfo import summary
import timm
import warnings
import pickle

from torchinfo import summary
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
    ViTModel,
    ViTForImageClassification,
    ViTMAEConfig,
    ViTMAEForPreTraining,
    AutoConfig,
    AutoImageProcessor,
    ViTMAEModel,
)
from feat import ImageFeatureExtractor


class MasterModel(nn.Module):
    def __init__(self, image_model_class, out_dim,
                 num_eigvecs=10, hidden_dim=768, dropout=0.1,
                 keep_unfrozen=None, train=True, model_pre=None):
        super().__init__()
        self.keep_unfrozen = keep_unfrozen
        self.stem = ImageFeatureExtractor(image_model_class, model_pre=model_pre,
                                          out_dim=hidden_dim, keep_unfrozen=keep_unfrozen, train=train)
        self.add_lap_enc = LaplacianPE(k=num_eigvecs, feat_name='pe', eigval_name='eigval', padding=True)
        # self.pos_embed = LaplacianPosEnc(model_type="DeepSet", num_layer=4, k=num_eigvecs,
        #                          lpe_dim=hidden_dim, num_post_layer=2)
        self.pos_embed = LaplacianPosEnc(model_type="Transformer", num_layer=4, k=num_eigvecs,
                                     lpe_dim=hidden_dim, n_head=4)
        self.performer = Performer(
                            dim = hidden_dim,
                            depth = 8,
                            heads = 8,
                            causal = False,
                            dim_head = 64,
                            ff_dropout = 0.1,
                            attn_dropout = 0.2,
                        )
        ## TODO Add dropout params to performer
        self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.ReLU()
            )

    def forward(self, inputs):
        # inputs should be a dgl grpah
        g = inputs
        if 'eigval' in g.ndata:
            EigVals, EigVecs = g.ndata['eigval'], g.ndata['pe']
        else:
            g = self.add_lap_enc(g)
            warnings.warn("Eigenvectors not precomputed.")
        x = g.ndata['img']
        x = self.stem(x)
        x = x + self.pos_embed(EigVals, EigVecs)
        x = self.performer(x.unsqueeze(0))
        # Batch norm?
        x = self.decoder(x.squeeze(0))
        return x

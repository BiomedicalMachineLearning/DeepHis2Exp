from gin import GIN, GIN_nopool
from gatv2 import GATv2
from gcn import GCN
import torch.nn.functional as F
import torch
import warnings
import pickle
from feat import ImageFeatureExtractor
from utilss import *
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/train.py
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gcn/train.py
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/gatv2/train.py

# Examples
### GCN
# in_size = 768
# out_size = 10
# model_gcn = GCN(in_size, 512, out_size)

### GIN
# model_gin = GIN_nopool(in_size, 512, out_size)

### GATV2
# heads = ([8] * 2) + [1] # ([args.num_heads] * args.num_layers) + [args.num_out_heads]
# model = GATv2(
#     2, # args.num_layers
#     768, # num_feats
#     8, # num_hidden
#     10, # n_classes
#     heads,
#     F.elu,
#     0.7, # in_drop
#     0.7, # attn_drop
#     0.2, # args.negative_slope
#     False, # Residual
# )

class MasterModelGNN(nn.Module):
    def __init__(self, image_model_class, graphnet, out_dim,
                 num_eigvecs=10, hidden_dim=768, dropout=0.1,
                 keep_unfrozen=None, train=True, model_pre=None):
        super().__init__()
        self.keep_unfrozen = keep_unfrozen
        self.stem = ImageFeatureExtractor(image_model_class, model_pre=model_pre,
                                          out_dim=hidden_dim, keep_unfrozen=keep_unfrozen, train=train)
        self.graphnet_type = graphnet
        if graphnet == "GAT":
            num_layers = 2
            heads = ([8] * num_layers) + [1]
            self.graphnet = GATv2(
                                    num_layers, # args.num_layers
                                    hidden_dim, # num_feats
                                    8, # num_hidden
                                    out_dim, # n_classes
                                    heads,
                                    F.elu,
                                    0.5, # in_drop (changed from 0.7)
                                    0.5, # attn_drop (changed from 0.7)
                                    0.2, # args.negative_slope
                                    False, # Residual
                                )
        elif graphnet == "GIN":
            self.graphnet = GIN_nopool(hidden_dim, hidden_dim, out_dim)
        elif graphnet == "GCN":
            self.graphnet = GCN(hidden_dim, hidden_dim, out_dim)
        self.decoder = nn.ReLU()

    def forward(self, inputs):
        # inputs should be a dgl grpah
        g = inputs
        x = g.ndata['img']
        x = self.stem(x)
        x = self.graphnet(g, x)
        # # Batch norm?
        x = self.decoder(x)
        return x

import os
import torch
import random
import argparse
import pickle as pk
import pytorch_lightning as pl
from utils import *
from HIST2ST import *
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='the id of gpu.')
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=12000, help='random seed.')
parser.add_argument('--epochs', type=int, default=350, help='number of epochs.')
parser.add_argument('--name', type=str, default='hist2ST', help='prefix name.')
# parser.add_argument('--data', type=str, default='her2st', help='dataset name:{"her2st","cscc"}.')
# parser.add_argument('--logger', type=str, default='../logs/my_logs', help='logger path.')
parser.add_argument('--norm', type=str, default="lognorm", help='Gene expression pre-processing')

parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout.')

parser.add_argument('--bake', type=int, default=5, help='the number of augmented images.')
parser.add_argument('--lamb', type=float, default=0.5, help='the loss coef of self-distillation.')


parser.add_argument('--nb', type=str, default='F', help='zinb or nb loss.')
parser.add_argument('--zinb', type=float, default=0.25, help='the loss coef of zinb.')

parser.add_argument('--prune', type=str, default='Grid', help='how to prune the edge:{"Grid","NA"}')
parser.add_argument('--policy', type=str, default='mean', help='the aggregation way in the GNN .')
parser.add_argument('--neighbor', type=int, default=4, help='the number of neighbors in the GNN.')

parser.add_argument('--tag', type=str, default='5-7-2-8-4-16-32', 
                    help='hyper params: kernel-patch-depth1-depth2-depth3-heads-channel,'
                         'depth1-depth2-depth3 are the depth of Convmixer, Multi-head layer in Transformer, and GNN, respectively'
                         'patch is the value of kernel_size and stride in the path embedding layer of Convmixer'
                         'kernel is the kernel_size in the depthwise of Convmixer module'
                         'heads are the number of attention heads in the Multi-head layer'
                         'channel is the value of the input and output channel of depthwise and pointwise. ')

args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),args.tag.split('-'))


import sys
sys.path.append("./")

import time
from window_adata import *
import os
import sys
import numpy as np
import pandas as pd 
import h5py
import json
import pickle
import anndata
import scanpy as sc
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import glob
import torchvision
import scprep as scp
import anndata as ad
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm
from scipy.stats import pearsonr,spearmanr
from scipy import stats
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
from matplotlib.image import imread
from scanpy import read_visium, read_10x_mtx
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from anndata import read as read_h5ad
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
from matplotlib.image import imread
import pickle

from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,)

""" Integrate two visium datasets """
data_dir1 = "./Alex_NatGen_6BreastCancer/"
data_dir2 = "./breast_cancer_10x_visium/"

samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
samps2 = ["block1", "block2", "FFPE"]

sampsall = samps1 + samps2
samples1 = {i:data_dir1 + i for i in samps1}
samples2 = {i:data_dir2 + i for i in samps2}

# Marker gene list
# gene_list = ["COX6C","TTLL12", "HSP90AB1", "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

# Highly variable genes
with open('../1000hvg_common.pkl', 'rb') as f:
    gene_list = pickle.load(f)
gene_list = list(gene_list)

# # Load windowed dataset
with open('../10x_visium_dataset_without_window.pickle', 'rb') as f:
    adata_dict0 = pickle.load(f)
for i in samps2:    
    adata_dict0[i].var_names_make_unique()
    
# Define the gridding size
sizes = [4000 for i in range(len(adata_dict0))]

# Split tiles into smaller patches according to gridding size
adata_dict = window_adata(adata_dict0, sizes)
# del adata_dict0

# For training
from data_vit import ViT_Anndata

def dataset_wrap(fold = 0, gene_list=gene_list, dataloader= True):
    test_sample = sampsall[fold] # Split one sample as test sample
    train_sample = list(set(sampsall)-set(test_sample)) # Other samples are training samples

    tr_name = list(set([i for i in list(adata_dict.keys()) for tr in train_sample if tr in i]))
    te_name = list(set([i for i in list(adata_dict.keys()) if test_sample in i]))

    trainset = ViT_Anndata(adata_dict = adata_dict, train_set = tr_name, gene_list = gene_list, norm=args.norm, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    testset = ViT_Anndata(adata_dict = adata_dict, train_set = te_name, gene_list = gene_list, norm=args.norm, train=True, flatten=False, ori=True, prune='NA', neighs=4, )

    print("LOADED TRAINSET")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    if dataloader==True:
        return train_loader, test_loader
    else:
        return trainset, testset
    
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import numpy as np
import torch.nn as nn

from gcn import *
from transformer import *
from NB_module import *
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

from torchvision import models, transforms

# set random seed
def setup_seed(seed=12000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1
import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from utils import *

"""For training only"""
import gc
from data_vit import ViT_Anndata
from pytorch_lightning import seed_everything
import time

start_time = time.time()
"""Training loops"""
seed=12000
epochs=350

"""Load dataset"""
train_loader, test_loader = dataset_wrap(fold=args.fold, dataloader=True)

"""Define model"""
model = Hist2ST(
    depth1=2, depth2=8, depth3=4,n_pos=128,
    n_genes=len(gene_list), learning_rate=1e-5,
    kernel_size=5, patch_size=7, fig_size=112,
    heads=16, channel=32, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
    policy='mean', 
)
setup_seed(seed)

"""Setup trainer"""
trainer = pl.Trainer(accelerator='auto', max_epochs=epochs, logger=False)
trainer.fit(model, train_loader)

"""Save model and clean memory"""
torch.save(model.state_dict(),f"./model/Hist2st-norm{args.norm}-{args.fold}.ckpt")
gc.collect()
del train_loader, model
    
end_time = time.time()
execution_time = end_time - start_time
print("Training time: ", execution_time/3600, " hours")
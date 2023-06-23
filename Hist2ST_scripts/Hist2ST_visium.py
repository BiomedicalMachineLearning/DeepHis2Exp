import sys
sys.path.append("./")
# sys.path.append("../../scripts/")

import time

from window_adata import *
from read_stimage_genes import read_gene_set_hvg,intersect_section_genes

import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict import *
from HIST2ST import *
from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from scipy import stats
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
import pickle
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import sys
from pathlib import Path
import os
import numpy as np
import pandas as pd
from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO
import h5py
import json
from matplotlib.image import imread
import matplotlib.pyplot as plt

import anndata
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,
)
from anndata import read as read_h5ad
from anndata import read_h5ad
import scanpy as sc
from scanpy import read_visium, read_10x_mtx

import glob
import torch
import torchvision
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from pathlib import Path, PurePath
from typing import Union, Dict, Optional, Tuple, BinaryIO

import h5py
import json
import numpy as np
import pandas as pd
from matplotlib.image import imread
import anndata
from anndata import (
    AnnData,
    read_csv,
    read_text,
    read_excel,
    read_mtx,
    read_loom,
    read_hdf,
)
from anndata import read as read_h5ad
def read_visium_alex(
    path: Union[str, Path],
    genome: Optional[str] = None,
    *,
    count_dir: str = "raw_feature_bc_matrix",
    library_id: str = None,
    load_images: Optional[bool] = True,
    source_image_path: Optional[Union[str, Path]] = None,
) -> AnnData:
    path = Path(path)
    adata = read_10x_mtx(path / count_dir)

    adata.uns["spatial"] = dict()

    from h5py import File

    adata.uns["spatial"][library_id] = dict()

    if load_images:
        files = dict(
            tissue_positions_file=path / 'spatial/tissue_positions_list.csv',
            scalefactors_json_file=path / 'spatial/scalefactors_json.json',
            hires_image=path / 'spatial/tissue_hires_image.png',
            lowres_image=path / 'spatial/tissue_lowres_image.png',
        )

        # check if files exists, continue if images are missing
        for f in files.values():
            if not f.exists():
                if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                    logg.warning(
                        f"You seem to be missing an image file.\n"
                        f"Could not find '{f}'."
                    )
                else:
                    raise OSError(f"Could not find '{f}'")

        adata.uns["spatial"][library_id]['images'] = dict()
        for res in ['hires', 'lowres']:
            try:
                adata.uns["spatial"][library_id]['images'][res] = imread(
                    str(files[f'{res}_image'])
                )
            except Exception:
                raise OSError(f"Could not find '{res}_image'")

        # read json scalefactors
        adata.uns["spatial"][library_id]['scalefactors'] = json.loads(
            files['scalefactors_json_file'].read_bytes()
        )

        adata.uns["spatial"][library_id]["metadata"] = {}

        # read coordinates
        positions = pd.read_csv(files['tissue_positions_file'], header=None)
        positions.columns = [
            'barcode',
            'in_tissue',
            'array_row',
            'array_col',
            'pxl_col_in_fullres',
            'pxl_row_in_fullres',
        ]
        positions.index = positions['barcode']

        adata.obs = adata.obs.join(positions, how="left")

        adata.obsm['spatial'] = adata.obs[
            ['pxl_row_in_fullres', 'pxl_col_in_fullres']
        ].to_numpy()
        adata.obs.drop(
            columns=['barcode', 'pxl_row_in_fullres', 'pxl_col_in_fullres'],
            inplace=True,
        )

        # put image path in uns
        if source_image_path is not None:
            # get an absolute path
            source_image_path = str(Path(source_image_path).resolve())
            adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
                source_image_path
            )

    return adata

def calculate_correlation(attr_1, attr_2):
    r = stats.pearsonr(attr_1, 
                       attr_2)[0]
    return r

def calculate_correlation_2(attr_1, attr_2):
    r = stats.spearmanr(attr_1, 
                       attr_2)[0]
    return r


data_dir1 = "../../data/Alex_NatGen/"
data_dir2 = "../../data/Breast_Cancer_10x/"

samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
# samps2 = ["block1", "block2", "FFPE", "CytAssist_FFPE"]
# samps2 = ["block1", "block2", "FFPE", "1168993F", "CytAssist_FFPE"]
samps2 = ["block1", "block2", "FFPE", "1168993F"]

# Removing CytAssist_FFPE until below is fixed
# Traceback (most recent call last):
#   File "/scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/scripts/histogene_ffpe2.py", line 810, in <module>
#     adata_dict = window_adata(adata_dict0, sizes)
#   File "/scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/scripts/window_adata.py", line 32, in window_adata
#     layer='image', library_id=key)
#   File "/scratch/imb/uqjxie6/benchmmarking/DeepHis2Exp/envs/Hist2ST/lib/python3.7/site-packages/squidpy/im/_container.py", line 300, in add_img
#     ) from None
# ValueError: Expected image to have `0` Z-dimension(s), found `2`.



sampsall = samps1 + samps2

samples1 = {i:data_dir1 + i for i in samps1}
samples2 = {i:data_dir2 + i for i in samps2}

adata_dict1 = {name: read_visium_alex(path, library_id = name, source_image_path = path + f"/image.tif") for name,path in samples1.items()}
adata_dict2 = {name: read_visium(path, library_id = name, source_image_path = path + "/image.tif") for name,path in samples2.items()}

adata_dict0 = {**adata_dict1, **adata_dict2}


sizes = [4000 for i in range(len(adata_dict0))]

adata_dict = window_adata(adata_dict0, sizes)

# gene_list = read_gene_set_hvg("../../data/pfizer2/") # train_adata.h5ad
gene_list = set(["COX6C","TTLL12", "HSP90AB1", 
           "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"])
# gene_list = ['CD4', 'TRAC', 'CXCR4']
# gene_list = set(gene_list)

gene_list = intersect_section_genes(gene_list, adata_dict)
n_genes = len(gene_list)
print("number of genes: ", n_genes)
# with open('../../data/hist2st_hvg1000.pickle', 'wb') as f:
#     pickle.dump(gene_list, f)



from data_vit import ViT_Anndata


device='cuda'
tag='5-7-2-8-4-16-32'
k,p,d1,d2,d3,h,c=map(lambda x:int(x),tag.split('-'))
dropout=0.2
random.seed(12000)
np.random.seed(12000)
torch.manual_seed(12000)
torch.cuda.manual_seed(12000)
torch.cuda.manual_seed_all(12000)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

df = pd.DataFrame()
i = int(sys.argv[1])
# i=1
fold = i
test_sample = sampsall[fold]
test_sample_orig = sampsall[fold]

import os
if os.path.exists(f"../../results/ffpe/hist2st_cor_{test_sample_orig}.csv"):
    sys.exit("Results already exist")


fold2name = dict(enumerate(sampsall))



# gene_list = ["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
#            "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
# genes = len(gene_list)

train_set = list(set(list(adata_dict.keys())) - set([i for i in list(adata_dict.keys()) if test_sample in i]))

trainset = ViT_Anndata(adata_dict = adata_dict, train_set = train_set, gene_list = gene_list,
            train=True,flatten=False,adj=True,ori=True,prune='NA',neighs=4, 
        )

print("LOADED TRAINSET")


train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

model=Hist2ST(
    depth1=d1, depth2=d2,depth3=d3,n_genes=n_genes,
    kernel_size=k, patch_size=p,
    heads=h, channel=c, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, n_pos=128,
)

logger=None
trainer = pl.Trainer(
    gpus=[0], max_epochs=350, 
    logger=logger,
)

start_train = time.perf_counter()
trainer.fit(model, train_loader)

end_train = time.perf_counter()
import os
if not os.path.isdir("../../trained_models/"):
    os.mkdir("../../trained_models/")

# torch.save(model.state_dict(),f"../../trained_models/{test_sample_orig}-Hist2ST.ckpt")

# Some local variable referencing error when model is inside function
##########################
# def evall(test_sample, gene_list):
    
#     train_set = list(set(list(adata_dict.keys())) - set(test_sample))
#     testset = ViT_Anndata(adata_dict = adata_dict, train_set = train_set, gene_list = gene_list,
#                 train=False,flatten=False,adj=True,ori=True,prune='NA',neighs=4, 
#             )
#     test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)

#     adata_pred, adata_truth = test(model, test_loader,'cuda')

#     adata_pred.var_names = gene_list
#     adata_truth.var_names = gene_list

#     pred_adata = adata_pred.copy()
#     test_dataset = adata_truth.copy()
    
#     test_sample = ','.join(list(test_sample))
    
#     with open(f"../../results/pf/hist2st_preds_{test_sample}_{i}.pkl", 'wb') as f:
#         pickle.dump([pred_adata,test_dataset], f)

#     for gene in pred_adata.var_names:
#         cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
#         df = df.append(pd.Series([gene, cor_val, test_sample, "Hist2ST"], 
#                              index=["Gene", "Pearson correlation", "Slide", "Method"]),
#                   ignore_index=True)

#     del model
#     torch.cuda.empty_cache()

#     df.to_csv("../../results/pf/hist2st_cor_{}_{i}.csv".format(test_sample, i))

#     with open("../../results/pf/hist2st_times.txt", 'a') as f:
#         f.write(f"{i} {test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")



# evall([i for i in list(adata_dict.keys()) if 'VLP78_A' in i], gene_list)

def test(model,test,device='cuda'):
    model=model.to(device)
    model.eval()
    preds=None
    ct=None
    gt=None
    loss=0
    adatas,adata_gts = [],[]
    with torch.no_grad():
        for patch, position, exp, adj, *_, center in tqdm(test):
            patch, position, adj = patch.to(device), position.to(device), adj.to(device).squeeze(0)
            pred = model(patch, position, adj)[0]
            preds = pred.squeeze().cpu().numpy()
            ct = center.squeeze().cpu().numpy()
            gt = exp.squeeze().cpu().numpy()
            adata = ad.AnnData(preds)
            adata.obsm['spatial'] = ct
            adata_gt = ad.AnnData(gt)
            adata_gt.obsm['spatial'] = ct
           
            adatas.append(adata)
            adata_gts.append(adata_gt)
    adata = ad.concat(adatas)
    adata_gt = ad.concat(adata_gts)
    return adata,adata_gt



test_sample = [i for i in list(adata_dict.keys()) if test_sample in i]
test_set = list(set(list(adata_dict.keys())) - set(test_sample))
testset = ViT_Anndata(adata_dict = adata_dict, train_set = test_set, gene_list = gene_list,
            train=False,flatten=False,adj=True,ori=True,prune='NA',neighs=4, 
        )
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)


adata_pred, adata_truth = test(model, test_loader,'cuda')

adata_pred.var_names = gene_list
adata_truth.var_names = gene_list

pred_adata = adata_pred.copy()
test_dataset = adata_truth.copy()

# test_sample = ','.join(list(test_sample))

# with open(f"../../results/pf_cv2/hist2st_preds_{test_sample_orig}.pkl", 'wb') as f:
#     pickle.dump([pred_adata,test_dataset], f)

for gene in pred_adata.var_names:
    pred = pred_adata.to_df().loc[:,gene]
    pred = pred.fillna(0)
    cor_val = calculate_correlation_2(pred, test_dataset.to_df().loc[:,gene])
    cor_pearson = calculate_correlation(pred, test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val,cor_pearson, test_sample_orig, "Hist2ST"], 
                         index=["Gene", "Spearman correlation", "Pearson correlation","Slide", "Method"]),
              ignore_index=True)

del model
torch.cuda.empty_cache()

df.to_csv("../../results/ffpe/hist2st_cor_{}.csv".format(test_sample_orig))

with open("../../results/ffpe/hist2st_times.txt", 'a') as f:
    f.write(f"{i} {test_sample_orig} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")


# gene_list = ["COX6C","TTLL12", "HSP90AB1", 
#            "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
# evall("CytAssist_FFPE", gene_list)

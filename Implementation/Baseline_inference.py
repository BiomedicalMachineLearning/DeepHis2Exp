import sys
import shutil
import os
import gc
import cv2
import tqdm
import torch
import random
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import anndata as ad
import argparse

# Add relative path
sys.path.append("../")

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from Models.Hist2ST.hist2st import *
from Models.STnet.stnet import *
from Models.HisToGene.histogene import *
from Models.DeepPT.deeppt import *
from Models.STimage.stimage import *
from Models.BLEEP.bleep import *
from Models.DeepSpaCE.deepspace import *
from Dataloader.Dataset_wiener import *
# from Dataloader.Dataset import *
# from Dataloader.Dataset_bunya import *

import glob
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
import gc

import os
from sklearn.metrics.pairwise import cosine_similarity
# import geopandas as gpd
# import splot
# from libpysal.weights.contiguity import Queen
# from splot.esda import moran_scatterplot, lisa_cluster
# from esda.moran import Moran, Moran_Local
# from esda.moran import Moran_BV, Moran_Local_BV
# from splot.esda import plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation
from scipy.sparse import issparse
from scipy.stats import pearsonr, spearmanr
import warnings
import tqdm
warnings.filterwarnings("ignore")

def get_R(data1,data2, dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X

    # Check if the variables are sparse matrices or numpy arrays
    adata1 = adata1.toarray() if issparse(adata1) else adata1
    adata2 = adata2.toarray() if issparse(adata2) else adata2
    
    print("Calculate Pearson and Spearman correlation score...")
    r1,p1=[],[]
    for g in range(data1.shape[1]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g], alternative='greater')
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:], alternative='greater')
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

def get_ssim(data1, data2, dim=1):
    """
    Some info about SSIM computation.
    data1: the ground truth data
    data2: the predicted data
    dim: the dimension to calculate the SSIM. If the dim = 1, calculate the SSIM at gene-wise, otherwise calculate the SSIM at spot-wise.
    """
    from skimage.metrics import structural_similarity as ssim
    
    adata1=data1.X
    adata2=data2.X

    # Check if the variables are sparse matrices or numpy arrays
    adata1 = adata1.toarray() if issparse(adata1) else adata1
    adata2 = adata2.toarray() if issparse(adata2) else adata2

    SSIM = []
    print("Calculate SSIM score...")
    for g in range(adata1.shape[dim]):
        mean_SSIM, full_SSIM = ssim(adata1[:, g], adata2[:, g], data_range=adata1.max()-adata2.min(), full=True)
        SSIM.append(mean_SSIM)
    return np.array(SSIM)

def get_MI(adata1, adata2, gene_list, spatial_matrix):
    moran_scores = []
    adata1.obsm["gpd"] = gpd.GeoDataFrame(adata1.obs, geometry=gpd.points_from_xy(spatial_matrix[:, 0], spatial_matrix[:, 1]))
    print("Calculate Moran's I score...")
    for gene in tqdm.tqdm(gene_list):
        x = adata1.to_df()[gene].values
        y = adata2.to_df()[gene].values
        w = Queen.from_dataframe(adata1.obsm["gpd"])
        moran_bv = Moran_BV(y, x, w)
        moran_scores.append(moran_bv.I)
    return moran_scores

def get_cosine(data1, data2):
    # Convert the anndata to numpy array
    adata1=data1.X.T
    adata2=data2.X.T
    # Calculate the consine similarity at gene wise
    print("Calculate Cosine similarity score...")
    cosine_sim = cosine_similarity(adata1, adata2)
    # Take the diag of similarity matrix
    cosine_score = np.diag(cosine_sim)
    return cosine_score

def make_res(dataset_name, colornorm, model_name, name):
    """
    input the dataset name, colornorm, methods, and names of the slides
    output the results of the methods with three metrics: Pearson correlation, Spearman correlation, and SSIM score
    """
    data1 = gt
    data2 = pred
    spatial_matrix = spatial_loc
    pcc, PCC_PValue = get_R(data1, data2, dim=1, func=pearsonr)
    SPC, SPC_PValue = get_R(data1, data2, dim=1, func=spearmanr)
    ssim_score = get_ssim(data1, data2)
    cosine_score = get_cosine(data1, data2)
    MI = get_MI(data1, data2, list(data2.var_names), spatial_matrix)
    df = {
    "Gene": list(data1.var_names),
    "Pearson correlation": pcc,
    "PCC_PValue": PCC_PValue,
    "Spearmanr correlation": SPC,
    "SPC_PValue": SPC_PValue,
    "SSIM_Score": ssim_score,
    "Cosine_Score": cosine_score,
    "Moran'I_Score": MI,
    "Slides": [name]*len(pcc),
    "Dataset": [dataset_name]*len(pcc),
    "Method": [model_name]*len(pcc),}
    df = pd.DataFrame(df)
    if not os.path.isdir(f"../Results/{dataset_name}"):
        os.mkdir(f"../Results/{dataset_name}")
    df.to_csv(f"../Results/{dataset_name}/{model_name}_{dataset_name}_{colornorm}_{name}_MI.csv")
    gc.collect()
    return df

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=5, help='dataset fold.')
parser.add_argument('--seed', type=int, default=42, help='random seed.')
parser.add_argument('--colornorm', type=str, default="reinhard", help='Color normalization methods.')
parser.add_argument('--dataset_name', type=str, default="Skin_Melanoma", help='Dataset choice.')
parser.add_argument('--model_name', type=str, default="hist2st", help='Model choice.')
parser.add_argument('--gene_list', type=str, default="func", help='Gene list choice.')
parser.add_argument('--HPC', type=str, default="wiener", help='HPC cluster')
args = parser.parse_args()

fold = args.fold
seed = args.seed
colornorm = args.colornorm
dataset_name = args.dataset_name
model_name = args.model_name
gene_list = args.gene_list
hpc = args.HPC

# fold = 0
# seed = 42
# colornorm = "reinhard"
# dataset_name = "Skin_Melanoma"
# model_name = "deeppt"
# gene_list = "func"

if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
    exp_norm = "log1p"
else:
    exp_norm = "lognorm"

# hpc = "vmgpu"
if hpc == "wiener":
    abs_path = "/afm03/Q2/Q2051/DeepHis2Exp/Models/Benchmarking_main"
    model_weight_path = "/afm03/Q2/Q2051/DeepHis2Exp/Model_Weights"
#     res_path = "/afm03/Q2/Q2051/DeepHis2Exp/Results"
    res_path = "/scratch/imb/uqyjia11/Yuanhao/DeepHis2Exp/Results"
    data_path = "/afm03/Q2/Q2051/DeepHis2Exp/Dataset"
elif hpc == "vmgpu":
    abs_path = "/afm01/UQ/Q2051/DeepHis2Exp/Implementation"
    model_weight_path = "/afm01/UQ/Q2051/DeepHis2Exp/Model_Weights"
    res_path = "/afm01/UQ/Q2051/DeepHis2Exp/Results"
    data_path = "/afm01/UQ/Q2051/DeepHis2Exp/Dataset"
elif hpc == "bunya":
    abs_path = "/QRISdata/Q2051/DeepHis2Exp/Implementation"
    model_weight_path = "/QRISdata/Q2051/DeepHis2Exp/Model_Weights"
    res_path = "/QRISdata/Q2051/DeepHis2Exp/Results"
    data_path = "/QRISdata/Q2051/DeepHis2Exp/Dataset"

print("Start training!")
print("Hyperparameters are as follows:")
print("Fold:", fold)
print("Color normalization method:", colornorm)
print("Dataset_name:", dataset_name)
print("Model_name:", model_name)
print("gene_list:", gene_list)
print("exp_norm:", exp_norm)
print("cluster:", hpc)


# For reproducing the results
seed_everything(seed)

tag = '5-7-2-8-4-16-32'
kernel,patch,depth1,depth2,depth3,heads,channel=map(lambda x:int(x),tag.split('-'))

# Load train and test dataset and wrap dataloader
# Train dataset is windowed WSI dataset, test dataset is WSI dataset

# Functional genes for visium dataset
target_gene_list = list(np.load(f'{data_path}/Gene_list/Gene_list_{gene_list}_{dataset_name}.npy', allow_pickle=True))

# Load sample names
if dataset_name == "Skin_Melanoma":
    if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
#         full_train_dataset = Skin_Melanoma(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=100, shuffle=False)
        test_dataset = Skin_Melanoma(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, shuffle=False)
    else:
#         full_train_dataset = Skin_Melanoma(train=True,fold=fold,r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)
        test_dataset = Skin_Melanoma(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

elif dataset_name == "BC_visium":
    if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
#         full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=8, 
#                                        exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, 
#                                        gene_list=target_gene_list, shuffle=False)
        test_dataset = BC_visium(train=False,fold=fold,r=112, neighs=8, 
                                 exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, 
                                 gene_list=target_gene_list, shuffle=False)
    else:
#         full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=8, color_norm=colornorm, 
#                                        exp_norm=exp_norm, num_subsets=15, gene_list=target_gene_list, shuffle=False)
        test_dataset = BC_visium(train=False,fold=fold,r=112, neighs=8, color_norm=colornorm, 
                                 exp_norm=exp_norm, num_subsets=15, gene_list=target_gene_list, shuffle=False)
        
elif dataset_name == "Skin_cSCC":
    if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
        test_dataset = CSCC(train=False,fold=fold,r=112, neighs=8, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, gene_list=target_gene_list, shuffle=False)
    else:
        test_dataset = CSCC(train=False,fold=fold,r=112, neighs=8, num_subsets=5, color_norm=colornorm, exp_norm=exp_norm, gene_list=target_gene_list, shuffle=False)

elif dataset_name == "BC_Her2ST":
    if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
        test_dataset = Her2st(train=False,fold=fold, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm=exp_norm, gene_list=target_gene_list, shuffle=False)
    else:
        test_dataset = Her2st(train=False,fold=fold,r=112, num_subsets=3, neighs=8, color_norm=colornorm, exp_norm=exp_norm, gene_list=target_gene_list, shuffle=False)
        
elif dataset_name == "Liver_visium":
    if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
        test_dataset = Liver_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, shuffle=False)
    else:
        test_dataset = Liver_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

elif dataset_name == "Kidney_visium":
    if model_name in ["stnet", "deeppt", "stimage", "bleep", "deepspace"]:
        test_dataset = Kidney_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, exp_norm=exp_norm, color_norm=colornorm, num_subsets=100, shuffle=False)
    else:
        test_dataset = Kidney_visium(train=False,fold=fold, r=112, neighs=8, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)


gc.collect()
    
# For real dataset only
# train_size = int(0.7 * len(full_train_dataset))
# valid_size = len(full_train_dataset) - train_size
# train_dataset, valid_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, valid_size])

if model_name == "bleep":
    if dataset_name == "Skin_Melanoma":
        full_train_dataset = Skin_Melanoma(train=True,fold=fold,r=112, neighs=1, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=100, shuffle=False)

    elif dataset_name == "BC_visium":
        full_train_dataset = BC_visium(train=True,fold=fold,r=112, neighs=1, color_norm=colornorm, 
                                        exp_norm=exp_norm, num_subsets=15, gene_list=target_gene_list, shuffle=False)
            
    elif dataset_name == "Skin_cSCC":
        full_train_dataset = CSCC(train=True,fold=fold,r=112, neighs=1, color_norm=colornorm, exp_norm=exp_norm, num_subsets=5, gene_list=target_gene_list, shuffle=False)

    elif dataset_name == "BC_Her2ST":
        full_train_dataset = Her2st(train=True,fold=fold, num_subsets=30, r=112, neighs=1, color_norm=colornorm, exp_norm=exp_norm, gene_list=target_gene_list, shuffle=False)

    elif dataset_name == "Liver_visium":
        full_train_dataset = Liver_visium(train=True,fold=fold,r=112, neighs=1, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

    elif dataset_name == "Kidney_visium":
        full_train_dataset = Kidney_visium(train=True,fold=fold,r=112, neighs=1, gene_list=target_gene_list, color_norm=colornorm, exp_norm=exp_norm, num_subsets=15, shuffle=False)

    tr_loader = DataLoader(full_train_dataset, batch_size=1, shuffle=True)
te_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define model and train
if model_name == "hist2st":
    model = Hist2ST(
        depth1=depth1, depth2=depth2,
        n_genes=len(test_dataset.gene_list), learning_rate=1e-5, label=None, 
        kernel_size=kernel, patch_size=patch, n_pos=256,
        heads=heads, channel=channel, dropout=0.2,
        zinb=0.25, nb=False,
        bake=5, lamb=0.5,)
elif model_name == "stnet":
    model = STModel(n_genes=len(test_dataset.gene_list), hidden_dim=1024, learning_rate=1e-6)
elif model_name == "histogene":
    model = HisToGene(patch_size=224, n_layers=4, n_genes=len(test_dataset.gene_list), dim=1024, 
                      learning_rate=1e-5, dropout=0.1, n_pos=256)
elif model_name == "deeppt":
    model = DeepPT(n_genes=len(test_dataset.gene_list), hidden_dim=512, learning_rate=1e-4,)
elif model_name == "stimage":
    model = STimage(n_genes=len(test_dataset.gene_list), ft=False, learning_rate=1e-5)
elif model_name == "bleep":
    model = BLEEP(n_genes=len(test_dataset.gene_list)) 
elif model_name == "deepspace":
    model = DeepSpaCE(n_genes=len(test_dataset.gene_list)) 
    
# Empty cache of GPU
torch.cuda.empty_cache()

# Create folder to save model weights
if not os.path.isdir(f"{model_weight_path}/{dataset_name}/"):
    os.mkdir(f"{model_weight_path}/{dataset_name}/")
early_stop = pl.callbacks.EarlyStopping(monitor='train_loss', mode='min', patience=30)
checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, dirpath=f"{model_weight_path}", 
                                                   filename=f"{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}", 
                                                   monitor="train_loss", mode="min")

# Define trainer for each model
if model_name == "hist2st":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         max_epochs=350, logger=True)
elif model_name == "stnet":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         max_epochs=50, logger=True)
elif model_name == "histogene":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         max_epochs=100, logger=True)
elif model_name == "deeppt":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         max_epochs=500, logger=True)
elif model_name == "stimage":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         enable_checkpointing=True,
                         max_epochs=100, logger=True)
elif model_name == "bleep":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         max_epochs=150, logger=True)   
elif model_name == "deepspace":
    trainer = pl.Trainer(accelerator='auto', 
                         callbacks=[early_stop, checkpoint_callback], 
                         max_epochs=100, logger=True)   

# Start predicting
# Modify the loading part to handle CPU-only machines
device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')

# Loading the best model
best_model_path = f"{model_weight_path}/{dataset_name}/{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.ckpt"
model.load_state_dict(torch.load(best_model_path, map_location=device))

# Inference
if model_name == "bleep":
    gc.collect()
    gt_exp, pred = bleep_inference(model, trainer, tr_loader, te_loader, method="average")
    pred = ad.AnnData(pred)
else:
    gc.collect()
    target_gene_list = test_dataset.gene_list
    out = trainer.predict(model, te_loader)
    pred = ad.AnnData(np.concatenate([out[i][0] for i in range(len(out))]))
    gt_exp = np.concatenate([out[i][1] for i in range(len(out))])
gt = sc.concat([test_dataset.meta_dict[sub_slide] for sub_slide in list(test_dataset.meta_dict.keys())])[:,target_gene_list]
gt.X = gt_exp
        
# Add the gene list to AnnData    
pred.var_names = test_dataset.gene_list
gt.var_names = test_dataset.gene_list

# Save AnnData to H5AD file
if not os.path.isdir(f"{res_path}/{dataset_name}"):
    os.mkdir(f"{res_path}/{dataset_name}")
pred.write(f"{res_path}/{dataset_name}/pred_{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.h5ad")
gt.write(f"{res_path}/{dataset_name}/gt_{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.h5ad")
gc.collect()

# Save spatial location to numpy array
spatial_loc = np.concatenate([test_dataset.meta_dict[key].obsm["spatial"] for key in list(test_dataset.meta_dict.keys())])
np.save(f'{res_path}/{dataset_name}/spatial_loc_{model_name}_{dataset_name}_{colornorm}_{test_dataset.te_names}_{gene_list}.npy', spatial_loc)
gc.collect()

# Remove the log file after all models are trained
# shutil.rmtree(f'{abs_path}/lightning_logs')
print("Finish training!")

# name = test_dataset.te_names
# df = make_res(dataset_name, colornorm, model_name, name)
# print("Finish evaluation!")

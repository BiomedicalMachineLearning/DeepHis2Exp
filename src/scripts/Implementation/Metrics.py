import glob
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import gc
import os
import argparse
import warnings
import tqdm

import splot
import geopandas as gpd
from sklearn.metrics.pairwise import cosine_similarity
from libpysal.weights.contiguity import Queen
from splot.esda import moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from esda.moran import Moran_BV, Moran_Local_BV
from splot.esda import plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation
from scipy.sparse import issparse
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="SCC_Chenhao", help='Dataset choice.')
parser.add_argument('--gene_list', type=str, default="func", help='Dataset choice.')
parser.add_argument('--fold', type=int, default=0, help='No. of slides')
parser.add_argument('--distributed', type=bool, default=False, help='Calculate the metric by slide')
args = parser.parse_args()

def get_R(data1,data2, dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X

    # Check if the variables are sparse matrices or numpy arrays
    adata1 = adata1.toarray() if issparse(adata1) else adata1
    adata2 = adata2.toarray() if issparse(adata2) else adata2
    
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

def get_ssim(data1, data2, num_breaks=None):
    """
    Some info about SSIM computation.
    data1: the ground truth data
    data2: the predicted data
    dim: the dimension to calculate the SSIM. If the dim = 1, calculate the SSIM at gene-wise, otherwise calculate the SSIM at spot-wise.
    """
    adata1 = data1.X
    adata2 = data2.X

    # Check if the variables are sparse matrices or numpy arrays
    adata1 = adata1.toarray() if issparse(adata1) else adata1
    adata2 = adata2.toarray() if issparse(adata2) else adata2

    SSIM = []
    for g in range(adata1.shape[1]):
        x = adata1[:,g]
        y = adata2[:,g]
        # Normalize the arrays if their maximum values are not zero
        x = x if np.max(x) == 0 else x / np.max(x)
        y = y if np.max(y) == 0 else y / np.max(y)  
    
        if num_breaks:
            # Discretize the normalized arrays into 'num_breaks' bins
            x = np.digitize(x, np.linspace(0, 1, num_breaks), right=False) - 1
            y = np.digitize(y, np.linspace(0, 1, num_breaks), right=False) - 1
            
            # Constants used in SSIM calculation
            C1 = (0.01 * (num_breaks - 1))**2
            C2 = (0.03 * (num_breaks - 1))**2
        else:
            C1 = (0.01)**2
            C2 = (0.03)**2
        
        mux = np.mean(x)
        muy = np.mean(y) 
        sigxy = np.cov(x, y)[0, 1]
        sigx = np.var(x)
        sigy = np.var(y)
        
        ssim = ((2 * mux * muy + C1) * (2 * sigxy + C2)) / ((mux**2 + muy**2 + C1) * (sigx + sigy + C2))
        SSIM.append(ssim)
        assert -1 <= ssim <= 1, "SSIM should be within the valid range [-1, 1]"
        
    return SSIM

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
    cosine_sim = cosine_similarity(adata1, adata2)
    # Take the diag of similarity matrix
    cosine_score = np.diag(cosine_sim)
    return cosine_score

def top_predictable_genes(df_all, dataset, method, num=5,):
    """
    input the results from the make_res function.
    output the top predictable genes with the number of positive Pearson correlation values.
    num is the number of top predictable genes.
    """
    df = df_all[df_all["Method"]==method]
    if num == "pos":
        top5_df = df[["Gene", "Pearson correlation"]].groupby("Gene", as_index=False).agg(['median', 'mean', 'max', 'min', 'std']).sort_values(by=('Pearson correlation', 'median'), ascending=False)
    else:
        top5_df = df[["Gene", "Pearson correlation"]].groupby("Gene", as_index=False).agg(['median', 'mean', 'max', 'min', 'std']).sort_values(by=('Pearson correlation', 'median'), ascending=False).head(num)
    top5_genes = list(top5_df.index)
    
    # Subset the results according to top predictable gene
    num_pos = []
    for g in top5_genes:
        subset_df = df[df["Gene"]==g]
        count_positive_corr = subset_df[subset_df['Pearson correlation'] > 0].shape[0]
        # print(f'Number of positive Pearson correlation values: {count_positive_corr}')
        num_pos.append(int(count_positive_corr))
    top5_df["Number of consistent samples"] = num_pos
    top5_df["Method"] = method
    top5_df["Dataset"] = dataset
    top5_df = top5_df[[(           'Method',       ''),
            (                     'Dataset',       ''),
            ('Number of consistent samples',       ''),
            (         'Pearson correlation',   'mean'),
            (         'Pearson correlation', 'median'),
            (         'Pearson correlation', 'min'),
            (         'Pearson correlation',    'max'),
            (         'Pearson correlation',    'std'),
            
            ]]

    return top5_df

def make_res(dataset_name, colornorm, Methods, names, distributed, gene_list):
    """
    input the dataset name, colornorm, methods, and names of the slides
    output the results of the methods with three metrics: Pearson correlation, Spearman correlation, and SSIM score
    """
    for method in tqdm.tqdm(Methods):
        if distributed:
            gc.collect()
            name = names[fold]
            file_path = f"../Results/{dataset_name}/gt_{method}_{dataset_name}_{colornorm}_{name}_{gene_list}.h5ad"
            if os.path.exists(file_path):
                data1 = sc.read_h5ad(f"../Results/{dataset_name}/gt_{method}_{dataset_name}_{colornorm}_{name}_{gene_list}.h5ad")
                data2 = sc.read_h5ad(f"../Results/{dataset_name}/pred_{method}_{dataset_name}_{colornorm}_{name}_{gene_list}.h5ad")
                spatial_matrix = np.load(f"../Results/{dataset_name}/spatial_loc_{method}_{dataset_name}_reinhard_{name}_{gene_list}.npy")
                pcc, PCC_PValue = get_R(data1, data2, dim=1, func=pearsonr)
                SPC, SPC_PValue = get_R(data1, data2, dim=1, func=spearmanr)
                ssim_score = get_ssim(data1, data2)
                cosine_score = get_cosine(data1, data2)
                data1.var_names = data2.var_names
                MI = get_MI(data1, data2, list(data1.var_names), spatial_matrix)
                PCC_BC_Visium = {
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
                "Method": [method]*len(pcc),}
                
                PCC_BC_Visium = pd.DataFrame(PCC_BC_Visium)
                if not os.path.isdir(f"../Results/{dataset_name}"):
                    os.mkdir(f"../Results/{dataset_name}")
                PCC_BC_Visium.to_csv(f"../Results/{dataset_name}/{method}_{dataset_name}_{colornorm}_{name}_MI_{gene_list}.csv")
            else:
                print(f"The file {file_path} does not exist. Skipping.")
    print("Organize the results into summary file!")
    res = glob.glob(f"../Results/{dataset_name}/*_MI_{gene_list}.csv")
    df = pd.concat([pd.read_csv(i, index_col=[0]) for i in res])
    df.reset_index(inplace=True)
    df.to_csv(f"../Results/Summary/{dataset_name}_summary_MI_{gene_list}.csv")
    return df

# Read the method names
dataset_name = args.dataset_name
colornorm = "reinhard"
Methods = ["deeppt", "hist2st", "histogene", "stimage", "stnet", "deepspace", "bleep"]
if dataset_name == "Kidney_visium":
    names = ["A", "B", "C", "D", "Visium14_C", "Visium14_D"]
elif dataset_name == "BC_Her2ST":
    names = list(set([i.split("_")[-2] for i in glob.glob(f"../Results/{dataset_name}/gt_stimage_{dataset_name}_reinhard*h5ad")]))
elif dataset_name == "BC_visium":
    names = ["1142243F", "1160920F", "CID4290", "CID4465", "CID44971", "CID4535", "FFPE", "block1", "block2"]
elif dataset_name == "Skin_Melanoma":
    names = ["Visium29_B1", "Visium29_C1", "Visium37_D1", "Visium38_B1", "Visium38_D1"]
elif dataset_name == "Skin_cSCC":
    names = ['P2_ST_rep1', 'P2_ST_rep2', 'P2_ST_rep3', 'P5_ST_rep1', 
         'P5_ST_rep2', 'P5_ST_rep3', 'P9_ST_rep1', 'P9_ST_rep2', 
         'P9_ST_rep3', 'P10_ST_rep1', 'P10_ST_rep2', 'P10_ST_rep3']
elif dataset_name == "Liver_visium":
    names = ["C73_A1_VISIUM", "C73_B1_VISIUM", "C73_C1_VISIUM", "C73_D1_VISIUM"]

names.sort()
fold = args.fold
distributed = args.distributed
gene_list = args.gene_list
print(f"Sample names:{names}")
print(f"Sample name:{names[fold]}")
print(f"fold:{fold}")
print(f"target gene list:{gene_list}")

# Organize the results into dataframe
df = make_res(dataset_name, colornorm, Methods, names, distributed, gene_list)

import gc
import os
import glob
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
import geopandas as gpd
import argparse
import splot
import warnings
import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from libpysal.weights.contiguity import Queen
from splot.esda import moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from esda.moran import Moran_BV, Moran_Local_BV
from splot.esda import plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default="SCC_Chenhao", help='Dataset choice.')
args = parser.parse_args()

def get_R(data1,data2, dim=1,func=pearsonr):
    from scipy.sparse import issparse
    adata1=data1.X
    adata2=data2.X

    # Check if the variables are sparse matrices or numpy arrays
    adata2 = adata2.toarray() if issparse(adata2) else adata2
    
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g], alternative='greater')
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:], alternative='greater')
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

def get_ssim(data1, data2):
    from skimage.metrics import structural_similarity as ssim
    adata1=data1.X
    adata2=data2.X
    ssim_score = ssim(adata1, adata2, data_range=adata1.max() - adata2.min())
    return ssim_score

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

def make_res(dataset_name, colornorm, Methods, names):
    """
    input the dataset name, colornorm, methods, and names of the slides
    output the results of the methods with three metrics: Pearson correlation, Spearman correlation, and SSIM score
    """
    for method in tqdm.tqdm(Methods):
        for i in range(len(names)):
            gc.collect()
            name = names[i]
            file_path = f"../Results/{dataset_name}/gt_{method}_{dataset_name}_{colornorm}_{name}_func.h5ad"
            if os.path.exists(file_path):
                data1 = sc.read_h5ad(f"../Results/{dataset_name}/gt_{method}_{dataset_name}_{colornorm}_{name}_func.h5ad")
                data2 = sc.read_h5ad(f"../Results/{dataset_name}/pred_{method}_{dataset_name}_{colornorm}_{name}_func.h5ad")
                spatial_matrix = np.load(f"../Results/Kidney_visium/spatial_loc_{method}_{dataset_name}_reinhard_{name}_func.npy")
                pcc, PCC_PValue = get_R(data1, data2, dim=1, func=pearsonr)
                SPC, SPC_PValue = get_R(data1, data2, dim=1, func=spearmanr)
                ssim_score = get_ssim(data1, data2)
                cosine_score = get_cosine(data1, data2)
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
                PCC_BC_Visium.to_csv(f"../Results/{dataset_name}/{method}_{dataset_name}_{colornorm}_{name}.csv")
            else:
                print(f"The file {file_path} does not exist. Skipping.")
    print("Organize the results into summary file!")
    res = glob.glob(f"../Results/{dataset_name}/*csv")
    df = pd.concat([pd.read_csv(i, index_col=[0]) for i in res])
    df.reset_index(inplace=True)
    df.to_csv(f"../Results/Summary/{dataset_name}_summary.csv")
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
    names = list(set([i.split("/")[3].split("_")[5] for i in glob.glob(f"../Results/{dataset_name}/gt_stimage_{dataset_name}_reinhard*h5ad")]))
elif dataset_name == "Skin_Melanoma":
    names = list(set([i.split("_")[-3]+"_"+i.split("_")[-2] for i in glob.glob(f"../Results/{dataset_name}/gt_deepspace_{dataset_name}_reinhard*h5ad")]))
elif dataset_name == "Skin_cSCC":
    names = list(set([i.split("/")[3].split("_")[5] + "_" + i.split("/")[3].split("_")[6] + "_" + i.split("/")[3].split("_")[7] for i in glob.glob(f"../Results/{dataset_name}/gt_deepspace_{dataset_name}_reinhard*h5ad")]))
elif dataset_name == "Liver_visium":
    names = list(set([i.split("/")[3].split("_")[5]+'_'+i.split("/")[3].split("_")[6]+'_'+i.split("/")[3].split("_")[7] for i in glob.glob(f"../Results/{dataset_name}/gt_deepspace_{dataset_name}_reinhard*h5ad")]))
    
names.sort()
print(f"Sample names:{names}")

# Organize the results into dataframe
df = make_res(dataset_name, colornorm, Methods, names)

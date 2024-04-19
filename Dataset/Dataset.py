import os
import gc
import cv2
import tqdm
import dgl
import glob
import torchvision
import numpy as np
import pandas as pd
import scprep as scp
import warnings
import pandas as pd
import torch
import torch.nn as nn
import anndata as ad
import json
import seaborn as sns
import scanpy as sc
import squidpy as sq
import PIL.Image as Image
import hdf5plugin
import torchstain

from torch_geometric import utils
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from torchvision import transforms
from squidpy.im import ImageContainer
from collections import defaultdict as dfd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix, minkowski_distance, distance
from typing import Union, Dict, Optional, Tuple, BinaryIO
from pathlib import Path, PurePath
from scanpy import read_visium, read_10x_mtx
from matplotlib.image import imread
from collections import defaultdict as dfd
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import minmax_scale


# ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")
gpu = "vmgpu"
if gpu =="wiener":
    abs_path = "/afm03/Q2/Q2051"
elif gpu == "vmgpu":
    abs_path = "/afm01/UQ/Q2051"
elif gpu == "bunya":
    abs_path = "/QRISdata/Q2051"
print("Dataset path:", abs_path)

# Build graph using pathology annotations (PAG)
def pathology_annotation_graph(labels, alpha=1e-8):
    """
    Input: List of cell types(pathology annotations)
    Output: Adjancent matrix
    Rule: Same category or not (1 or a near 0 value)
    """
    import numpy as np
    import torch

    # Initialize an empty graph as a 2D list filled with zeros
    graph = [[alpha for _ in range(len(labels))] for _ in range(len(labels))]

    # Iterate through each pair of labels
    for i in range(len(labels)):
      for j in range(i + 1, len(labels)):
          # Check if the labels are the same
          if labels[i] == labels[j]:
              graph[i][j] = 1
              graph[j][i] = 1
    graph = torch.tensor(graph)
    return graph

def window_adata(adata_dict, sizes, img_dict, threshold=100):
    """
    Window every element in adata_dict.
    
    adata_dict is a dictionary to store the adata, key is the name of slide, value is the adata object.
    sizes is the crop size you want to use during tiling.
    img_dict is a dictionary to store the H&E image data, the key is the name of slide, value is image numpy array.
    Threshold is to filter the tiles with small number of cells.
    """
    adata_sub_dict, img_sub_dict, img_all_dict = {}, {}, {}

    for name, adata in adata_dict.items():
        adata.obs_names_make_unique()
        img = ImageContainer(img_dict[name])
        # split image into smaller crops
        crops = img.generate_equal_crops(size=sizes) 
        crops_img = list(img.generate_equal_crops(size=sizes, as_array="image", squeeze=True))  
        print(f"windowing {name}, Croped into {len(crops_img)} tiles")
        summ, available = 0, 0
        num_obs = []
        img_sub_sub_dict = {}
        for i, crop in enumerate(crops):
            adata_crop = crop.subset(adata, spatial_key="spatial")
            summ += adata_crop.n_obs
            img_all_dict[name+"__"+str(i)] = crops_img[i]
            if adata_crop.n_obs > threshold:
                available += adata_crop.n_obs
                namee = name+"__"+str(i)
                adata_sub_dict[namee] = adata_crop
                img_sub_dict[namee] = crops_img[i]
                img_sub_sub_dict[namee] = crops_img[i]
                num_obs.append(adata_crop.n_obs)
        n_min = np.array(num_obs).min()
        n_max = np.array(num_obs).max()
        print(f"There are {len(img_sub_sub_dict.keys())} tiles available. Min_spots:{n_min}, Max_spots:{n_max}")
        del img_sub_sub_dict
    print(f"Overall, there are {len(img_sub_dict.keys())} tiles and {sum([adata_sub_dict[k].n_obs for k in list(adata_sub_dict.keys())])} spots available.")
    return adata_sub_dict, img_sub_dict, img_all_dict
    

def subset_adatas(adata_dict, num_subsets=4, shuffle=True):
    """
    Subset the AnnData object into several subsets.
    adata_dict: dictionary, key is the name of slide, value is the adata object.
    num_subsets: int, number of subsets you want to split.
    """
    import numpy as np

    subset_adatas = {}
    for name, adata in adata_dict.items():
        # Get the number of observations in the full AnnData
        num_observations = adata.shape[0]

        # Set the size of each subset
        subset_size = num_observations // num_subsets
        print("subset_size:", subset_size)

        # Create an array of indices for random sampling without replacement
        random_indices = np.random.choice(num_observations, num_observations, replace=False)

        # Iterate through the subsets and create new AnnData objects
        for i in range(num_subsets):
            # Extract indices for the current subset
            start_index = i * subset_size
            end_index = (i + 1) * subset_size
            if shuffle:
                subset_indices = random_indices[start_index:end_index]
            else:
                subset_indices = np.arange(start_index, end_index)
            # Create the subset AnnData
            subset_anndata = adata[subset_indices, :].copy()

            # Append the subset AnnData to the list
            subset_adatas[f"{name}-{i}"] = subset_anndata
    return subset_adatas

# Color normalization
target = cv2.cvtColor(cv2.imread(f"{abs_path}/DeepHis2Exp/Dataset/Reference_Normalization/ref_HE.png"), cv2.COLOR_BGR2RGB)
def color_normalization(img, target, method="macenko"):
    """
    img: numpy array, RGB image
    method: str, "raw", "macenko" or "reinhard"
    target: numpy array, RGB image
    """

    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Lambda(lambda x: x*255)
    ])
    if method == "macenko":
        Normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        Normalizer.fit(T(target))
        img, Macenko_H, Macenko_E = Normalizer.normalize(I=T(img), stains=True)
    elif method == "reinhard":
        Normalizer = torchstain.normalizers.ReinhardNormalizer(backend='torch')
        Normalizer.fit(T(target))
        img = Normalizer.normalize(I=T(img))
    elif method == "raw":
        img = img
    return img

class WeightedGraph_Anndata(torch.utils.data.Dataset):
    """
    Some information for customized weighted graph construction.
    """
    def __init__(self, fold=0, gene_list=None, num_subsets=15,
                 train=True, r=112, exp_norm='lognorm', SLG=False, PAG=False, HSG=True,
                 neighs=8, color_norm="reinhard", target=target, distance_mode="distance",
                 ):
        super(WeightedGraph_Anndata, self).__init__()
        """
        patch, center, exp, adj, oris, sfs, edge_index, edge_weight
        """
        self.train = train
        self.target = target
        self.r = r
        self.exp_norm = exp_norm
        self.SLG = SLG
        self.PAG = PAG
        self.HSG = HSG
        self.train = train
        self.target = target
        self.data_dir = f'{abs_path}/DeepHis2Exp/Dataset/BC_visium/'
        names = ['1142243F', '1160920F', 'CID4290', 'CID4465', 'CID44971', 'CID4535', ] # 'FFPE', 'block1', 'block2'
        names.sort()

        samples = names
        print(f"Datasize:{len(samples)}")
        
        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))

        self.names = tr_names if train else te_names
        self.names.sort()
        
        print('Loading whole slide imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))).permute(1,0,2) for i in tqdm.tqdm(self.names)}

        adata_dict = {}
        path = self.data_dir
        for name in tqdm.tqdm(self.names):
            if name in ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]:
                adata_dict[name] = sc.read_h5ad(f"{path}{name}.h5ad")
            elif name in ["block1", "block2", "FFPE"]:
                adata_dict[name] = sc.read_h5ad(f"{path}{name}.h5ad")
        self.adata_dict = adata_dict
        self.meta_dict = subset_adatas(adata_dict, num_subsets=num_subsets)
        self.names = self.meta_dict.keys()
        self.id2name = dict(enumerate(self.names))
    
        print('Loading imgs...')
        self.patch_dict=dfd(lambda :None)
        
        print('Loading spatial coordinates...')
        self.center_dict = {
            i:np.floor(m.obsm["spatial"].astype(np.int64)).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_row', 'array_col']].values for i,m in self.meta_dict.items()}
        
        print('Loading gene expression')
        self.gene_set = gene_list
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)
        
        # For ZINB distribution
        self.ori_dict = {
            i:m.to_df()[self.gene_set].values.astype(np.float64) 
            for i,m in self.meta_dict.items()
        }
        self.counts_dict = {
            i: (m.sum(1)) / np.median(m.sum(1)) for i,m in self.ori_dict.items()
        }

        # Color normalization
        self.color_norm = color_norm
        print('Loading imgs...')
        self.patch_dict=dfd(lambda :None)
        for ID in tqdm.tqdm(self.names):
            im = self.img_dict[ID.split("-")[0]]
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                # Color normalization
                patch = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(patch)
                patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        del self.img_dict

        # if self.train:
        self.adj_dict = {
                i: torch.zeros(m.shape[0], m.shape[0])
                for i, m in self.loc_dict.items()
            }
        if self.HSG:
            print('Loading histogical image similarity graph...')
            self.HSG_dict = {
                i: torch.ones(m.shape[0], m.shape[0]) - torch.tensor(kneighbors_graph(m.reshape(m.shape[0], -1).numpy(), n_neighbors=m.shape[0]-1, mode="distance", metric="cosine", n_jobs=-1).toarray())
                for i,m in self.patch_dict.items()
            }
        if self.SLG:
            print('Calculating spatial distance matrices...')
            self.SLG_dict = {
                i: torch.tensor(kneighbors_graph(m.reshape(m.shape[0], -1), n_neighbors=neighs, mode="connectivity", metric="euclidean", n_jobs=-1).toarray())
                for i, m in self.loc_dict.items()
            }
        if self.PAG:
            print('Loading pathology annotations...')
            self.cls_dict = {}
            for i,m in self.meta_dict.items():
                if 'Classification' in m.obs.columns:
                    self.cls_dict[i] = m.obs["Classification"].values
                else:
                    self.cls_dict[i] = ["Unknown"]*len(m.obs_names)
            print('Loading pathology annotation graph...')
            self.PAG_dict = {
                i: pathology_annotation_graph(m, alpha=0)
                for i,m in self.cls_dict.items()
            }
        
        # Create a DGL graph from the adjacency matrix
        self.edge_index_dict, self.edge_weights_dict = {}, {}
        for i in self.names:
            weights_graph = self.adj_dict[i]
            
            # Construct weighted graph
            if self.SLG:
                weights_graph = weights_graph+self.SLG_dict[i]

            if self.PAG:
                PAG = self.PAG_dict[i]
                weights_graph = weights_graph+PAG # sum of all distance graphs

            if self.HSG:
                HSG = self.HSG_dict[i]
                weights_graph = weights_graph+HSG
            self.adj_dict[i] = weights_graph
            # Assign weights to edges
            self.edge_index_dict[i], self.edge_weights_dict[i] = from_scipy_sparse_matrix(csr_matrix(weights_graph))

    def __getitem__(self, index):
        ID = self.id2name[index]
        exps = torch.Tensor(self.exp_dict[ID]) #  Spot-level gene expression
        oris = torch.Tensor(self.ori_dict[ID])
        sfs = torch.Tensor(self.counts_dict[ID])
        centers = self.center_dict[ID] # Spatial coordinates for tiling and position embedding
        loc = self.loc_dict[ID]
        positions = torch.LongTensor(loc)
        adj = self.adj_dict[ID].to(torch.float32) # Adjacency matrix construct from spatial coordinates
        patches = self.patch_dict[ID] # Image patches

        edge_index = self.edge_index_dict[ID]
        edge_weights = self.edge_weights_dict[ID]
        data = [patches, positions, exps, adj, torch.Tensor(oris),torch.Tensor(sfs), edge_index, edge_weights]
        return data
        
    def __len__(self):
        return len(self.exp_dict)
    
    def get_img(self, name):
        path = f"{self.data_dir}/{name}/image.tif"
        im = Image.open(path).convert('RGB')
        return im

        
# Breast cancer Alex & 10X dataset
class BC_visium(torch.utils.data.Dataset):
    """
    Some Information about BC_visium
    patch, center, exp, adj, oris, sfs = batch
    """
    def __init__(self, train=True, fold=0, gene_list=None, num_subsets=3,
                r=112, exp_norm='lognorm', GAG=False, PAG=False, flatten=False,
                 neighs=8, color_norm="reinhard", target=target, shuffle=True,
                 ):
        super(BC_visium, self).__init__()
     
        self.data_dir = f'{abs_path}/DeepHis2Exp/Dataset/BC_visium/'
        self.r = r
        self.gene_list = gene_list
        
        names = ['1142243F', '1160920F', 'CID4290', 'CID4465', 'CID44971', 'CID4535', 'FFPE', 'block1', 'block2']
        names.sort()
        self.train = train
        self.target = target
        self.flatten=flatten
        self.color_norm = color_norm
        self.GAG = GAG
        self.PAG = PAG
        
        samples = names
        print(f"Datasize:{len(samples)}")
        
        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))
        
        if train:
            self.names = tr_names
        else:
            self.names = te_names
            
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        
        print('Loading metadata...')
        # Read the adata from the path dictionary and create a dictionary to save the adata.
        path = self.data_dir
        adata_dict = {}
        for name in tqdm.tqdm(self.names):
            if name in ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]:
#                 adata_dict[name] = read_visium_Alex(f"{path}/{name}", library_id = name, source_image_path = f"{path}/{name}/image.tif")
                adata_dict[name] = sc.read_h5ad(f"{path}{name}.h5ad")
            elif name in ["block1", "block2", "FFPE"]:
#                 adata_dict[name] = read_visium_10x(f"{path}/{name}", library_id = name, source_image_path = f"{path}/{name}/image.tif")
                adata_dict[name] = sc.read_h5ad(f"{path}{name}.h5ad")
            
        self.meta_dict = subset_adatas(adata_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = gene_list
        
        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)
        
        # For ZINB distribution
        self.ori_dict = {
            i:m.to_df()[self.gene_set].values.astype(np.float64) 
            for i,m in self.meta_dict.items()
        }
        self.counts_dict = {
            i: (m.sum(1)) / np.median(m.sum(1)) for i,m in self.ori_dict.items()
        }    
        
        print('Loading spatial coordinates...')
        self.center_dict = {
            i:np.floor(m.obsm["spatial"].astype(np.int64)).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_row', 'array_col']].values for i,m in self.meta_dict.items()}         
            
        self.patch_dict=dfd(lambda :None)
      
        self.te_names = te_names[0].split("-")[0]
        print("Eval set: ", self.te_names)
        self.id2name = dict(enumerate(self.names))
        self.adata_dict = {k: v for k, v in adata_dict.items() if k in self.names}
    
        print('Loading image patches...')
        self.patch_dict=dfd(lambda :None)
        for ID in tqdm.tqdm(self.names):
#             im = self.img_dict[ID.split("-")[0]].permute(1,0,2)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2*self.r, 2*self.r))
            for i in range(n_patches
                          ):
                x, y = centers[i]
                patch = self.img_dict[ID.split("-")[0]].permute(1,0,2)[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                patch = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(patch)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        del self.img_dict
        print('Calculating adjacency matrices or distance matrices...')
        self.SLG_dict = {
            i: features_construct_graph(m, k=neighs, mode="connectivity", metric="euclidean", n_jobs=-1)
            for i,m in self.loc_dict.items()
        }

        if self.train:
            self.adj_dict = self.SLG_dict
            # Weighted graph is only used in training, in testing we use SLG only
            # Because we don't have the ground truth cell type annotation and gene expression during the testing
            if self.GAG:
                print('Loading gene association graph...')
                self.GAG_dict = {
                    i: features_construct_graph(m, k=0, pca=None, mode="distance", metric="cosine", n_jobs=-1)
                    # for i,m in self.exp_dict.items()
                    for i,m in self.exp_dict.items()
                }
                self.adj_dict = {
                    # i: distance2adj(self.SLG_dict[i] * self.GAG_dict[i])
                    i: self.adj_dict[i] * self.GAG_dict[i]
                    for i in self.names
                }
            if self.PAG:
                print('Loading pathology annotations...')
                self.cls_dict = {}
                for i,m in self.adata_dict.items():
                    if 'Classification' in m.obs.columns:
                        self.cls_dict[i] = m.obs["Classification"].values
                    else:
                        self.cls_dict[i] = ["Unknown"]*len(m.obs_names)
                print('Loading pathology annotation graph...')
                self.PAG_dict = {
                    i: pathology_annotation_graph(m, alpha=0)
                    for i,m in self.cls_dict.items()
                }
                self.adj_dict = {
                    # i: distance2adj(self.SLG_dict[i] * self.GAG_dict[i] * self.PAG_dict[i])
                    i: self.adj_dict[i] * self.PAG_dict[i]
                    for i in self.names }
        else:
            self.adj_dict = self.SLG_dict
        

    def __getitem__(self, index):
        ID = self.id2name[index]
        exps = torch.Tensor(self.exp_dict[ID]) #  Spot-level gene expression
        oris = torch.Tensor(self.ori_dict[ID])
        sfs = torch.Tensor(self.counts_dict[ID])
        centers = self.center_dict[ID] # Spatial coordinates for tiling and position embedding
        loc = self.loc_dict[ID]
        positions = torch.LongTensor(loc)
        adj = self.adj_dict[ID].to(torch.float32) # Adjacency matrix construct from spatial coordinates
        patches = self.patch_dict[ID] # Image patches
        # cls_lbl = self.cls_dict[ID] # Cell type annotation
        data = [patches, positions, exps, adj, oris, sfs]
        return data
        
    def __len__(self):
        return len(self.exp_dict)
    
    def get_img(self, name):
        path = f"{self.data_dir}/{name}/image.tif"
        im = Image.open(path).convert('RGB')
        return im
    
# Her2st breast cancer dataset
class Her2st(torch.utils.data.Dataset):
    """
    Some Information about HER2ST
    patch, center, exp, adj, oris, sfs = batch
    This code is modified from Hist2ST github repository.
    """
    def __init__(self, train=True, fold=0, r=112, flatten=False, ori=True, adj=True,
                 neighs=8, color_norm="reinhard", exp_norm="log1p",
                 num_subsets=3, target=target, shuffle=True,
                 gene_list=None):
        super(Her2st, self).__init__()
        
        self.cnt_dir = f'{abs_path}/DeepHis2Exp/Dataset/her2st/data/ST-cnts'
        self.img_dir = f'{abs_path}/DeepHis2Exp/Dataset/her2st/data/ST-imgs'
        self.pos_dir = f'{abs_path}/DeepHis2Exp/Dataset/her2st/data/ST-spotfiles'
        self.lbl_dir = f'{abs_path}/DeepHis2Exp/Dataset/her2st/data/ST-pat/lbl'
        self.r = r

        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        samples = names
        print(f"Datasize:{len(samples)}")

        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        
        print('Loading metadata...')
        self.adata_dict = {i:self.get_meta(i) for i in tqdm.tqdm(self.names)}
        self.meta_dict = {}
        for name in self.names:
            cnt_mtx = ad.AnnData(self.adata_dict[name].iloc[:,:-7])
            cnt_mtx.obs = self.adata_dict[name].iloc[:,-7:]
            cnt_mtx.obsm["spatial"] = cnt_mtx.obs[['pixel_x','pixel_y']].values
            cnt_mtx.var_names = list(self.adata_dict[name].iloc[:,:-7].columns)
            self.meta_dict[name] = cnt_mtx
        self.meta_dict = subset_adatas(self.meta_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = list(self.meta_dict.keys())
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':6
        }
        
        for i in self.names:
            idx=self.meta_dict[i]
            if i in ['A1','B1','C1','D1','E1','F1','G2','H1']:
                lbl=self.get_lbl(i)
                lbl=lbl.loc[idx,:]['label'].values
                lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                self.label[i]=lbl
            else:
                self.label[i]=torch.full((len(idx),),-1)
                    
        self.gene_set = list(gene_list)
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)

        if self.ori:
            self.ori_dict = {i:m.to_df()[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m.obs[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i: torch.Tensor(kneighbors_graph(m, n_neighbors=neighs, metric="euclidean", mode='connectivity', include_self=True).toarray())
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        self.color_norm = color_norm
        self.target = target
        
        print("Croping WSI into patches...")
        for ID in self.names:
            im = self.img_dict[ID.split("-")[0]].permute(1,0,2)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                x, y = centers[i]
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                normalized_img = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(normalized_img)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)

            self.patch_dict[ID]=patches
        
        
    def __getitem__(self, index):
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        label=self.label[ID]
        exps = torch.Tensor(exps)
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.meta_dict)

    def get_img(self,name):
        pre = self.img_dir+'/'+name[0]+'/'+name
        fig_name = os.listdir(pre)[0]
        path = pre+'/'+fig_name
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = self.cnt_dir+'/'+name+'.tsv'
        df = pd.read_csv(path,sep='\t',index_col=0)

        return df

    def get_pos(self,name):
        path = self.pos_dir+'/'+name+'_selection.tsv'
        # path = self.pos_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name, gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join((pos.set_index('id')))

        return meta

    def get_lbl(self,name):
        # path = self.pos_dir+'/'+name+'_selection.tsv'
        path = self.lbl_dir+'/'+name+'_labeled_coordinates.tsv'
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id
        df.drop('pixel_x', inplace=True, axis=1)
        df.drop('pixel_y', inplace=True, axis=1)
        df.drop('x', inplace=True, axis=1)
        df.drop('y', inplace=True, axis=1)
        df.set_index('id',inplace=True)
        return df


# CSCC skin cancer dataset
class CSCC(torch.utils.data.Dataset):
    """
    Some information about cSCC
    """
    def __init__(self,train=True,r=224, exp_norm="log1p", fold=0,
                 flatten=False, ori=True, adj=True, num_subsets=3,
                 color_norm="reinhard", prune='grid', neighs=8, shuffle=True,
                 gene_list=None):
        super(CSCC, self).__init__()

        self.dir = f'{abs_path}/DeepHis2Exp/Dataset/CSCC/'
        self.r = r

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)
        
        self.ori = ori
        self.adj = adj
        self.train = train
        self.flatten = flatten
        self.gene_list = gene_list
        samples = names
        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print(te_names)
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        print('Loading metadata...')
        self.adata_dict = {i:self.get_meta(i) for i in tqdm.tqdm(self.names)}
        meta_dict = {}
        for name in self.names:
            cnt_mtx = ad.AnnData(self.adata_dict[name].iloc[:,:-6])
            cnt_mtx.obs = self.adata_dict[name].iloc[:,-6:]
            cnt_mtx.var_names = list(self.adata_dict[name].iloc[:,:-6].columns)
            cnt_mtx.obsm["spatial"] = cnt_mtx.obs[['pixel_x','pixel_y']].values
            cnt_mtx.var_names = self.get_var_names(name)
            meta_dict[name] = cnt_mtx
        self.meta_dict = subset_adatas(meta_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = list(gene_list)

        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                # i:scp.transform.log(m.to_df().values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)
        
        
        if self.ori:
            # self.ori_dict = {i:m.to_df().values for i,m in self.meta_dict.items()}
            self.ori_dict = {i:m.to_df()[self.gene_set].values for i,m in self.meta_dict.items()}

            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m.obs[['pixel_x','pixel_y']].values).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i: torch.Tensor(kneighbors_graph(m, n_neighbors=neighs, metric="euclidean", mode='connectivity', include_self=True).toarray())
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.color_norm = color_norm
        self.target = target
        
        print("Croping WSI into patches...")
        for ID in self.names:
            im = self.img_dict[ID.split("-")[0]].permute(1,0,2)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            
            for i in range(n_patches):
                x, y = centers[i]
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                normalized_img = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(normalized_img)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)

            self.patch_dict[ID]=patches

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        adj=self.adj_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im

    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df
    
    def get_var_names(self, name):
        cnt = self.get_cnt(name)
        gene_list = list(cnt.columns)
        
        return gene_list
    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

    def get_overlap(self,meta_dict,gene_list):
        gene_set = set(gene_list)
        for i in meta_dict.values():
            gene_set = gene_set&set(i.columns)
        return list(gene_set)

class Liver_visium(torch.utils.data.Dataset):
    """
    Some Information about Liver_visium
    patch, center, exp, adj, oris, sfs = batch
    """
    def __init__(self, train=True, fold=0,r=112,flatten=False,ori=True,adj=True,
                 prune='Grid',neighs=8, color_norm="reinhard", 
                 num_subsets=3, target=target, exp_norm="log1p", shuffle=True,
                 gene_list=None):
        super(Liver_visium, self).__init__()
        
        self.data_dir = f'{abs_path}/DeepHis2Exp/Dataset/Liver_visium/GSE240429'
        self.r = r

        self.gene_list = gene_list
        names = [name.split('/')[-1].split(".")[0] for name in glob.glob(self.data_dir+'/*h5ad')]        
        names.sort()
        self.train = train
        self.ori = ori
        self.adj = adj
        samples = names
        print(f"Datasize:{len(samples)}")

        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        
        print('Loading metadata...')
        adata_dict = {name: sc.read_h5ad(f"{self.data_dir}/{name}.h5ad") for name in tqdm.tqdm(self.names)}
        
        
        for i, m in adata_dict.items():
            adata_dict[i].var_names_make_unique() 
        self.meta_dict = subset_adatas(adata_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = list(gene_list)

        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                # i:scp.transform.log(m.to_df().values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)

        if self.ori:
            self.ori_dict = {i:m.to_df()[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m.obsm["spatial"]).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_row','array_col']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i: torch.Tensor(kneighbors_graph(m, n_neighbors=neighs, metric="euclidean", mode='connectivity', include_self=True).toarray())
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        self.color_norm = color_norm
        self.target = target
        
        print("Croping WSI into patches...")
        for ID in self.names:
            im = self.img_dict[ID.split("-")[0]].permute(1,0,2)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))

            for i in range(n_patches):
                x, y = centers[i]
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                normalized_img = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(normalized_img)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)

            self.patch_dict[ID]=patches
        
        
    def __getitem__(self, index):
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.meta_dict)
    
    def get_img(self, name):
        path = f"{self.data_dir}/{name}.tiff"
        im = Image.open(path).convert('RGB')
        return im
    
class Kidney_visium(torch.utils.data.Dataset):
    """
    Some Information about kidney_visium
    patch, center, exp, adj, oris, sfs = batch
    """
    def __init__(self, train=True, fold=0, r=112, flatten=False, ori=True, adj=True,
                 prune='Grid',neighs=8, color_norm="reinhard", 
                 num_subsets=3, target=target, exp_norm="log1p", shuffle=True,
                 gene_list=None):
        super(Kidney_visium, self).__init__()
        self.data_dir = f'{abs_path}/DeepHis2Exp/Dataset/Kidney_visium/Processed_Data'
        self.r = r

        self.gene_list = gene_list
        names = [name.split('/')[-1].split(".")[0] for name in glob.glob(self.data_dir+'/*h5ad')]        
        names.sort()
        self.train = train
        self.ori = ori
        self.adj = adj
        samples = names
        print(f"Datasize:{len(samples)}")

        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        
        print('Loading metadata...')
        adata_dict = {name: sc.read_h5ad(f"{self.data_dir}/{name}.h5ad") for name in tqdm.tqdm(self.names)}
        
        for i, m in adata_dict.items():
            adata_dict[i].var_names_make_unique() 
        self.meta_dict = subset_adatas(adata_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = list(gene_list)

        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)

        if self.ori:
            self.ori_dict = {i:m.to_df()[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m.obsm["spatial"]).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_row','array_col']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i: torch.Tensor(kneighbors_graph(m, n_neighbors=neighs, metric="euclidean", mode='connectivity', include_self=True).toarray())
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        self.color_norm = color_norm
        self.target = target
        
        print("Croping WSI into patches...")
        for ID in tqdm.tqdm(self.names):
            im = self.img_dict[ID.split("-")[0]].permute(1,0,2)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                x, y = centers[i]
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                normalized_img = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(normalized_img)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)

            self.patch_dict[ID]=patches
        
        
    def __getitem__(self, index):
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.meta_dict)
    
    def get_img(self, name):
        path = f"{self.data_dir}/{name}.tif"
        im = Image.open(path).convert('RGB')
        return im


class Skin_Melanoma(torch.utils.data.Dataset):
    """
    Some Information about Skin_Melanoma
    patch, center, exp, adj, oris, sfs = batch
    """
    def __init__(self, train=True, fold=0, r=112, flatten=False, ori=True, adj=True,
                neighs=8, color_norm="reinhard", 
                 num_subsets=3, target=target, exp_norm="log1p", shuffle=True,
                 gene_list=None):
        super(Skin_Melanoma, self).__init__()
        self.data_dir = f'{abs_path}/DeepHis2Exp/Dataset/Skin_Melanoma/processed_data'
        self.r = r

        self.gene_list = gene_list
        names = ["Visium29_B1", "Visium29_C1", "Visium37_D1", "Visium38_B1", "Visium38_D1"]       
        names.sort()
        self.train = train
        self.ori = ori
        self.adj = adj
        samples = names
        print(f"Datasize:{len(samples)}")

        te_names = [samples[fold]]
        self.te_names = te_names[0]
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        
        print('Loading metadata...')
        adata_dict = {name: sc.read_h5ad(f"{self.data_dir}/{name}.h5ad") for name in tqdm.tqdm(self.names)}
        
        for i, m in adata_dict.items():
            adata_dict[i].var_names_make_unique() 
        self.meta_dict = subset_adatas(adata_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = list(gene_list)

        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)

        if self.ori:
            self.ori_dict = {i:m.to_df()[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m.obsm["spatial"]).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_row','array_col']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i: torch.Tensor(kneighbors_graph(m, n_neighbors=neighs, metric="euclidean", mode='connectivity', include_self=True).toarray())
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
        self.color_norm = color_norm
        self.target = target
        
        for ID in tqdm.tqdm(self.names):
            im = self.img_dict[ID.split("-")[0]].permute(1,0,2)
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                x, y = centers[i]
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                normalized_img = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(normalized_img)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)

            self.patch_dict[ID]=patches
        
        
    def __getitem__(self, index):
        ID=self.id2name[index]
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        exps = torch.Tensor(exps)
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        return data
        
    def __len__(self):
        return len(self.meta_dict)
    
    def get_img(self, name):
        path = f"{self.data_dir}/{name}.tif"
        im = Image.open(path).convert('RGB')
        return im

class Generalization_BC_visium(torch.utils.data.Dataset):
    """
    Some Information about BC_visium
    patch, center, exp, adj, oris, sfs = batch
    """
    def __init__(self, train="Alex", gene_list=None, num_subsets=3,
                r=112, exp_norm='lognorm', GAG=False, PAG=False, flatten=False,
                neighs=8, color_norm="reinhard", target=target, shuffle=True,
                 ):
        super(Generalization_BC_visium, self).__init__()
        self.data_dir = f'{abs_path}/DeepHis2Exp/Dataset/BC_visium'
        self.r = r
        self.gene_list = gene_list
        
        alex_names = ['1142243F', '1160920F', 'CID4290', 'CID4465', 'CID44971', 'CID4535']
        tenx_names = ['block1', 'block2']
        self.target = target
        self.flatten=flatten
        self.color_norm = color_norm
        self.GAG = GAG
        self.PAG = PAG
        
        if train == "Alex":
            tr_names = alex_names
            te_names = tenx_names
        elif train == "tenx":
            tr_names = tenx_names
            te_names = alex_names
        print(f"Datasize:{len(tr_names)}")
        print("Train set: ", tr_names)
        print("Test set: ", te_names)
        
        self.names = tr_names
            
        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in tqdm.tqdm(self.names)}
        
        print('Loading metadata...')
        # Read the adata from the path dictionary and create a dictionary to save the adata.
        path = self.data_dir
        adata_dict = {}
        for name in tqdm.tqdm(self.names):
            if name in ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]:
                adata_dict[name] = sc.read_h5ad(f"{path}/{name}.h5ad")
            elif name in ["block1", "block2", "FFPE"]:
                adata_dict[name] = sc.read_h5ad(f"{path}/{name}.h5ad")
            
        self.meta_dict = subset_adatas(adata_dict, num_subsets=num_subsets, shuffle=shuffle)
        self.names = self.meta_dict.keys()
        self.gene_set = gene_list
        
        print('Loading gene expression')
        self.exp_norm = exp_norm
        if self.exp_norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.meta_dict.items()
            }
        elif self.exp_norm=="Pearson residuals":
            self.exp_dict={}
            for i, m in self.meta_dict.items():
                m = m[:, self.gene_set]
                sc.experimental.pp.normalize_pearson_residuals(m)
                m.X = np.nan_to_num(m.X, nan=0)
                self.exp_dict[i] = (m.to_df().values).astype(np.float64)
        
        # For ZINB distribution
        self.ori_dict = {
            i:m.to_df()[self.gene_set].values.astype(np.float64) 
            for i,m in self.meta_dict.items()
        }
        self.counts_dict = {
            i: (m.sum(1)) / np.median(m.sum(1)) for i,m in self.ori_dict.items()
        }    
        
        print('Loading spatial coordinates...')
        self.center_dict = {
            i:np.floor(m.obsm["spatial"].astype(np.int64)).astype(int)
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m.obs[['array_row', 'array_col']].values for i,m in self.meta_dict.items()}         
        self.patch_dict=dfd(lambda :None)

        self.id2name = dict(enumerate(self.names))
        self.adata_dict = {k: v for k, v in adata_dict.items() if k in self.names}
    
        print('Loading image patches...')
        self.patch_dict=dfd(lambda :None)
        for ID in tqdm.tqdm(self.names):
            patches = self.patch_dict[ID]
            centers = self.center_dict[ID]
            n_patches = len(centers)
            patch_dim = 3 * self.r * self.r * 4
            
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches, 3, 2*self.r, 2*self.r))
            for i in range(n_patches):
                x, y = centers[i]
                patch = self.img_dict[ID.split("-")[0]].permute(1,0,2)[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]

                # Color normalization
                patch = color_normalization(patch.numpy(), self.target, method=self.color_norm) 
                patch = torch.Tensor(patch)

                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i] = patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        del self.img_dict
        print('Calculating adjacency matrices or distance matrices...')
        self.SLG_dict = {
            i: features_construct_graph(m, k=neighs, mode="connectivity", metric="euclidean", n_jobs=-1)
            for i,m in self.loc_dict.items()
        }
        self.adj_dict = self.SLG_dict
        

    def __getitem__(self, index):
        ID = self.id2name[index]
        exps = torch.Tensor(self.exp_dict[ID]) #  Spot-level gene expression
        oris = torch.Tensor(self.ori_dict[ID])
        sfs = torch.Tensor(self.counts_dict[ID])
        centers = self.center_dict[ID] # Spatial coordinates for tiling and position embedding
        loc = self.loc_dict[ID]
        positions = torch.LongTensor(loc)
        adj = self.adj_dict[ID].to(torch.float32) # Adjacency matrix construct from spatial coordinates
        patches = self.patch_dict[ID] # Image patches
        data = [patches, positions, exps, adj, oris, sfs]
        return data
        
    def __len__(self):
        return len(self.exp_dict)
    
    def get_img(self, name):
        path = f"{self.data_dir}/{name}/image.tif"
        im = Image.open(path).convert('RGB')
        return im

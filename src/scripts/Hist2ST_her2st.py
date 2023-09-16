import sys
sys.path.append("./")
import time

import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict import *
from HIST2ST import *
# from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
import pickle
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

import os
import glob
import torch
import torchvision
import numpy as np
import scanpy as sc
import pandas as pd 
import scprep as scp
import anndata as ad
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFile, Image
from utils import read_tiff, get_data
from graph_construction import calcADJ
from collections import defaultdict as dfd
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,fold=0,r=4,flatten=True,ori=False,adj=False,prune='Grid',neighs=4):
        super(ViT_HER2ST, self).__init__()
        
        self.cnt_dir = 'data/her2st/data/ST-cnts'
        self.img_dir = 'data/her2st/data/ST-imgs'
        self.pos_dir = 'data/her2st/data/ST-spotfiles'
        self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//r

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        # gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))

        
        gene_list_path = "../../scripts/gene_list.pkl"
        with open(gene_list_path, 'rb') as f:
            gene_list = pickle.load(f)

        
        self.gene_list = gene_list
        names = os.listdir(self.cnt_dir)
        names.sort()
        names = [i[:2] for i in names]
        self.train = train
        self.ori = ori
        self.adj = adj
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        # samples = names[1:33]
        samples = names

        print("Number of samples", len(samples))
        te_names = [samples[fold]]
        print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        print('Loading metadata...')
        self.meta_dict = {i:self.get_meta(i) for i in self.names}
        self.label={i:None for i in self.names}
        self.lbl2id={
            'invasive cancer':0, 'breast glands':1, 'immune infiltrate':2, 
            'cancer in situ':3, 'connective tissue':4, 'adipose tissue':5, 'undetermined':-1
        }
        if not train and self.names[0] in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
            self.lbl_dict={i:self.get_lbl(i) for i in self.names}
            # self.label={i:m['label'].values for i,m in self.lbl_dict.items()}
            idx=self.meta_dict[self.names[0]].index
            lbl=self.lbl_dict[self.names[0]]
            lbl=lbl.loc[idx,:]['label'].values
            # lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
            self.label[self.names[0]]=lbl
        elif train:
            for i in self.names:
                idx=self.meta_dict[i].index
                if i in ['A1','B1','C1','D1','E1','F1','G2','H1','J1']:
                    lbl=self.get_lbl(i)
                    lbl=lbl.loc[idx,:]['label'].values
                    lbl=torch.Tensor(list(map(lambda i:self.lbl2id[i],lbl)))
                    self.label[i]=lbl
                else:
                    self.label[i]=torch.full((len(idx),),-1)
        self.gene_set = list(gene_list)
        self.exp_dict = {
            i:scp.transform.log(scp.normalize.library_size_normalize(m[self.gene_set].values)) 
            for i,m in self.meta_dict.items()
        }
        if self.ori:
            self.ori_dict = {i:m[self.gene_set].values for i,m in self.meta_dict.items()}
            self.counts_dict={}
            for i,m in self.ori_dict.items():
                n_counts=m.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[i]=sf
        self.center_dict = {
            i:np.floor(m[['pixel_x','pixel_y']].values).astype(int) 
            for i,m in self.meta_dict.items()
        }
        self.loc_dict = {i:m[['x','y']].values for i,m in self.meta_dict.items()}
        self.adj_dict = {
            i:calcADJ(m,neighs,pruneTag=prune)
            for i,m in self.loc_dict.items()
        }
        self.patch_dict=dfd(lambda :None)
        self.lengths = [len(i) for i in self.meta_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))
        self.flatten=flatten
    def __getitem__(self, index):
        ID=self.id2name[index]
        im = self.img_dict[ID]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID]
        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        adj = self.adj_dict[ID]
        patches = self.patch_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4
        label=self.label[ID]
        exps = torch.Tensor(exps)
        if patches is None:
            n_patches = len(centers)
            if self.flatten:
                patches = torch.zeros((n_patches,patch_dim))
            else:
                patches = torch.zeros((n_patches,3,2*self.r,2*self.r))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                if self.flatten:
                    patches[i] = patch.flatten()
                else:
                    patches[i]=patch.permute(2,0,1)
            self.patch_dict[ID]=patches
        data=[patches, positions, exps]
        if self.adj:
            data.append(adj)
        if self.ori:
            data+=[torch.Tensor(oris),torch.Tensor(sfs)]
        data.append(torch.Tensor(centers))
        return data
        
    def __len__(self):
        return len(self.exp_dict)

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

    def get_meta(self,name,gene_list=None):
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

import torch
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
# from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def pk_load(fold,mode='train',flatten=False,dataset='her2st',r=4,ori=True,adj=True,prune='Grid',neighs=4):
    assert dataset in ['her2st','cscc']
    if dataset=='her2st':
        dataset = ViT_HER2ST(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    elif dataset=='cscc':
        dataset = ViT_SKIN(
            train=(mode=='train'),fold=fold,flatten=flatten,
            ori=ori,neighs=neighs,adj=adj,prune=prune,r=r
        )
    return dataset


def calculate_correlation(attr_1, attr_2):
    r = pearsonr(attr_1, 
                       attr_2)[0]
    return r

name=[*[f'A{i}' for i in range(2,7)],*[f'B{i}' for i in range(1,7)],
      *[f'C{i}' for i in range(1,7)],*[f'D{i}' for i in range(1,7)],
      *[f'E{i}' for i in range(1,4)],*[f'F{i}' for i in range(1,4)], *[f'G{i}' for i in range(1,4)]]

patients = ['P2', 'P5', 'P9', 'P10']
reps = ['rep1', 'rep2', 'rep3']
skinname = []
for i in patients:
    for j in reps:
        skinname.append(i+'_ST_'+j)
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

cnt_dir = 'data/her2st/data/ST-cnts'
names = os.listdir(cnt_dir)
names.sort()
names = [i[:2] for i in names]

df = pd.DataFrame()
i = int(sys.argv[1])
# i=1
fold = i
test_sample = names[fold]



data='her2st'
prune='Grid' if data=='her2st' else 'NA'
# genes=171 if data=='cscc' else 785
gene_list_path = "../../scripts/gene_list.pkl"
with open(gene_list_path, 'rb') as f:
    gene_list = pickle.load(f)

genes = len(gene_list)
trainset = pk_load(fold,'train',dataset=data,flatten=False,adj=True,ori=True,prune=prune)
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

model=Hist2ST(
    depth1=d1, depth2=d2,depth3=d3,n_genes=genes,
    kernel_size=k, patch_size=p,
    heads=h, channel=c, dropout=0.2,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
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
if not os.path.isdir("./model/"):
    os.mkdir("./model/")

torch.save(model.state_dict(),f"./model/{fold}-Hist2ST.ckpt")
testset = pk_load(fold,'test',dataset=data,flatten=False,adj=True,ori=True,prune=prune)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
adata_pred, adata_truth = test(model, test_loader,'cuda')
# gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))

## UPDATE: should keep this same as STimage
# n_genes = len(gene_list)

adata_pred.var_names = gene_list
adata_truth.var_names = gene_list

pred_adata = adata_pred.copy()
test_dataset = adata_truth.copy()

for gene in pred_adata.var_names:
    cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, test_sample, "Hist2ST"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
              ignore_index=True)

del model
torch.cuda.empty_cache()

df.to_csv("../../results/her2st/hist2st_cor_{}.csv".format(test_sample))

# with open("../../results/her2st/hist2st_times.txt", 'a') as f:
#     f.write(f"{test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")

# with open(f"../../results/her2st/hist2st_preds_{test_sample}.pkl", 'wb') as f:
#     pickle.dump([pred_adata,test_dataset], f)

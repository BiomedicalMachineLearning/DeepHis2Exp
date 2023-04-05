import argparse
import configparser
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import read_h5ad
import scanpy as sc

def read_gene_set(path):
    DATA_PATH = Path(path)
    adata_all = read_h5ad(DATA_PATH / "all_adata.h5ad")
    # gene_selection == "top250":
    adata_all.var["mean_expression"] = np.asarray(np.mean(adata_all.X, axis=0)).squeeze()
    comm_genes = adata_all.var["mean_expression"].sort_values(ascending=False
                                                              ).index[0:250]
    return set(comm_genes)

def intersect_section_genes(geneset, adata_dict):
    comm_genes = geneset.intersection(*[set(v.var_names) for k,v in adata_dict.items()])
    return list(comm_genes)


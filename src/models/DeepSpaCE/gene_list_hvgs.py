from pathlib import Path
from anndata import read_h5ad
import numpy as np
import sys
import pickle

with open('../../scripts/1000hvg_common.pkl', 'rb') as f:
    gene_list = pickle.load(f).to_list()

print(','.join(gene_list))


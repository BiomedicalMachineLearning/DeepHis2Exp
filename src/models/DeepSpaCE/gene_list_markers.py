from pathlib import Path
from anndata import read_h5ad
import numpy as np
import sys

# DATA_PATH = Path("../../data/pfizer2")

# if (DATA_PATH / "train_adata.h5ad").exists():
#     adata_all = read_h5ad(DATA_PATH / "train_adata.h5ad")
#     comm_genes = adata_all.var_names[adata_all.var.highly_variable]
#     comm_genes = set(comm_genes)
# else:
#     sys.exit("pass gene list")

print(','.join(list(set(["COX6C","TTLL12", "HSP90AB1", 
           "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]))))


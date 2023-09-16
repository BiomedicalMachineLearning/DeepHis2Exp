import pickle
gene_list_path = "../../scripts/gene_list.pkl"
with open(gene_list_path, 'rb') as f:
    gene_list = pickle.load(f)
    
print(','.join(gene_list))
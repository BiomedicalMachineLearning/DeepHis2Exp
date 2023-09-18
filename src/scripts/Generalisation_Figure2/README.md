# Assessing the model generalisation capabilities on out-of-domain dataset

The scripts are similar to the benchmarking scripts. The difference is the dataset splitting strategy. The related codes were highlight below.
···python
def dataset_wrap():
    train_sample = list(set(samps1)-set(["1160920F","CID4290"])) # Alex visium samples
    val_sample = ["1160920F","CID4290"] # Alex visium samples
    test_sample = samps2 # 10x visium samples

    tr_name = list(set([i for i in list(adata_dict.keys()) for tr in train_sample if tr in i]))
    val_name = list(set([i for i in list(adata_dict.keys()) for val in val_sample if val in i]))
    te_name = list(set([i for i in list(adata_dict.keys()) for te in test_sample if te in i]))

    trainset = ViT_Anndata(adata_dict = adata_dict, train_set = tr_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    valset = ViT_Anndata(adata_dict = adata_dict, train_set = val_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    testset = ViT_Anndata(adata_dict = adata_dict, train_set = te_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )

    print("LOADED TRAINSET")
    train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)
    val_loader = DataLoader(valset, batch_size=1, num_workers=0, shuffle=True)
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    return train_loader, val_loader, test_loader
```
The *Figures* folder shows the results that generated from the scripts.

![Benchmarking Overview](https://github.com/BiomedicalMachineLearning/DeepHis2Exp/blob/main/Figures/Figure2.png)
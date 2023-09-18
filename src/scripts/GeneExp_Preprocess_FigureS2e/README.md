# Tutorial
Please cd ./src/models/Hist2ST/ to run the command below:
python /scratch/imb/uqyjia11/Yuanhao/10X_visium/Hist2ST-main/HIST2ST_train.py --fold 0 --norm log1p

fold is the sample index, norm is the gene expression preprocessing methods. 

We assess 4 kinds of preprocessing methods log1p, MinmaxScale, log1p+Normalization, Raw respectively.

The related codes are highlight below:
```python
        if self.norm=="lognorm":
            self.exp_dict = {
                i:scp.transform.log(scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values)).astype(np.float64)
                for i,m in self.adata_dict.items()
            }
        elif self.norm=="raw":
            self.exp_dict = {
                i:m.to_df()[self.gene_set].values.astype(np.float64) 
                for i,m in self.adata_dict.items()
            }
        elif self.norm=="norm":
            self.exp_dict = {
                i:scp.normalize.library_size_normalize(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.adata_dict.items()
            }
        elif self.norm=="log1p":
            self.exp_dict = {
                i:scp.transform.log(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.adata_dict.items()
            }
        elif self.norm=="minmax":
            self.exp_dict = {
                i:MinMaxScaler().fit_transform(m.to_df()[self.gene_set].values).astype(np.float64)
                for i,m in self.adata_dict.items()
            }
```
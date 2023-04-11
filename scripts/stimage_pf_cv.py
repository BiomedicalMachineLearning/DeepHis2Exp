import sys
import time
import stlearn as st
st.settings.set_figure_params(dpi=300)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
file = Path("../models/STimage/stimage").resolve()
parent= file.parent
sys.path.append(str(parent))
from PIL import Image
from stimage._utils import gene_plot, Read10X, ReadOldST, tiling, ensembl_to_id
from stimage._model import CNN_NB_multiple_genes, negative_binomial_layer, negative_binomial_loss
from stimage._data_generator import DataGenerator
import tensorflow as tf
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
# import geopandas as gpd
from sklearn.neighbors import KDTree
from anndata import read_h5ad
from tensorflow.keras import backend as K
import scanpy as sc

import matplotlib.pyplot as plt
from libpysal.weights.contiguity import Queen
from libpysal import examples
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import splot
from splot.esda import moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from esda.moran import Moran_BV, Moran_Local_BV
from splot.esda import plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation
import pickle
from scipy import stats

def plot_correlation(df, attr_1, attr_2):
    r = stats.pearsonr(df[attr_1], 
                       df[attr_2])[0] **2

    g = sns.lmplot(data=df,
        x=attr_1, y=attr_2,
        height=5, legend=True
    )
    # g.set(ylim=(0, 360), xlim=(0,360))

    g.set_axis_labels(attr_1, attr_2)
    plt.annotate(r'$R^2:{0:.2f}$'.format(r),
                (max(df[attr_1])*0.9, max(df[attr_2])*0.9))
    return g


def calculate_correlation(attr_1, attr_2):
    r = stats.pearsonr(attr_1, 
                       attr_2)[0]
    return r

def calculate_correlation_2(attr_1, attr_2):
    r = stats.spearmanr(attr_1, 
                       attr_2)[0]
    return r


DATA_PATH = Path("../data/pfizer")
OUT_PATH = Path("../trained_models")

adata_all = read_h5ad(DATA_PATH / "all_adata.h5ad")
# adata_all.obs['tile_path'] = adata_all.obs['tile_path'].apply(lambda x: x.replace('/clusterdata/uqxtan9/Q1851/Xiao/Working_project/','../data/'))

# adata_all = ensembl_to_id(adata_all)

samples = adata_all.obs["library_id"].unique().tolist()

# gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
#            "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

gene_list_path = "./gene_list.pkl"
with open(gene_list_path, 'rb') as f:
    gene_list = pickle.load(f)

df = pd.DataFrame()

i = int(sys.argv[1])
test_sample = samples[i]
n_genes = len(gene_list)

adata_all_train_valid = adata_all[adata_all.obs["library_id"].isin(
    adata_all.obs.library_id.cat.remove_categories(test_sample).unique())]

training_index = adata_all_train_valid.obs.sample(frac=0.7, random_state=1).index
training_dataset = adata_all_train_valid[training_index,].copy()

valid_index = adata_all_train_valid.obs.index.isin(training_index)
valid_dataset = adata_all_train_valid[~valid_index,].copy()

test_index = adata_all.obs.library_id == test_sample
test_dataset_1 = adata_all[test_index,].copy()


train_gen = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=training_dataset, 
                      genes=gene_list, aug=False),
        output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
        output_shapes=([299,299,3], tuple([1]*n_genes))
)
train_gen_ = train_gen.shuffle(buffer_size=500).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
valid_gen = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=valid_dataset, 
                      genes=gene_list), 
        output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
        output_shapes=([299,299,3], tuple([1]*n_genes))
)
valid_gen_ = valid_gen.shuffle(buffer_size=500).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_gen_1 = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=test_dataset_1, 
                      genes=gene_list), 
        output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
        output_shapes=([299,299,3], tuple([1]*n_genes))
)
test_gen__1 = test_gen_1.batch(1)

K.clear_session()
model = CNN_NB_multiple_genes((299, 299, 3), n_genes)
# WEIGHT_PATH = Path("/clusterdata/uqjxie6/Q2051/STimage_project/pretrained_model")
# model.load_weights(OUT_PATH / "CNN_NB_1000HVG.h5")
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                        restore_best_weights=False)

start_train = time.perf_counter()

train_history = model.fit(train_gen_,
                      epochs=100,
                      validation_data=valid_gen_,
                      callbacks=[callback]
                      )

end_train = time.perf_counter()

if save_model_weights:
    model.save(OUT_PATH / "stimage_model_weights.h5")

test_predictions = model.predict(test_gen__1)
from scipy.stats import nbinom
y_preds = []
for i in range(n_genes):
    n = test_predictions[i][:, 0]
    p = test_predictions[i][:, 1]
    y_pred = nbinom.mean(n, p)
    y_preds.append(y_pred)
test_dataset_1.obsm["predicted_gene"] = np.array(y_preds).transpose()

test_dataset_1_ = test_dataset_1[:,gene_list].copy()
test_dataset_1_.X = test_dataset_1_.obsm["predicted_gene"]

pred_adata = test_dataset_1_
test_dataset = test_dataset_1

for gene in pred_adata.var_names:
    cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, test_sample, "STimage"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
                  ignore_index=True)

df.to_csv("../results/pf_cv/stimage_cor_{}.csv".format(test_sample))

with open("../results/pf_cv/stimage_times.txt", 'a') as f:
    f.write(f"{test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")

with open(f"../results/pf_cv/stimage_preds_{test_sample}.pkl", 'wb') as f:
    pickle.dump([pred_adata,test_dataset], f)


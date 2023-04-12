import time
import sys
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
import pickle
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

from scipy import stats
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as preprocess_densenet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable

import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KDTree

import numpy as np

# def center_crop(img, new_width=None, new_height=None):        

#     width = img.shape[1]
#     height = img.shape[0]

#     if new_width is None:
#         new_width = min(width, height)

#     if new_height is None:
#         new_height = min(width, height)

#     left = int(np.ceil((width - new_width) / 2))
#     right = width - int(np.floor((width - new_width) / 2))

#     top = int(np.ceil((height - new_height) / 2))
#     bottom = height - int(np.floor((height - new_height) / 2))

#     if len(img.shape) == 2:
#         center_cropped_img = img[top:bottom, left:right]
#     else:
#         center_cropped_img = img[top:bottom, left:right, ...]

#     return center_cropped_img

# class DataGenerator(keras.utils.Sequence):
#     """
#     data generator for multiple branches gene prediction model
#     """

#     def __init__(self, adata, dim=(299, 299), n_channels=3, genes=None, aug=False, tile_path="tile_path"):
#         'Initialization'
#         self.dim = (224,224)
#         self.adata = adata
#         self.n_channels = n_channels
#         self.genes = genes
#         self.num_genes = len(genes)
#         self.aug = aug
#         self.tile_path = tile_path
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(self.adata.n_obs)

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Find list of IDs
#         obs_temp = self.adata.obs_names[index]

#         # Generate data
#         X_img = self._load_img(obs_temp)
#         y = self._load_label(obs_temp)

#         return X_img, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(self.adata.n_obs)

#     def _load_img(self, obs):
#         img_path = self.adata.obs.loc[obs, 'tile_path']
#         # X_img = image.load_img(img_path, target_size=self.dim)
#         X_img = image.load_img(img_path)
#         X_img = image.img_to_array(X_img).astype('uint8')
        
#         X_img = center_crop(X_img,new_width=224, new_height=224)
        
#         #         X_img = np.expand_dims(X_img, axis=0)
#         #         n_rotate = np.random.randint(0, 4)
#         #         X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
#         if self.aug:
#             X_img = seq_aug(image=X_img)
# #         X_img = preprocess_resnet(X_img)
#         return X_img

#     def _load_label(self, obs):
#         batch_adata = self.adata[obs, self.genes].copy()

#         return tuple([batch_adata.to_df()[i].values for i in self.genes])

#     def get_classes(self):
#         return self.adata.to_df().loc[:, self.genes]


def STNet(tile_shape, output_shape, mean_exp_tf):
    tile_input = Input(shape=tile_shape, name = "tile_input")
    DenseNet121_base = DenseNet121(input_tensor=tile_input, weights='imagenet', include_top=False)
    for layer in DenseNet121_base.layers:
        layer.trainable = False
    
    cnn = DenseNet121_base.output
    cnn = GlobalAveragePooling2D()(cnn)
#     cnn = Dropout(0.5)(cnn)
#     cnn = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
#                 activity_regularizer=tf.keras.regularizers.l2(0.01))(cnn)
    # cnn = Dense(256, activation='relu')(cnn)
    
    outputs = Dense(output_shape, activation='linear', bias_initializer=mean_exp_tf)(cnn)
    model = Model(inputs=tile_input, outputs=outputs)

#     optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanSquaredError()])    
    return model



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
adata_all = read_h5ad(DATA_PATH / "all_adata_224.h5ad")
# adata_all.obs['tile_path'] = adata_all.obs['tile_path'].apply(lambda x: x.replace('/clusterdata/uqxtan9/Q1851/Xiao/Working_project/','../data/'))

# adata_all = ensembl_to_id(adata_all)

samples = adata_all.obs["library_id"].unique().tolist()

# gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
#            "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

# gene_list_path = "./gene_list.pkl"
# with open(gene_list_path, 'rb') as f:
#     gene_list = pickle.load(f)
from read_stimage_genes import read_gene_set_hvg
gene_list = read_gene_set_hvg("../data/pfizer/", out="list")


df = pd.DataFrame()

i = int(sys.argv[1])
test_sample = samples[i]

n_genes = len(gene_list)

adata_all_train_valid = adata_all[adata_all.obs["library_id"].isin(
    adata_all.obs.library_id.cat.remove_categories(test_sample).unique())]

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
                          genes=gene_list, aug=False, dim=(224, 224)),
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([224,224,3], tuple([1]*n_genes))
)
train_gen_ = train_gen.shuffle(buffer_size=500).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
valid_gen = tf.data.Dataset.from_generator(
            lambda:DataGenerator(adata=valid_dataset, 
                          genes=gene_list, dim=(224, 224)), 
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([224,224,3], tuple([1]*n_genes))
)
valid_gen_ = valid_gen.shuffle(buffer_size=500).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_gen_1 = tf.data.Dataset.from_generator(
            lambda:DataGenerator(adata=test_dataset_1, 
                          genes=gene_list, dim=(224, 224)), 
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([224,224,3], tuple([1]*n_genes))
)
test_gen__1 = test_gen_1.batch(1)


K.clear_session()
mean_exp = training_dataset[:,gene_list].to_df().mean()
mean_exp_tf = tf.keras.initializers.RandomUniform(minval=mean_exp, 
                                                  maxval=mean_exp)
model = STNet((224, 224, 3), n_genes, mean_exp_tf)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                        restore_best_weights=False)

start_train = time.perf_counter()
train_history = model.fit(train_gen_,
                      epochs=100,
                      validation_data=valid_gen_,
                      callbacks=[callback]
                      )

end_train = time.perf_counter()
test_predictions = model.predict(test_gen__1)

test_dataset_1.obsm["predicted_gene"] = test_predictions
test_dataset_1_ = test_dataset_1[:,gene_list].copy()
test_dataset_1_.X = test_dataset_1_.obsm["predicted_gene"]

pred_adata = test_dataset_1_
test_dataset = test_dataset_1

with open(f"../results/pf_cv/stnet_preds_{test_sample}.pkl", 'wb') as f:
    pickle.dump([pred_adata,test_dataset], f)

for gene in pred_adata.var_names:
    pred = pred_adata.to_df().loc[:,gene]
    pred = pred.fillna(0)
    cor_val = calculate_correlation_2(pred, test_dataset.to_df().loc[:,gene])
    cor_pearson = calculate_correlation(pred, test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val,cor_pearson, test_sample, "STNET"], 
                         index=["Gene", "Spearman correlation", "Pearson correlation","Slide", "Method"]),
              ignore_index=True)

df.to_csv("../results/pf_cv/stnet_cor_{}.csv".format(test_sample))


with open("../results/pf_cv/stnet_times.txt", 'a') as f:
    f.write(f"{test_sample} {end_train - start_train} - {time.strftime('%H:%M:%S', time.localtime())}")



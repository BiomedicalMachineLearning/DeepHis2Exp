# Tutorial
Please cd ./src/models/Hist2ST/ to run the command below:
python /scratch/imb/uqyjia11/Yuanhao/10X_visium/Hist2ST-main/HIST2ST_train.py --fold 0 --norm log1p

fold is the sample index, norm is the gene expression preprocessing methods. We assess 4 kinds of preprocessing methods log1p, MinmaxScale, log1p+Normalization, Raw.
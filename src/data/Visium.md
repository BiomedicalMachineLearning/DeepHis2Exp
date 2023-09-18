# README

Steps:
1. Copy `data_vit.py` to the same folder as Hist2ST
2. Copy visium data from `Q2051/STimage_project/STimage_dataset/RAW/Alex_NatGen_6BreastCancer/` and `Q2051/STimage_project/STimage_dataset/RAW/breast_cancer_10x_visium/`.
   Samples used are
   ```
    data_dir1 = "../../data/Alex_NatGen/"
    data_dir2 = "../../data/Breast_Cancer_10x/"
    
    samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
    samps2 = ["block1", "block2", "FFPE", "1168993F"]
   ```
4. Edit the files in all directories to have all `*.tif` images renamed to `image.tif`.
5. Notes for `Hist2ST_visium.py`
    - This script should be run from the same directory as `window_adata.py`, see also the job script to see the run directory
    - By default it does leave 1 out cross validation
    - Line 215: `sizes = [4000 for i in range(len(adata_dict0))]` defines the gridding size
    - You can edit the gene list in the file
    - Line 184-185: You need to modify the path of the data so that it's correct
    - Line 301: You can uncomment this line to save the model
    - Line 405,407: You can edit this line to save the results elsewhere
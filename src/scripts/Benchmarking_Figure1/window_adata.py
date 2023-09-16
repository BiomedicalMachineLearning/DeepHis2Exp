import numpy as np
import squidpy as sq
import anndata as ad
import scanpy as sc
from squidpy.im import ImageContainer

def adata_img_crop_tissue(adata, img, out_dim):
    r = int(out_dim/2)
    coords = adata.obsm['spatial'].astype(np.int64)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    crop_corner = img.crop_corner(y_min - r, x_min - r,
                                  size=((2*r) + y_max-y_min, (2*r) + x_max-x_min))
    adata_crop = crop_corner.subset(adata)
    return crop_corner, adata_crop

# sizes = [3000 for i in range(len(adata_dict))]

def window_adata(adata_dict, sizes):
    "Window every element in `adata_dict'"
    size_dict = {}
    for i,k in enumerate(adata_dict):
        size_dict[k] = sizes[i]

    adata_sub_dict = {}

    for key, val in adata_dict.items():
        print("windowing", key)
        adata = val
        img = ImageContainer()
        img.add_img(adata.uns['spatial'][key]['metadata']['source_image_path'], 
                    layer='image', library_id=key)

        # crop to tissue
        crop_corner, adata_tissue = adata_img_crop_tissue(adata, img, 224)
        print("Num spots: ", adata_tissue.n_obs)
        # split image into smaller crops
        crops = crop_corner.generate_equal_crops(size=size_dict[key]) # average 600-800 spots per sample
        crops_img = crop_corner.generate_equal_crops(size=size_dict[key], as_array="image", squeeze=True) # average 600-800 spots per sample
        crops_img = list(crops_img)
        summ = 0
        for i, crop in enumerate(crops):
            adata_crop = crop.subset(adata_tissue)
            summ += adata_crop.n_obs
            print(adata_crop.n_obs)
            if adata_crop.n_obs > 10:
                namee = key+"_"+str(i)
                adata_sub_dict[namee] = adata_crop
        print("Total: ", summ)
        
    return adata_sub_dict

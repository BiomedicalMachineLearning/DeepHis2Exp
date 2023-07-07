import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
# from scipy.sparse import csr_matrix
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

class HERDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, expr_path, reduced_mtx_path):
        #image_path is the path of an entire slice of visium h&e stained image (~2.5GB)
        
        #spatial_pos_csv
            #barcode name
            #detected tissue boolean
            #x spot index
            #y spot index
            #x spot position (px)
            #y spot position (px)
        
        #expression_mtx
            #feature x spot (alphabetical barcode order)
    
        #barcode_tsv
            #spot barcodes - alphabetical order
        
        # NEED TO TRANSPOSE IMAGE BC I THINK X_Y ARE SWAPPED (CHECKED W/ HIST2ST)
        self.whole_image = cv2.imread(image_path).transpose(1,0,2)
        self.spatial_pos_csv = self.load_spatial_pos(spatial_pos_path)
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = self.load_barcode(expr_path)
        self.reduced_matrix = np.load(reduced_mtx_path).T  #cell x features
        
        print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def load_barcode(self, expr_path):
        expr = pd.read_csv(expr_path,sep='\t',index_col=0)
        return expr.index.to_frame().reset_index(drop=True)
        
    def load_spatial_pos(self, spatial_pos_path):
        df = pd.read_csv(spatial_pos_path,sep='\t')
        df[0]  = df['x'].astype(str) + 'x' + df['y'].astype(str)
        df["pixel_x"] = df["pixel_x"].astype(int)
        df["pixel_y"] = df["pixel_y"].astype(int)
        df = df[[0, "selected", "x", "y", "pixel_x", "pixel_y"]]
        df.columns = list(range(6))
        return df

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx,0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
        image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
        image = self.transform(image)
        
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx,:]).float()  #cell x features (3467)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1,v2]

        return item


    def __len__(self):
        return len(self.barcode_tsv)
    
# class HERDataset2(torch.utils.data.Dataset):
#     def __init__(self, image_path, spatial_pos_path, expr_path, reduced_mtx_path):
#         #image_path is the path of an entire slice of visium h&e stained image (~2.5GB)
        
#         #spatial_pos_csv
#             #barcode name
#             #detected tissue boolean
#             #x spot index
#             #y spot index
#             #x spot position (px)
#             #y spot position (px)
        
#         #expression_mtx
#             #feature x spot (alphabetical barcode order)
    
#         #barcode_tsv
#             #spot barcodes - alphabetical order

#         self.whole_image = cv2.imread(image_path)
#         self.spatial_pos_csv = self.load_spatial_pos(spatial_pos_path)
#         # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
#         self.barcode_tsv = self.load_barcode(expr_path)
#         self.reduced_matrix = np.load(reduced_mtx_path).T  #cell x features
        
#         print("Finished loading all files")

#     def transform(self, image):
#         image = Image.fromarray(image)
#         # Random flipping and rotations
#         if random.random() > 0.5:
#             image = TF.hflip(image)
#         if random.random() > 0.5:
#             image = TF.vflip(image)
#         angle = random.choice([180, 90, 0, -90])
#         image = TF.rotate(image, angle)
#         return np.asarray(image)

#     def load_barcode(self, expr_path):
#         expr = pd.read_csv(expr_path,sep='\t',index_col=0)
#         return expr.index.to_frame().reset_index(drop=True)
        
#     def load_spatial_pos(self, spatial_pos_path):
#         df = pd.read_csv(spatial_pos_path,sep='\t')
#         df[0]  = df['x'].astype(str) + 'x' + df['y'].astype(str)
#         df["pixel_x"] = df["pixel_x"].astype(int)
#         df["pixel_y"] = df["pixel_y"].astype(int)
#         df = df[[0, "selected", "x", "y", "pixel_x", "pixel_y"]]
#         df.columns = list(range(6))
#         return df

#     def __getitem__(self, idx):
#         item = {}
#         barcode = self.barcode_tsv.values[idx,0]
#         v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
#         v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
#         print(self.whole_image.shape)
#         print(self.whole_image.shape)

#         image = self.whole_image[(v1-56):(v1+56),(v2-56):(v2+56)]
#         print(image.shape)
#         # image = self.transform(image)
        
#         item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
#         item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx,:]).float()  #cell x features (3467)
#         item['barcode'] = barcode
#         item['spatial_coords'] = [v1,v2]

#         return item


#     def __len__(self):
#         return len(self.barcode_tsv)
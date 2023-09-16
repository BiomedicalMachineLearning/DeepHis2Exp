import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

import config as CFG
from models import CLIPModel, CLIPModel_ViT, CLIPModel_ViT_L, CLIPModel_CLIP, CLIPModel_resnet101, CLIPModel_resnet152
from dataset_her2 import HERDataset
import scanpy as sc
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd

import scanpy as sc
import pickle

def build_loaders_inference():
    names = os.listdir("data/her2st/ST-cnts")
    names.sort()
    names = [i[:2] for i in names]
    # print(names)
    # print(len(names))
    datasets = [
        HERDataset(image_path = f"data/her2st/images/{samp}.jpeg",
           spatial_pos_path = f"data/her2st/ST-spotfiles/{samp}_selection.tsv",
           # reduced_mtx_path = f"data/filtered_expression_matrices/{samp}/harmony_matrix.npy",
           reduced_mtx_path =  f"data/filtered_expression_matrices/her2_subset/{samp}/hvg_matrix_plusmarkers.npy",
           expr_path = f"data/her2st/ST-cnts/{samp}.tsv")
        for samp in names
    ]

    # dataset = torch.utils.data.ConcatDataset([datasets[fold]])
    dataset = torch.utils.data.ConcatDataset(datasets)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    
    print("Finished building loaders")
    return test_loader


def get_image_embeddings(model_path, model):
    test_loader = build_loaders_inference()
    # model = CLIPModel().cuda()

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot') # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")
    
    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
    
    return torch.cat(test_image_embeddings)


def get_spot_embeddings(model_path, model):
    test_loader = build_loaders_inference()
    # model = CLIPModel().cuda()

    state_dict = torch.load(model_path)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('module.', '')  # remove the prefix 'module.'
        new_key = new_key.replace('well', 'spot') # for compatibility with prior naming
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict)
    model.eval()

    print("Finished loading model")

    spot_embeddings = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # spot_features = model.spot_encoder(batch["reduced_expression"].cuda()) 
            # spot_embeddings = model.spot_projection(spot_features)
            spot_embeddings.append(model.spot_projection(batch["reduced_expression"].cuda()))
    return torch.cat(spot_embeddings)


#2265x256, 2277x256
def find_matches(spot_embeddings, query_embeddings, top_k=1):
    #find the closest matches 
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    
    return indices.cpu().numpy()


#outputs:
#data sizes: (3467, 2378) (3467, 2349) (3467, 2277) (3467, 2265)

def save_embeddings(model_path, save_path, datasize, dim, fold):

    # datasize = [4784, 2432, 1211, 1162, 1127, 4895, 3798, 3987, 2518]
    # model_path = "clip/best.pt"
    # save_path = "clip/embeddings/"
    model = CLIPModel(spot_embedding=dim).cuda()


    img_embeddings_all = get_image_embeddings(model_path, model)
    spot_embeddings_all = get_spot_embeddings(model_path, model)

    img_embeddings_all = img_embeddings_all.cpu().numpy()
    spot_embeddings_all = spot_embeddings_all.cpu().numpy()
    print(img_embeddings_all.shape)
    print(spot_embeddings_all.shape)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(datasize)):
        index_start = sum(datasize[:i])
        index_end = sum(datasize[:i+1])
        image_embeddings = img_embeddings_all[index_start:index_end]
        spot_embeddings = spot_embeddings_all[index_start:index_end]
        print(image_embeddings.shape)
        print(spot_embeddings.shape)
        np.save(save_path + "img_embeddings_" + str(i+1) + ".npy", image_embeddings.T)
        np.save(save_path + "spot_embeddings_" + str(i+1) + ".npy", spot_embeddings.T)

if not os.path.exists("clip_her2_subset/embeddings_0"):        
    SAVE_EMBEDDINGS = True
else:
    SAVE_EMBEDDINGS = False

names = os.listdir("data/her2st/ST-cnts")
names.sort()
names = [i[:2] for i in names]

datasize = [np.load(f"data/filtered_expression_matrices/her2_subset/{name}/hvg_matrix_plusmarkers.npy").shape[1] for name in names]
if SAVE_EMBEDDINGS:
    for fold in range(36):
        save_embeddings(model_path=f"clip_her2_subset/best_{fold}.pt", save_path=f"clip_her2_subset/embeddings_{fold}/",
                        datasize=datasize, dim=769, fold=fold)
    
samps = names

#infer spot embeddings and expression
# fold = int(sys.argv[1])
df = pd.DataFrame()

spot_expressions = [np.load(f"data/filtered_expression_matrices/her2_subset/{samp}/hvg_matrix_plusmarkers.npy")
                   for samp in samps]

for fold  in range(36):
    print(f"EVALUATING FOLD {fold}________________________________")
    print(f"GT SHAPE {spot_expressions[fold].shape}_____________________")
    # spot_expression1 = np.load("filtered_expression_matrices/1/harmony_matrix.npy")
    # spot_expression2 = np.load("filtered_expression_matrices/2/harmony_matrix.npy")
    # spot_expression3 = np.load("filtered_expression_matrices/3/harmony_matrix.npy")
    # spot_expression4 = np.load("filtered_expression_matrices/4/harmony_matrix.npy")

    save_path = f"clip_her2_subset/embeddings_{fold}/"
    spot_embeddings = [np.load(save_path + f"spot_embeddings_{i+1}.npy") for i in range(36)]
    # spot_embeddings1 = np.load(save_path + "spot_embeddings_1.npy")
    # spot_embeddings2 = np.load(save_path + "spot_embeddings_2.npy")
    # spot_embeddings3 = np.load(save_path + "spot_embeddings_3.npy")
    # spot_embeddings4 = np.load(save_path + "spot_embeddings_4.npy")
    image_embeddings = np.load(save_path + f"img_embeddings_{fold+1}.npy")


    #query
    image_query = image_embeddings
    expression_gt = spot_expressions[fold]
    # spot_embeddings.pop(fold)
    # spot_expressions.pop(fold)
    spot_embeddings = spot_embeddings[:fold] + spot_embeddings[fold+1:]
    spot_expressions_rest = spot_expressions[:fold] + spot_expressions[fold+1:]
    
    spot_key = np.concatenate(spot_embeddings, axis = 1)
    expression_key = np.concatenate(spot_expressions_rest, axis = 1)

    method = "average"
    save_path = ""
    if image_query.shape[1] != 256:
        image_query = image_query.T
        print("image query shape: ", image_query.shape)
    if expression_gt.shape[0] != image_query.shape[0]:
        expression_gt = expression_gt.T
        print("expression_gt shape: ", expression_gt.shape)
    if spot_key.shape[1] != 256:
        spot_key = spot_key.T
        print("spot_key shape: ", spot_key.shape)
    if expression_key.shape[0] != spot_key.shape[0]:
        expression_key = expression_key.T
        print("expression_key shape: ", expression_key.shape)

    if method == "simple":
        indices = find_matches(spot_key, image_query, top_k=1)
        matched_spot_embeddings_pred = spot_key[indices[:,0],:]
        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        matched_spot_expression_pred = expression_key[indices[:,0],:]
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "average":
        print("finding matches, using average of top 50 expressions")
        indices = find_matches(spot_key, image_query, top_k=50)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            matched_spot_embeddings_pred[i,:] = np.average(spot_key[indices[i,:],:], axis=0)
            matched_spot_expression_pred[i,:] = np.average(expression_key[indices[i,:],:], axis=0)

        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "weighted_average":
        print("finding matches, using weighted average of top 50 expressions")
        indices = find_matches(spot_key, image_query, top_k=50)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_key.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], expression_key.shape[1]))
        for i in range(indices.shape[0]):
            a = np.sum((spot_key[indices[i,0],:] - image_query[i,:])**2) #the smallest MSE
            weights = np.exp(-(np.sum((spot_key[indices[i,:],:] - image_query[i,:])**2, axis=1)-a+1))
            if i == 0:
                print("weights: ", weights)
            matched_spot_embeddings_pred[i,:] = np.average(spot_key[indices[i,:],:], axis=0, weights=weights)
            matched_spot_expression_pred[i,:] = np.average(expression_key[indices[i,:],:], axis=0, weights=weights)

        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)



    true = expression_gt
    pred = matched_spot_expression_pred

    print(pred.shape)
    print(true.shape)
    print(np.max(pred))
    print(np.max(true))
    print(np.min(pred))
    print(np.min(true))

    #genewise correlation
    corr = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        corr[i] = np.corrcoef(pred[i,:], true[i,:],)[0,1] #corrcoef returns a matrix
    #remove nan
    corr = corr[~np.isnan(corr)]
    print("Mean correlation across cells: ", np.mean(corr))

    corr = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr[i] = np.corrcoef(pred[:,i], true[:,i],)[0,1] #corrcoef returns a matrix
    #remove nan
    corr = corr[~np.isnan(corr)]
    print("number of non-zero genes: ", corr.shape[0])
    print("mean correlation: ", np.mean(corr))
    print("max correlation: ", np.max(corr))
    print("number of genes with correlation > 0.3: ", np.sum(corr > 0.3))

    save_path=f"clip_her2_subset/preds_{fold}/"
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if save_path != "":
        # np.save(save_path + "matched_spot_embeddings_pred.npy", matched_spot_embeddings_pred.T)
        # np.save(save_path + "matched_spot_expression_pred.npy", matched_spot_expression_pred.T)

        np.save(save_path + "matched_spot_embeddings_pred.npy", matched_spot_embeddings_pred.T)
        np.save(save_path + "matched_spot_expression_pred.npy", matched_spot_expression_pred.T)


    expression_gt = np.load(f"data/filtered_expression_matrices/her2_subset/{samps[fold]}/hvg_matrix_plusmarkers.npy")
    matched_spot_expression_pred_1 = np.load(f"clip_her2_subset/preds_{fold}/matched_spot_expression_pred.npy")
    assert expression_gt.shape == matched_spot_expression_pred_1.shape

    # compute correlation between GT and pred of top 50 genes ranked by mean
    def compute_corr(expression_gt, matched_spot_expression_pred, top_k=50, qc_idx=None):
        #cells are in columns, genes are in rows
        if qc_idx is not None:
            expression_gt = expression_gt[:,qc_idx]
            matched_spot_expression_pred = matched_spot_expression_pred[:,qc_idx]
        mean = np.mean(expression_gt, axis=1)
        ind = np.argpartition(mean, -top_k)[-top_k:]
        corr = np.zeros(top_k)
        for i in range(top_k):
            corr[i] = np.corrcoef(expression_gt[ind[i],:], matched_spot_expression_pred[ind[i],:])[0,1]
        return np.mean(corr)

    def compute_corr_marker(expression_gt, matched_spot_expression_pred, gene_idx=None, return_list=False):
        #cells are in columns, genes are in rows
        if gene_idx is not None:
            expression_gt = expression_gt[gene_idx]
            matched_spot_expression_pred = matched_spot_expression_pred[gene_idx]
        # mean = np.mean(expression_gt, axis=1)
        ind = np.arange(expression_gt.shape[0])
        corr = np.zeros(expression_gt.shape[0])
        for i in range(len(corr)):
            corr[i] = np.corrcoef(expression_gt[ind[i],:], matched_spot_expression_pred[ind[i],:])[0,1]
        return np.mean(corr) if not return_list else corr



    # # I think this means ranked by mean expression
    # print("mean correlation of top 50 genes ranked by mean")
    # print("GT vs BLEEP: ", compute_corr(expression_gt, matched_spot_expression_pred_1, top_k=50))
    # with open("hvgs_union.pickle", 'rb') as a:
    #     hvg_union = pickle.load(a)

    # gene_list = ["COX6C","TTLL12", "HSP90AB1", 
    #        "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

    # hvg_union[gene_list] = True

    # gene_idx = hvg_union[hvg_union].index.get_indexer(gene_list)

    # print("mean correlation of 12 marker genes")
    # print("GT vs BLEEP: ", compute_corr_marker(expression_gt, matched_spot_expression_pred_1, gene_idx=gene_idx, return_list=False))
    gene_list_path = "../../scripts/gene_list.pkl"
    with open(gene_list_path, 'rb') as f:
        gene_list = pickle.load(f)
    gene_idx = np.arange(len(gene_list))

    cor_pearson = compute_corr_marker(expression_gt, matched_spot_expression_pred_1, gene_idx=gene_idx, return_list=True)
    df = pd.DataFrame(zip(gene_list, cor_pearson, *list(zip(*[(samps[fold],"BLEEP") for i in range(len(gene_list))]))),
            columns = ["Gene", "Pearson correlation","Slide", "Method"])
    df.to_csv(f"../../results/her2st_new/bleep_cor_{samps[fold]}.csv")


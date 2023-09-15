""" Integrate two visium datasets """
data_dir1 = "./Alex_NatGen_6BreastCancer/"
data_dir2 = "./breast_cancer_10x_visium/"

samps1 = ["1142243F", "CID4290", "CID4465", "CID44971", "CID4535", "1160920F"]
samps2 = ["block1", "block2", "FFPE",]

sampsall = samps1 + samps2
samples1 = {i:data_dir1 + i for i in samps1}
samples2 = {i:data_dir2 + i for i in samps2}

# Marker gene list
gene_list = ["COX6C","TTLL12", "HSP90AB1", "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

# # Load windowed dataset
import pickle
with open('10x_visium_dataset_without_window.pickle', 'rb') as f:
    adata_dict0 = pickle.load(f)
    
# For testing
from data_vit import ViT_Anndata
def dataset_wrap(fold = 0, dataloader= True):
    test_sample = sampsall[fold]
    print(f"Test sample: {test_sample}")
    test_sample_orig = sampsall[fold] # Split one sample as test sample
    te_name = list(set([i for i in list(adata_dict0.keys()) if test_sample in i]))
    testset = ViT_Anndata(adata_dict = adata_dict0, train_set = te_name, gene_list = gene_list, train=True, flatten=False, ori=True, prune='NA', neighs=4, )
    test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
    if dataloader==True:
        return test_loader
    else:
        return testset

from scipy.stats import pearsonr,spearmanr
def get_R(data1,data2,dim=1,func=pearsonr):
    adata1=data1.X
    adata2=data2.X
    r1,p1=[],[]
    for g in range(data1.shape[dim]):
        if dim==1:
            r,pv=func(adata1[:,g],adata2[:,g])
        elif dim==0:
            r,pv=func(adata1[g,:],adata2[g,:])
        r1.append(r)
        p1.append(pv)
    r1=np.array(r1)
    p1=np.array(p1)
    return r1,p1

"""For debuging only"""
import gc
from data_vit import ViT_Anndata
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time

start_time = time.time()
"""Training loops"""
seed=12000
epochs=50
dim=1024
name = "efficient"
ft_dict = {}
for fold in range(9):
    """print sample name"""
    te_name = sampsall[fold]
    print(te_name)
    """Reproducibility"""
    setup_seed(seed)

    """Load dataset"""
    test_loader = dataset_wrap(fold = fold, dataloader= True)
    tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomGrayscale(0.1),
            transforms.RandomRotation(90),
            transforms.RandomHorizontalFlip(0.2),
        ])
    name="resnet"
    """ResNet50 Backbone from VICReg"""
    model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    for patch, *_ in test_loader:
        patch = tf(patch.squeeze(0))
        emb = model(patch)
        ft_dict[te_name] = emb.detach().cpu()

import pickle
# Open the file in binary mode
with open(f'./Embedding/{name}_inference.pkl', 'rb') as file:
      
    # Call load method to deserialze
    ft_dict = pickle.load(file)
  
    print(ft_dict)
        
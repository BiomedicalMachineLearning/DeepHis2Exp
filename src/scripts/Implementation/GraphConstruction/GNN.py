# %%
# !pip install torch_geometric

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv, GATConv, GINEConv
from torch_geometric.nn import GINConv, GCNConv, BatchNorm, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

# Build graph using spatial coordinates (SLG), gene expression similarity (GAG) or image features (HSG)
def PCA_process(X, nps):
    # Dimension reduction by pca
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     # It is same as pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC

def features_construct_graph(features, k=4, pca=None, mode="connectivity", metric="cosine", n_jobs=-1):
    """
    Calculate histological similarity graph directly use patch image(cell/spot level) similarity.
    The weight is from gene expression similarity between cells.
    k is the number of nearest nodes, if k=0 we consider all nodes.
    if mode = connectivity, it will return adjancent matrix.
    if mode = distance, it will return disatnce matrix.

    Test codes:
    img_features = torch.randn(10, 512) # 10 is number of cells, 512 is the dimension of image embeddings
    exp_features = torch.randn(10, 300) # 10 is number of cells, 300 is the dimension of gene expression
    HSG = features_construct_graph(img_features, k=4)
    GAG = features_construct_graph(exp_features, k=4)
    """
    from sklearn.neighbors import kneighbors_graph
    print("start features construct graph")
    features = PCA_process(features, nps=pca) if pca is not None else features
    k = features.shape[0]-1 if k == 0 else k
    print("k: ", k)
    print("features_construct_graph features", features.shape)
    A = kneighbors_graph(features, k, mode=mode, metric=metric, include_self="auto", n_jobs=n_jobs).toarray()
    A = torch.Tensor(A)
    return A

def distance2adj(adj):
    # Convert distance matrix to adjacency matrix
    distance_threshold = torch.mean(adj) + torch.std(adj)
    print("distance boundary:", distance_threshold)
    adj = (adj >= distance_threshold).to(torch.float32)
    return adj

def distance_matrix_to_edge_index_and_weight(distance_matrix, threshold):
    # Create binary mask based on the threshold
    adjacency_matrix = (distance_matrix <= threshold).float()

    # Find non-zero indices (edge indices)
    edge_indices = adjacency_matrix.nonzero(as_tuple=False).t()

    # Extract non-zero values as edge weights
    edge_weights = distance_matrix[edge_indices[0], edge_indices[1]]

    return edge_indices, edge_weights

def adjancency_matrix_to_edge_index_and_weight(adjancency_matrix):
    # Find non-zero indices (edge indices)
    edge_indices = adjancency_matrix.nonzero(as_tuple=False).t()
    return edge_indices


# %%
# torch_geometric Version
class GNN(torch.nn.Module):
    def __init__(self, num_feature, nhid=256, residual=True, gnn="GIN",
                 avgpool=True, maxpool=True, channelattn=True, 
                 lstm=True, dropout=0.2,):
        super(GNN, self).__init__()
        """
        num_feature is input feature dimension
        num_gene is output dimension
        nhid is hidden dimension
        residual is whether to use residual connection
        avgpool is whether to use average pooling
        maxpool is whether to use max pooling
        lstm is whether to use lstm
        channel is whether to use channel attention
        dropout is dropout rate
        """
        self.residual=residual
        self.avg=avgpool
        self.max=maxpool
        self.channel=channelattn
        self.lstm = lstm
        if gnn == "GIN":
            self.conv1 = GINConv(Seq(Lin(num_feature, nhid), ReLU(), Lin(nhid, nhid)))
            self.conv2 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
            self.conv3 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
            self.conv4 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        elif gnn == "GAT":
            self.conv1 = GATConv(num_feature, nhid, heads=2, concat=False)
            self.conv2 = GATConv(nhid, nhid, heads=2, concat=False)
            self.conv3 = GATConv(nhid, nhid, heads=2, concat=False)
            self.conv4 = GATConv(nhid, nhid, heads=2, concat=False)
        elif gnn == "GCN-mean":
            aggr="mean"
            self.conv1 = GCNConv(int(num_feature), nhid, aggr=aggr, normalize=True)
            self.conv2 = GCNConv(nhid, nhid, aggr=aggr, normalize=True)
            self.conv3 = GCNConv(nhid, nhid, aggr=aggr, normalize=True)
            self.conv4 = GCNConv(nhid, nhid, aggr=aggr, normalize=True)
        elif gnn == "GCN-max":
            aggr="max"
            self.conv1 = GCNConv(int(num_feature), nhid, aggr=aggr, normalize=True)
            self.conv2 = GCNConv(nhid, nhid, aggr=aggr, normalize=True)
            self.conv3 = GCNConv(nhid, nhid, aggr=aggr, normalize=True)
            self.conv4 = GCNConv(nhid, nhid, aggr=aggr, normalize=True)
        self.mlp1 = nn.Linear(nhid, nhid)
        self.dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(nhid,nhid,2)

    def forward(self, x, adj):
        # Convert adjancency matrix to edge_index
        edge_index = adjancency_matrix_to_edge_index_and_weight(adj).to(x.device)
        
        # convolution->batch normalization->relu->mean&max pooling
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        x4 = F.relu(self.conv4(x3, edge_index))
        x4 = self.dropout(x4)
        x_out = x4

        # Residual connection
        if self.residual:
            x2 = x2 + x1
            x3 = x3 + x2
            x4 = x4 + x3
            x_sum = x1 + x2 + x3 + x4
            x_out = x_sum

        # Global Avgpooling
        if self.avg:
            x_avg = torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0).mean(dim=0)
            x_out = x
        # Global Maxpooling
        if self.max:
            x_max = torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0).max(dim=0)[0]
            x_out = x_max
        # LSTM
        if self.lstm:
            x_lstm = self.LSTM(torch.cat([x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0))[0].mean(dim=0)
            x_out = x_lstm
        # Channel attention
        if self.channel:
            x_channel = F.relu(self.mlp1(x_avg)+self.mlp1(x_max)+self.mlp1(x_lstm))
            x_out = x_channel
        return x_out

# %%
class GNN_HSG(torch.nn.Module):
    def __init__(self, num_feature, nhid=256, residual=True, gnn="GIN",
                 avgpool=True, maxpool=True, channelattn=True, 
                 lstm=True, dropout=0.2,):
        super(GNN_HSG, self).__init__()
        """
        num_feature is input feature dimension
        num_gene is output dimension
        nhid is hidden dimensions
        residual is whether to use residual connection
        avgpool is whether to use average pooling
        maxpool is whether to use max pooling
        lstm is whether to use lstm
        channel is whether to use channel attention
        dropout is dropout rate
        """
        self.residual=residual
        self.avg=avgpool
        self.max=maxpool
        self.channel=channelattn
        self.lstm = lstm
        if gnn == "GIN":
            self.conv1 = GINConv(Seq(Lin(num_feature, nhid), ReLU(), Lin(nhid, nhid)))
            self.conv2 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
            self.conv3 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
            self.conv4 = GINConv(Seq(Lin(nhid, nhid), ReLU(), Lin(nhid, nhid)))
        elif gnn == "GAT":
            self.conv1 = GATConv(num_feature, nhid)
            self.conv2 = GATConv(nhid, nhid)
            self.conv3 = GATConv(nhid, nhid)
            self.conv4 = GATConv(nhid, nhid)
        elif gnn == "GCN":
            self.conv1 = GCNConv(int(num_feature), nhid)
            self.conv2 = GCNConv(nhid, nhid)
            self.conv3 = GCNConv(nhid, nhid)
            self.conv4 = GCNConv(nhid, nhid)
        self.mlp1 = nn.Linear(nhid, nhid)
        self.dropout = nn.Dropout(dropout)
        self.LSTM = nn.LSTM(nhid,nhid,2)

    def forward(self, x):
        # Construct graph using image features
        edge_index = self.HSG_construction(x)
        
        # convolution->batch normalization->relu->mean&max pooling
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.dropout(x1)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.dropout(x2)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = self.dropout(x3)
        x4 = F.relu(self.conv4(x3, edge_index))
        x4 = self.dropout(x4)

        # Residual connection
        if self.residual:
            x2 = x2 + x1
            x3 = x3 + x2
            x4 = x4 + x3
            x_sum = x1 + x2 + x3 + x4

        # Average pooling
        if self.avg:
            x_avg = torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0).mean(dim=0)
        # Max pooling
        if self.max:
            x_max = torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0).max(dim=0)[0]
        # LSTM
        if self.lstm:
            x_lstm = self.LSTM(torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0))[0].mean(dim=0)
        # Channel attention
        if self.channel:
            x = F.relu(self.mlp1(x_avg)+self.mlp1(x_max)+self.mlp1(x_lstm))
            x_out = x
        return x_out
    
    def HSG_construction(self, x):
        # Construct graph using image features
        # x is the input features
        HSG = torch.as_tensor(features_construct_graph(x.detach().cpu().numpy(), k=0, pca=None, mode="connectivity", metric="cosine", n_jobs=-1)).detach().requires_grad_(True).clone().to(x.device)
        return adjancency_matrix_to_edge_index_and_weight(HSG).requires_grad_(True).to(x.device)

# %%
# DGL Version
import dgl
import numpy as np
from dgl.nn import SAGEConv, GATConv, GINEConv

class GNN_dgl(torch.nn.Module):
    def __init__(self, num_feature, nhid=256, residual=False, gnn="GIN",
                 avgpool=False, maxpool=False, channelattn=False, feat_drop=0.2,
                 lstm=True,):
        super(GNN_dgl, self).__init__()
        """
        num_feature is input feature dimension
        num_gene is output dimension
        nhid is hidden dimension
        residual is whether to use residual connection
        avgpool is whether to use average pooling
        maxpool is whether to use max pooling
        lstm is whether to use lstm
        channel is whether to use channel attention
        dropout is dropout rate
        """
        self.residual=residual
        self.avg=avgpool
        self.max=maxpool
        self.channel=channelattn
        self.lstm = lstm
        if gnn == "GIN":
            self.conv1 = GINEConv(nn.Linear(num_feature, nhid), learn_eps=True)
            self.conv2 = GINEConv(nn.Linear(num_feature, nhid), learn_eps=True)
            self.conv3 = GINEConv(nn.Linear(num_feature, nhid), learn_eps=True)
            self.conv4 = GINEConv(nn.Linear(num_feature, nhid), learn_eps=True)

        elif gnn == "GAT":
            self.conv1 = GATConv(in_feats=num_feature, out_feats=nhid, num_heads=2, feat_drop=feat_drop, attn_drop=0, activation=nn.ReLU())
            self.conv2 = GATConv(nhid, nhid, heads=2, concat=False)
            self.conv3 = GATConv(nhid, nhid, heads=2, concat=False)
            self.conv4 = GATConv(nhid, nhid, heads=2, concat=False)

        elif gnn == "GCN-mean":
            aggr="mean"
            self.conv1 = SAGEConv(in_feats=num_feature, out_feats=nhid, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv2 = SAGEConv(in_feats=num_feature, out_feats=nhid, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv3 = SAGEConv(in_feats=num_feature, out_feats=nhid, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv4 = SAGEConv(in_feats=num_feature, out_feats=nhid, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())

        elif gnn == "GCN-max":
            aggr="max"
            self.conv1 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv2 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv3 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv4 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
        
        elif gnn == "GCN-pool":
            aggr="pool"
            self.conv1 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv2 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv3 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv4 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
        
        elif gnn == "GCN-lstm":
            aggr="lstm"
            self.conv1 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv2 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv3 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())
            self.conv4 = SAGEConv(in_feats=1024, out_feats=1024, aggregator_type=aggr, feat_drop=feat_drop, norm=None, activation=nn.ReLU())

        self.mlp1 = nn.Linear(nhid, nhid)
        self.LSTM = nn.LSTM(nhid,nhid,2)

    def forward(self, edge_index, x, edge_weight):
        # convolution->batch normalization->relu->mean&max pooling
        x1 = self.conv1(edge_index, x, edge_weight)
        x2 = self.conv2(edge_index, x1, edge_weight)
        x3 = self.conv3(edge_index, x2, edge_weight)
        x4 = self.conv4(edge_index, x3, edge_weight)

        # Residual connection
        if self.residual:
            x2 = x2 + x1
            x3 = x3 + x2
            x4 = x4 + x3
            x_sum = x1 + x2 + x3 + x4
            x_out = x_sum

        # Global Avgpooling
        if self.avg:
            x_avg = torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0).mean(dim=0)
            x_out = x
        # Global Maxpooling
        if self.max:
            x_max = torch.cat([x_sum.unsqueeze(0), x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0).max(dim=0)[0]
            x_out = x_max
        # LSTM
        if self.lstm:
            x_lstm = self.LSTM(torch.cat([x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)], dim=0))[0].mean(dim=0)
            x_out = x_lstm
        # Channel attention
        if self.channel:
            x_channel = F.relu(self.mlp1(x_avg)+self.mlp1(x_max)+self.mlp1(x_lstm))
            x_out = x_channel
        return x_out

# %%
"""
Node masking or feature masking
"""
def node_masking(x, dropout_rate):
    """
    x is the node features of shape (number_of_nodes, hidden_dim)
    dropout_rate is the probability of dropping out a node (feature)
    Reference: https://arxiv.org/pdf/2001.07524.pdf
    """

    # Mask the rows of x with dropout_rate
    # Creating a mask to randomly drop nodes (features) in the input tensor x based on dropout_rate
    print(f"Input features:\n{x}")

    # Initializing a tensor drop_node_mask with values of 1 - dropout_rate for each feature in x
    drop_ft_mask = x.new_full((x.size(0),), 1 - dropout_rate, dtype=torch.float)
    print(f"Dropout rate:\n{drop_ft_mask}")

    # Applying Bernoulli distribution to randomly generate binary values (0 or 1) for each feature in drop_node_mask
    drop_ft_mask = torch.bernoulli(drop_ft_mask)
    print(f"Feature mask:\n{drop_ft_mask}")

    # Reshaping the drop_node_mask to have a shape of (1, number_of_features) to match the shape of x
    drop_ft_mask = torch.reshape(drop_ft_mask, (1, drop_ft_mask.shape[0]))

    print(x.shape, drop_ft_mask.shape)
    # Element-wise multiplication of the input tensor x with the drop_node_mask to dropout (zero out) random features
    drop_ft_mask = drop_ft_mask.T * x
    print(f"Masked features:\n{drop_ft_mask}")

    return drop_ft_mask

############################################################################################################
def feature_masking(x, dropout_rate):
    # Mask the columns of x with dropout_rate
    # Creating a mask to randomly drop nodes (features) in the input tensor x based on dropout_rate
    print(f"Input features:\n{x}")

    # Initializing a tensor drop_node_mask with values of 1 - dropout_rate for each feature in x
    drop_ft_mask = x.new_full((x.size(1),), 1 - dropout_rate, dtype=torch.float)
    print(f"Dropout rate:\n{drop_ft_mask}")

    # Applying Bernoulli distribution to randomly generate binary values (0 or 1) for each feature in drop_node_mask
    drop_ft_mask = torch.bernoulli(drop_ft_mask)
    print(f"Feature mask:\n{drop_ft_mask}")

    # Reshaping the drop_node_mask to have a shape of (1, number_of_features) to match the shape of x
    drop_ft_mask = torch.reshape(drop_ft_mask, (1, drop_ft_mask.shape[0]))

    print(x.shape, drop_ft_mask.shape)
    # Element-wise multiplication of the input tensor x with the drop_node_mask to dropout (zero out) random features
    drop_ft_mask = x * drop_ft_mask
    print(f"Masked features:\n{drop_ft_mask}")
    return drop_ft_mask




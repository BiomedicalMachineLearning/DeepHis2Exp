import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import seaborn as sns
import matplotlib.pyplot as plt

class encoder1(nn.Module):
    def __init__(self,):
        super().__init__()
        
        """ResNet50 Backbone from VICReg"""
        model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
        
        """ Freeze the weight of Backbone """
        for param in model.parameters():
            param.requires_grad = False
            
        self.model = model
        
    def forward(self, x):
        x = self.model(x)
        return x

class encoder2(nn.Module):
    def __init__(self,):
        super().__init__()
        
        """Swin Transformer Backbone """
        model = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)

        """ Freeze the weight of Backbone """
        for param in model.parameters():
            param.requires_grad = False    
            
        self.model = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, num_heads=2, concat_heads=True, alpha=0.2):
        """
        Args:
            c_in: Dimensionality of input features
            c_out: Dimensionality of output features
            num_heads: Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads: If True, the output of the different heads is concatenated instead of averaged.
            alpha: Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Args:
            node_feats: Input features of the node. Shape: [batch_size, c_in]
            adj_matrix: Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs: If True, the attention weights are printed during the forward pass
                               (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        # Returns indices where the adjacency matrix is not 0 => edges
        edges = adj_matrix.nonzero(as_tuple=False)
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat(
            [
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )  # Index select returns a tensor with node_feats_flat being indexed at the desired positions

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats, attn_probs.permute(0, 3, 1, 2)


class GAT(nn.Module):
    def __init__(self, dim=1024, num_layer=3):
        super().__init__()
        """
        node_feature:[N, dim]
        adj:[N, N]
        """
        self.GAT = nn.ModuleList([GATLayer(c_in=dim, c_out=dim) for i in range(num_layer)])
        self.jknet = nn.Sequential(
            nn.LSTM(dim,dim,2),
        )
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, adj):
        x, adj = x.unsqueeze(0), adj.unsqueeze(0)
        jk = []
        for layer in self.GAT:
            x, attention_map = layer(x,adj)
            x = self.dropout(x)
            jk.append(x)
        x = torch.cat(jk,0)
        x = self.jknet(x)[0].mean(0)
        return x, attention_map
    
    


class Attention(nn.Module):
    """
    args : 
    in_dim : Dimensionality of representation
    

    """
    def __init__(self, in_dim = 1024):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim, bias=False)
        self.key = nn.Linear(in_dim, in_dim, bias=False)
        self.value = nn.Linear(in_dim, in_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x1, x2):
        query = self.query(x1) # shape: (batch_size, seq_len, in_dim)
        key = self.key(x2).T # shape: (batch_size, seq_len, in_dim*2)
        value = self.value(x2) # shape: (batch_size, seq_len, in_dim*2)
        energy = torch.mm(query, key) # shape: (batch_size, seq_len1, seq_len2)
        attention = self.softmax(energy) # shape: (batch_size, seq_len1, seq_len2)
        attended = torch.mm(attention, value) # shape: (batch_size, seq_len1, in_dim*2)
        return attended, attention

def visualize_attention_map(attention_weights):
    # Convert attention weights to a numpy array
    attention_weights_np = attention_weights.detach().numpy()

    # Plot the attention map as a heatmap
#     plt.figure(figsize=(10, 8))
    ax = sns.heatmap(attention_weights_np, cmap="YlGnBu", annot=True, fmt=".2f")
    ax.set_xlabel("Input 2")
    ax.set_ylabel("Input 1")
    plt.title("Attention Map")
    plt.show()

# if __name__ == "__main__":
#     x1 = torch.rand(100,1024)
#     x2 = torch.rand(100,1024)

#     model = Attention(in_dim=1024)
#     cat_ft, weight = model(x1,x2)
    
#     print(cat_ft, cat_ft.shape)
#     print(weight, weight.shape)
#     # Count the number of trainable parameters
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     # Count the total number of parameters
#     total_params = sum(p.numel() for p in model.parameters())

#     print(f"Trainable Parameters: {trainable_params}")
#     print(f"Total Parameters: {total_params}")

#     # Assuming you have already executed the code to obtain `cat_ft`
#     visualize_attention_map(weight)


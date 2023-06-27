import torch
import torch.nn as nn
import torch.nn.functional as F

from Loss_fn import *

""" Img Encoder """
class VICReg(nn.Module):
    def __init__(self,):
        super().__init__()
        
        """ResNet50 Backbone from VICReg"""
        model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
        
        """ Freeze the weight of Backbone """
        for param in model.parameters():
            param.requires_grad = False
            
        self.model = model
        
    def forward(self, x):
        x = self.model(x.squeeze())
        return x

""" Gene Expression Encoder """
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims = 2048):
        super(VariationalEncoder, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.linear1 = nn.Linear(785, 1024)
        self.linear2 = nn.Linear(1024, latent_dims)
        self.linear3 = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device) # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x.squeeze(), start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

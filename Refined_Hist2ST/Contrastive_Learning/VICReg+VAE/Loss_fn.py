import torch
import torch.nn as nn
import torch.nn.functional as F

def VICReg_Loss(img_h, exp_h, alpha=25, beta=25, gamma=1):
    dim=img_h.shape[-1]
    # Invariance
    Invariance = F.mse_loss(img_h, exp_h)
#     print(f"Invariance:{Invariance}")

    # Variance
    x = img_h - img_h.mean(dim=0)
    y = exp_h - exp_h.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    Variance = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
#     print(f"Variance:{Variance}")

    # Covariance
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        x = x.flatten()[:-1]
        x = x.view(n - 1, n + 1)
        x = x[:, 1:]
        x = x.flatten()
        return x

    cov_x = (x.T @ x) 
    cov_y = (y.T @ y) 
    Covariance = off_diagonal(cov_x).pow_(2).sum().div(dim) + off_diagonal(cov_y).pow_(2).sum().div(dim)
#     print(f"Covariance:{Covariance}")

    # Total loss
    loss = alpha*Invariance + beta*Variance + gamma*Covariance
#     print(f"Total Loss:{loss}")
    return loss


def CosineSimilarity(img_h, exp_h):
    Cos_sim = nn.CosineSimilarity(eps=1e-6)
    Cos_Sim = Cos_sim(img_h, exp_h).mean()
#     print(f" CosineSimilarity(mean):{Cos_Sim}")
    return Cos_Sim


def PairwiseDistance(img_h, exp_h, p=2):
    if p==1:
        # L1 Norm
        pair_sim = nn.PairwiseDistance(p=p, eps=1e-3)
        Pair_Sim = pair_sim(img_h, exp_h).mean()
#         print(f" Pairwise Similarity(L1 Norm):{Pair_Sim}")
    else:
        # L2 Norm
        pair_sim = nn.PairwiseDistance(p=p, eps=1e-3)
        Pair_Sim = pair_sim(img_h, exp_h).mean()
#         print(f" Pairwise Similarity(L2 Norm):{Pair_Sim}")
    return Pair_Sim


def KL_Div(img_h, exp_h, reduction = "batchmean"):
    if reduction == "batchmean":
        # Batchmean
        KL_loss = nn.KLDivLoss(reduction = "batchmean")
        KL_Loss = KL_loss(img_h, exp_h)
#         print(f"Pointwise KL-divergence(batchmean):{KL_Loss}")
    elif reduction == "mean":
        # Mean
        KL_loss = nn.KLDivLoss(reduction = "mean")
        KL_Loss = KL_loss(img_h, exp_h)
#         print(f"Pointwise KL-divergence(mean):{KL_Loss}")
    elif reduction == "sum":
        # Sum
        KL_loss = nn.KLDivLoss(reduction = "sum")
        KL_Loss = KL_loss(img_h, exp_h)
#         print(f"Pointwise KL-divergence(sum):{KL_Loss}")
    return KL_Loss
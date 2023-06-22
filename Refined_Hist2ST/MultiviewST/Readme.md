MultiviewST is a deep learning model, which using attention mechanism to integrate latent space from different feature extractors.

In this case, we integrate the the latent space from VICReg(official verision) and Swin-Transformer(torchvision).

The extracted features would be fed into GAT, the message-passing processing would be helpful to share info between neibouring spots.


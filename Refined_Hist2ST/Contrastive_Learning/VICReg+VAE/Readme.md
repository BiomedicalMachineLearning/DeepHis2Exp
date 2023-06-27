This self-supervised learning model utilizes VICReg encoder to extract embedding from H&E images and leverages VAE to extract embedding from gene expression.

The contrastive loss including 3 terms, variance, invariance and covariance respectively.

Total Loss = alpha * variance + beta * invariance + gamma * covariance

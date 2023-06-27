This self-supervised learning model utilizes VICReg encoder to extract embedding from H&E images and leverages VAE to extract embedding from gene expression.

The contrastive loss includes 3 terms, variance, invariance and covariance respectively.

Invariance: the mean square distance between the embedding vectors.

Variance: a hinge loss to maintain the standard deviation (over a batch) of each embedding variable above a given threshold. This term forces the embedding vectors of samples
within a batch to be different.

Covariance: a term that attracts the covariances (over a batch) between every pair of
(centred) embedding variables towards zero. This term decorrelates the variables of each
embedding and prevents an informational collapse in which the variables would vary
together or be highly correlated.

Total Loss = alpha * variance + beta * invariance + gamma * covariance

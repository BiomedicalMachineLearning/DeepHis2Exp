* Regclass
    * Hist2ST backbone + MLP classifier
    * Regulize regression task via classification loss.
* Hist2ST with pretrained image encoder
    * Replace the convmixer module in Hist2ST with pre-trained model. (Feature extraction only, without finetuning)
* Simplify Hist2ST via GAT
    * Integate pretrained model and graph attention neural network to extract features.
    
![Hist2STVariants](https://github.com/BiomedicalMachineLearning/DeepHis2Exp/blob/main/Figures/S7.png)
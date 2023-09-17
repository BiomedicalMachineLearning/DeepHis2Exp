# Hist2ST Variants
* Regclass
    * Hist2ST backbone + MLP classifier
    * Regulize regression task via classification loss.
* Hist2ST with pretrained image encoder
    * Replace the convmixer module in Hist2ST with pre-trained model. (Feature extraction only, without finetuning)
* Simplify Hist2ST via GAT
    * Integate pretrained model and graph attention neural network to extract features.
    
Because the needed environment is similar to Hist2ST. 
To simplify the codes, you can cd ./src/models/Hist2ST/ to run the code.

Note: The dataset that Regclass used is slight different from Hist2ST. It includes cell type annoations as classification labels, the details were shown in ./Regclass/data_Regclass.py

![Hist2STVariants](https://github.com/BiomedicalMachineLearning/DeepHis2Exp/blob/main/Figures/S7.png)
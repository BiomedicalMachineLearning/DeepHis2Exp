# About

The following models were trained on Visium breast cancer spot tiles (about 50k images). Both are Pytorch-based.

- ViTMAE: https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining

- VICReg (resnet): https://github.com/facebookresearch/vicreg

The scripts in the github can used directly to train models yourself - you just need to pass in a directory of images. See the example code in the links.

The models have not been tested for downstream performance yet.


```python
%cd ~/Q2051/STimage_project/pretrained_backbones/
%ls
```

    /afm03/Q2/Q2051/STimage_project/pretrained_backbones
    [0m[01;34mmae[0m/  [01;34mvicreg[0m/


## MAE


```python
%ls mae
```

    [0m[01;34mmae-055-100e-005warm-9lr[0m/  [01;34mmae-055-e3-100e-01warm-albu[0m/
    [01;34mmae-055-e3-100e-01warm[0m/    [01;34moutputs-nocrop-e3[0m/


The hidden/output dim is 768.

I suggest using one of `mae-055-100e-005warm-9lr` or `mae-055-e3-100e-01warm`.

Example code to load the MAE model:


```python
from transformers import (
    ViTImageProcessor,
    ViTModel,
    ViTForImageClassification,
    ViTMAEForPreTraining,
)

modeldir = "./mae/mae-055-100e-005warm-9lr"
feature_extractor = ViTImageProcessor.from_pretrained(modeldir)
model = ViTModel.from_pretrained(modeldir)
```

## VICReg


```python
%ls vicreg
```

    [0m[01;34mvicreg-exp-0.5-pre-nocrop[0m/   [01;34mvicreg-exp-0.7-nopre-nocrop-albu200512[0m/
    [01;34mvicreg-exp-0.7-nocrop-albu[0m/  [01;34mvicreg-exp-0.7-pre-nocrop-albu200512[0m/


The hidden/output dim is 2048.

I suggest using one of `vicreg-exp-0.7-nopre-nocrop-albu200512` or `vicreg-exp-0.7-pre-nocrop-albu200512`.

Example code to load the resnet model:


```python
!git clone https://github.com/facebookresearch/vicreg vicreg_source
```

    Cloning into 'vicreg_source'...
    remote: Enumerating objects: 38, done.[K
    remote: Counting objects: 100% (25/25), done.[K
    remote: Compressing objects: 100% (17/17), done.[K
    remote: Total 38 (delta 14), reused 8 (delta 8), pack-reused 13[K
    Unpacking objects: 100% (38/38), done.



```python
import sys
sys.path.append('./vicreg_source/')
import resnet
import torch

backbone, embedding = resnet.__dict__["resnet50"](zero_init_residual=True)
state_dict = torch.load("./vicreg/vicreg-exp-0.7-nopre-nocrop-albu200512/resnet50.pth", map_location="cpu")
backbone.load_state_dict(state_dict, strict=False)
```




    <All keys matched successfully>



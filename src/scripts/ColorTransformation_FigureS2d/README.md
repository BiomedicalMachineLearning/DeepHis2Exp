# Data transformation methods
* Reinhard color normalization
* H&E intensity color augmentation
* RandStainNA color transformation

# Tutorial
Before reproduce the Hist2ST model under different color transformation methods, you have to open *Image_aug&norm.ipynb* to generate transformed tiles.
[stainlib](https://github.com/sebastianffx/stainlib) and [RandStainNA](https://github.com/yiqings/RandStainNA) are provide some tutorials that how to install and implement their tools.
After generating transformed tiles, you can run the scripts by the command below.

python Hist2ST-color-transformation.py --ImgProcess Reinhard

# Acknowledgement
We use Stainlib to implement Reinhard color normalization and H&E intensity color augmentation.
<https://github.com/sebastianffx/stainlib>

RandStainNA was implemented by <https://github.com/yiqings/RandStainNA>
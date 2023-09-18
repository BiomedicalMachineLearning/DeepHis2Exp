# Assessment of different augmentaion methods
* ST-Net
* HisToGene
* Hist2ST
* DeepSpaCE
* STimage
* BLEEP

## Summary of data preprocessing methods
| Model       | Tile size (pixels) | Data augmentation                                       | Gene expression Pre-processing                   |
|-------------|---------------------|--------------------------------------------------------|---------------------------------------------------|
| ST-Net      | 224 x 224           | Random Rotation & Flipping                            | Log Transformation                               |
| HisToGene   | 112 x 112           | Random Rotation & Flipping + ColorJitter              | Normalization + Log Transformation                |
| Hist2ST     | 112 x 112           | Random Rotation & Flipping + ColorJitter             | Normalization + Log Transformation                |
|             |                     | (Self-distillation strategy)                           |                                                   |
| STimage     | 299 x 299           | Color Normalization (Vahadane) + Random one of       | Log Transformation                               |
|             |                     | flipping, cropping, noise addition, blurring,        | Remove tiles with low tissue coverage (< 70%)      |
|             |                     | distortion, contrast adjustment, colour-shifting +   |                                                   |
|             |                     | Remove tiles with low tissue coverage (< 70%)        |                                                   |
| DeepSpaCE   | 224 x 224           | Remove tiles with high RGB values                    | SCTransform + MinMax Scaling                      |
| BLEEP       | 224 x 224           | Random rotation & flip                                 | Normalization + Log Transformation                |

## Related codes
The related codes were highlight here.
```python
import albumentations as albu
import torchvision.transforms as tf

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
aug_software = {}

methods = ["ST-net", "HistoGene", "Hist2ST", "STimage", "BLEEP", "randaug", "autoaug"]
for method in methods:
    if method=="ST-net":
        aug_software[method] = tf.Compose([tf.RandomHorizontalFlip(),
                          tf.RandomVerticalFlip(),
                          tf.RandomApply([torchvision.transforms.RandomRotation((90, 90))]), ]) 
        
    elif method=="HistoGene":
        aug_software[method] = tf.Compose([
            tf.ColorJitter(0.5,0.5,0.5),
            tf.RandomHorizontalFlip(),
            tf.RandomRotation(degrees=180),
        ])
        
    elif method=="Hist2ST":
        aug_software[method] = tf.Compose([
                tf.RandomGrayscale(0.1),
                tf.RandomRotation(90),
                tf.RandomHorizontalFlip(0.2),
            ])
        
    elif method=="DeepSpace":
        aug_software[method] = albu.Compose([
                albu.RandomRotate90(p=0.5),
                albu.Flip(p=0.5),
                albu.Transpose(p=0.5),
                albu.RandomResizedCrop(height=size, width=size, scale=(0.5, 1.0), p=0.5),
                albu.HueSaturationValue(p=0.5),
                albu.ChannelShuffle(p=0.5),
                albu.RGBShift(p=0.5),])
                    
    elif method=="BLEEP":
        # Random flipping and rotations
        angle = random.choice([180, 90, 0, 270])
        aug_software[method] = tf.Compose([
                tf.RandomRotation(angle),
                tf.RandomHorizontalFlip(0.5),
                tf.RandomVerticalFlip(0.5),
            ])
    elif method=="STimage":
        class AddGaussianNoise(object):
            def __init__(self, mean=0., std=1.):
                self.std = std
                self.mean = mean

            def __call__(self, tensor):
                return tensor + torch.randn(tensor.size()) * self.std + self.mean

            def __repr__(self):
                return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
            
        stimage = tf.Compose([
            tf.RandomHorizontalFlip(0.5),
            tf.RandomVerticalFlip(0.5),
            tf.RandomApply([tf.RandomAffine(degrees=(-45,45), scale=(0.8, 1.2)),
                           tf.RandomGrayscale(0.5),
                           tf.GaussianBlur(1, np.random.rand() * 1.9 + 0.1),
                           tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                           ], p=0.5,), ])
        def stimage_aug(x):
            aug_img = stimage(x.to(torch.uint8))
            return aug_img.to(torch.float32)
        aug_software[method] = stimage_aug
        
    elif method=="randaug":
        def randaug(x):
            aug_img = tf.RandAugment()(x.to(torch.uint8))
            return aug_img.to(torch.float32)
        aug_software[method] = randaug
            
    elif method=="autoaug":
        def autoaug(x):
            aug_img = tf.AutoAugment()(x.to(torch.uint8))
            return aug_img.to(torch.float32)
        aug_software[method] = autoaug
```
# U^2-Net (U Square Net)
Implementation of U^2-Net(U Square Net) architecture in TensorFlow for salient object detection and segmentation.

This repository contains the tensorflow implementation of the below paper. The original implementation was done on PyTorch. 

[U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)

The model has been trained on the ECSSD dataset. The main model weights are available [here](https://drive.google.com/file/d/1-K9lMWTWN8oXD3z2EEOhdIQ57iapIWv3/view?usp=sharing) and the lighter model weights are available [here](https://drive.google.com/file/d/1aPlkXTOsuZrx_HT9cXBC0I_hewa74Ns8/view?usp=sharing).

Predicted mask                    | Original mask:

![pic1](https://raw.githubusercontent.com/Akhilesh64/U-2-Net/main/predicted_masks/img1.png)       ![pic2](https://raw.githubusercontent.com/Akhilesh64/U-2-Net/main/predicted_masks/ground_truth.png)

If you find this useful please cite this work using:

```
@InProceedings{Qin_2020_PR,
title = {U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection},
author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Dehghan, Masood and Zaiane, Osmar and Jagersand, Martin},
journal = {Pattern Recognition},
volume = {106},
pages = {107404},
year = {2020}
}
```

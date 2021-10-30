# U^2-Net (U Square Net)
Implementation of U^2-Net(U Square Net) architecture in TensorFlow for salient object detection and segmentation.

This repository contains the tensorflow implementation of the below paper. The original implementation was done on PyTorch. 

[U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)

The model has been trained and tested on the DUTS and ECSSD dataset using tf v2.4.1 and keras v2.4.3. The weights can be downloaded from here : [DUTS_main](https://drive.google.com/file/d/18XYx_1N9s3qQjW74UVC9aD4wgH9FAyOL/view?usp=sharing), [DUTS_light](https://drive.google.com/file/d/1KN-ZCMQ0x0zPAmqkRxZ8kmXb45unZPXE/view?usp=sharing), [ECSSD_main](https://drive.google.com/file/d/1jqmiiKczV1PuJg_zQ7l4Z1FyyzM6VZY4/view?usp=sharing), [ECSSD_light](https://drive.google.com/file/d/1sUJ-RpK18qGVKKS5eg116bpB8xS8Ld7L/view?usp=sharing).

Predicted mask                    | Original mask:

![pic1](https://raw.githubusercontent.com/Akhilesh64/U-2-Net/main/predicted_masks/img1.png)       ![pic2](https://raw.githubusercontent.com/Akhilesh64/U-2-Net/main/predicted_masks/ground_truth.png)

If you find this useful please cite the original paper using:

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


Update (29/11/2020):

Added Scripts used for training on larger datasets like DUTS which could cause memory leak. The updated scripts can be modified according to the dataset used.

# Data Preparing

## Image-to-image translation datasets for stage one

1. Respacing the source and target domain images (both should be gray) with the same XY plane resolutions, and crop/pad to the size of [512, 512, d] in terms of [width, height, depth].

2. Normlizae each 3D images to [0, 1], and extract 2D slices from 3D volumes along depth-axis.

3. Stack the list of 2D slices at zero dimension for the two domains respectively, resulting in 3D tensor with size of [N, 512, 512], and then save them as the follows:

```bash
.
└── DARUNET
    └──datasets
            ├── A_imgs.npy
            └── B_imgs.npy
```

## Semantic Segmentation datasets for stage two
1. Respacing the source and target domain images and labels (both should be gray) with the same XY plane resolutions, and make sure the spacing ratio is about [1, 1, 4] and the size is [512, 512, d] in terms of [width, height, depth].

2. Normlizae each 3D images to [0, 1], rename the paired image and label to ensure the only name difference is 'img' and 'label', and save them as follows
```bash
.
├── TransUNet
    └──datasets
            ├── source_training_npy
            │      ├──2_img.npy
            │      ├──2_label.npy
            │      ├──3_img.npy
            │      └──3_label.npy
            └── target_training_npy
                   ├──5_img.npy
                   ├──5_label.npy
                   ├──8_img.npy
                   └──8_label.npy
```
3. run stage_1.5_i2i_inference.py to generate the transferred source domain images

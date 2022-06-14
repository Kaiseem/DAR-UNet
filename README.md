# DAR-Unet
The codes for the work "A novel 3D unsupervised domain adaptation framework for cross-modality medical image segmentation"(https://ieeexplore.ieee.org/abstract/document/9741336). Meanwhile, the 5th solution in CrossMoDA 2021 challenge.

## 1. Prepare data

- Please go to ["./datasets/README.md"](datasets/README.md) for details.

## 2. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train/Test

- Train stage one image-to-image translation model for style transfer

```bash
python stage_1_i2i_train.py --name sourceAtotargetB
```

- Generate target-like source domain images

```bash
python stage_1.5_i2i_inference.py --ckpt_path YOUR_PATH --source_npy_dirpath SOURCE_PATH --target_npy_dirpath TARGET_PATH --save_npy_dirpath SAVE_PATH --k_means_clusters 6
```

- Train stage two DAR-UNET model for semantice segmentation

```bash
python stage_2_seg_train.py --name experiment --train_dataroot SOURCE2TARGET_PATH --val_dataroot TARGET_VAL_PATH --num_classes NUM_CLASS --epoch_max 100
```


## References
* [MUNIT](https://github.com/NVlabs/MUNIT)
* [Holocrons](https://github.com/frgfm/Holocron)

## Citation

```bibtex
@article{yao2022darunet,
  title={A novel 3D unsupervised domain adaptation framework for cross-modality medical image segmentation},
  author={Yao, Kai and Su, Zixian and Huang, Kaizhu and Yang, Xi and Sun, Jie and Hussain, Amir and Coenen, Frans},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}
@article{dorent2022crossmoda,
  title={CrossMoDA 2021 challenge: Benchmark of Cross-Modality Domain Adaptation techniques for Vestibular Schwnannoma and Cochlea Segmentation},
  author={Dorent, Reuben and Kujawa, Aaron and Ivory, Marina and Bakas, Spyridon and Rieke, Nicola and Joutard, Samuel and Glocker, Ben and Cardoso, Jorge and Modat, Marc and Batmanghelich, Kayhan and others},
  journal={arXiv preprint arXiv:2201.02831},
  year={2022}
}
```

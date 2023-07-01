# DAR-Unet
The codes for the work "A novel 3D unsupervised domain adaptation framework for cross-modality medical image segmentation"(https://ieeexplore.ieee.org/abstract/document/9741336). Meanwhile, our work is the 5th solution in CrossMoDA 2021 challenge (https://arxiv.org/pdf/2201.02831.pdf), which did not utilize multiple nnUnet to ensemble results. 

## 1. Prepare data

- Please go to ["./datasets/README.md"](datasets/README.md) for details.

- The generated multi-style data for segmentation can be directly download here [Google Drive](https://drive.google.com/file/d/1s3IYf69P1WJBBH5YTFSXwoi7LT9gQUS3/view?usp=sharing) [Baidu_Pan](https://pan.baidu.com/s/1P5XAl_xW3oV42fefVLaMXQ?pwd=7so5)

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
If you have any questions, please send us email {kai.yao19, zixian.su20}@student.xjtlu.edu.cn.

If you find our work is useful, please cite our work.

```bibtex
@article{yao2022darunet,
  title={A novel 3D unsupervised domain adaptation framework for cross-modality medical image segmentation},
  author={Yao, Kai and Su, Zixian and Huang, Kaizhu and Yang, Xi and Sun, Jie and Hussain, Amir and Coenen, Frans},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}

@article{dorent2023crossmoda,
  title={CrossMoDA 2021 challenge: Benchmark of cross-modality domain adaptation techniques for vestibular schwannoma and cochlea segmentation},
  author={Dorent, Reuben and Kujawa, Aaron and Ivory, Marina and Bakas, Spyridon and Rieke, Nicola and Joutard, Samuel and Glocker, Ben and Cardoso, Jorge and Modat, Marc and Batmanghelich, Kayhan and others},
  journal={Medical Image Analysis},
  volume={83},
  pages={102628},
  year={2023},
  publisher={Elsevier}
}
```

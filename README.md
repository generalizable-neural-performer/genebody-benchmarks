# GeneBody benchmark - A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose

GeneBody Benchmark reivison of the implementation of paper "A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose", NeurIPS 2021. This repository borrows most of the code from the [original implementation](https://github.com/LemonATsu/A-NeRF).



## Installation
The code is tested with Python3.8, PyTorch == 1.9.0 and CUDA == 10.2. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an anaconda environment:
```
conda create -n anerf python=3.8
conda activate anerf

# install pytorch for your corresponding CUDA environments
pip install torch

# install pytorch3d: note that doing `pip install pytorch3d` directly may install an older version with bugs.
# be sure that you specify the version that matches your CUDA environment. See: https://github.com/facebookresearch/pytorch3d
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html

# install other dependencies
pip install -r requirements.txt
```

## How to Use

### 1. Prepare datasets
Please download the GeneBody *Test10* subset from Onedrive [link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EgWKPko5WXdClIg_zsjDSxwBH7LM4waKyJkWaslC-BVfSQ?e=JaDZdQ), and the dataset is organized as below.
```
├──data/
├───genebody_origin/
  ├───amanda
  ├───barry

then cd in core, run python load_genebody.py
then you can get file like this:   

├──data/
├───genebody/
  ├───amanda_train.h5/
  ├───amanda_test.h5/
  ├───barry_train.h5/
  ├───barry_test.h5/
```

### 2. Train GeneBody
To train on GeneBody sequence, eg. `amanda`, you can run  
```
python run_nerf.py --config configs/genebody/genebody.txt --subject amanda --basedir logs --expname GeneBody_amanda --no_reload
```

### 3. Evaluation
To evaluate our pretrained models on Genebody, run:
```
python run_render.py --nerf_args config/genebody/genebody.txt --ckptpath logs/genebody_amanda/150000.tar --dataset genebody --entry amanda --render_type val --render_res 512 512  --runname genebody_test_amanda
```

### 4. Result in GeneBody sequence:
|  | psnr | ssim | lpsips |
| --- | --- | --- | --- |
| zhuna | 10.139 | 0.405 | 0.310 |
| natacha | 18.429 | 0.383 | 0.240 |
| mahaoran | 21.587 | 0.737 | 0.245 |
| amanda | 16.976 | 0.699 | 0.161 |
| fuzhizhi | 13.898 | 0.371 | 0.205 |
| barry | 20.644 | 0.692 | 0.325 |
| jinyutong | 12.649 | 0.412 | 0.250 |
| joseph_matanda | 9.233 | 0.298 | 0.362 |
| maria | 12.406 | 0.50 | 0.119 |
| soufianou_boubacar_moumouni | 19.70 | 0.578 | 0.202 |
| Average | 15.5661 | 0.5075 | 0.2419 |

A-Nerf's performance in some cases are bad, and can barely render realistic image in most unseen pose.
Here is some [discussion](https://github.com/LemonATsu/A-NeRF/issues/8) with the author.

## Citation
If you find this repo is useful for your research, please cite the following papers
```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
@inproceedings{su2021anerf,
    title={A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose},
    author={Su, Shih-Yang and Yu, Frank and Zollh{\"o}fer, Michael and Rhodin, Helge},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2021}
}
```

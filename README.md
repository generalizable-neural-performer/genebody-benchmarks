# GNR benchmark-IBRNet: Learning Multi-View Image-Based Rendering
GNR Benchmark reivison of the implementation of paper "IBRNet: Learning Multi-View Image-Based Rendering", CVPR 2021. This repository borrows most of the code from the [original implementation](https://github.com/googleinterns/IBRNet).

> IBRNet: Learning Multi-View Image-Based Rendering  
> [Qianqian Wang](https://www.cs.cornell.edu/~qqw/), [Zhicheng Wang](https://www.linkedin.com/in/zhicheng-wang-96116897/), [Kyle Genova](https://www.kylegenova.com/), [Pratul Srinivasan](https://pratulsrinivasan.github.io/), [Howard Zhou](https://www.linkedin.com/in/howard-zhou-0a34b84/), [Jonathan T. Barron](https://jonbarron.info), [Ricardo Martin-Brualla](http://www.ricardomartinbrualla.com/), [Noah Snavely](https://www.cs.cornell.edu/~snavely/), [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/)    
> CVPR 2021
> 

## Installation
The code is tested with Python3.7, PyTorch == 1.5 and CUDA == 10.2. We recommend you to use [anaconda](https://www.anaconda.com/) to make sure that all dependencies are in place. To create an anaconda environment:
```
conda env create -f environment.yml
conda activate ibrnet
```

## Datasets

### 1. Training datasets
```
├──data/
    ├──genebody/
        ├──amanda/
        ├──barry/
```
Please first `cd data/`, and then download datasets into `data/`. The organization of the datasets should be the same as above.

#### (a) **GeneBody**
Download our data [Genebody](https://generalizable-neural-performer.github.io/genebody.html) from OneDrive for training and evaluation.

## Evaluation
First download our [pretrained models](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1) under the project `root/pretrained/` directory.

You can use `eval/eval.py` to evaluate the pretrained model. For example, to obtain the PSNR, SSIM and LPIPS on the *amanda* scene on Genebody dataset, run:
```
cd eval/
python eval.py --config ../configs/eval_genebody.txt --eval_scenes amanda
``` 
Note that the `rootdir` and `ckpt_path` should be changed accordingly.
## Reconstruction
We provide a script for reconstructing the geometry of given scene by using [marching cube](https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_marching_cubes_lewiner.py), run:

```
cd eval/
python reconstruct.py --config ../configs/eval_genebody.txt --eval_scenes amanda
```

## Training
We recommand using the RenderPeople [pretrained model](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1) to re-implement our results on Genebody dataset.
```
# this example uses 8 GPUs (nproc_per_node=8) 
python -m torch.distributed.launch --nproc_per_node=8 train.py --config configs/train_genebody.txt
```
Alternatively, you can train with a single GPU by setting `distributed=False` in `configs/train_genebody.txt` and running:
```
python train.py --config configs/train_genebody.txt
```
## Citation
```
@inproceedings{wang2021ibrnet,
  author    = {Wang, Qianqian and Wang, Zhicheng and Genova, Kyle and Srinivasan, Pratul and Zhou, Howard  and Barron, Jonathan T. and Martin-Brualla, Ricardo and Snavely, Noah and Funkhouser, Thomas},
  title     = {IBRNet: Learning Multi-View Image-Based Rendering},
  booktitle = {CVPR},
  year      = {2021}
}

```

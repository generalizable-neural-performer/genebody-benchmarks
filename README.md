# Genebody benchmark - Neural Volumes

This repository contains training and evaluation code for the paper 
[Neural Volumes](https://arxiv.org/abs/1906.07751) to use [Genebody](https://generalizable-neural-performer.github.io/) and [V-sense](https://v-sense.scss.tcd.ie/news/v-sense-volumetric-video-quality-database/) dataset. Most of the code is borrowed from the [original implementation](https://github.com/facebookresearch/neuralvolumes).


## Installation

* Python (3.6+)
  * PyTorch (1.2+)
  * NumPy
  * Pillow
  * Matplotlib
* ffmpeg (in PATH, needed to render videos)

## Datasets
Download our data [Genebody](https://generalizable-neural-performer.github.io/genebody.html) from OneDrive for training and evaluation.

Please organize your genebody folder as follows
```
├──genebody/
  ├──amanda/
  ├──barry/
```
## Evaluation
To render and evaluate a video of a trained model on GeneBody dataset, eg. `amanda`:
```
python render_genebody.py experiments/config_genebody.py --datadir path_to_genebody --subject amanda
```

## Training

To train the model on GeneBody dataset:
```
python train_genebody.py experiments/config_genebody.py  --datadir path_to_genebody --subject amanda
```

## Citation
```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }

@article{Lombardi:2019,
    author = {Stephen Lombardi and Tomas Simon and Jason Saragih and Gabriel Schwartz and Andreas Lehrmann and Yaser Sheikh},
    title = {Neural Volumes: Learning Dynamic Renderable Volumes from Images},
    journal = {ACM Trans. Graph.},
    issue_date = {July 2019},
    volume = {38},
    number = {4},
    month = jul,
    year = {2019},
    issn = {0730-0301},
    pages = {65:1--65:14},
    articleno = {65},
    numpages = {14},
    url = {http://doi.acm.org/10.1145/3306346.3323020},
    doi = {10.1145/3306346.3323020},
    acmid = {3323020},
    publisher = {ACM},
    address = {New York, NY, USA},
}
```
# Genebody benchmark-Neural Volumes

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
Please first `cd data/`, and then download datasets into `data/`. The organization of the datasets should be the same as above.
```
├──data/
    ├──genebody/
        ├──amanda/
        ├──barry/
```

#### (a) **GeneBody**
Download our data [Genebody](https://generalizable-neural-performer.github.io/genebody.html) from OneDrive for training and evaluation.


## Evaluation
To render and evaluate a video of a trained model on GeneBody dataset:
```
python render_genebody.py experiments/config_genebody.py
```
## Training

To train the model on GeneBody dataset:
```
python train_genebody.py experiments/config_genebody.py
```

## Citation

```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
```
# Genebody Benchmark - Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans

## Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.

## Datasets
Please first `cd data/`, and then download datasets into `data/`. The organization of the datasets should be the same as above.
```
├──data/
  ├──genebody/
    ├──amanda/
    ├──barry/
```

## Evaluation
To evaluate our pretrained models on Genebody, run:

```
python run.py  --type evaluate --cfg_file configs/genebody_test/amanda.yaml exp_name amanda test_novel_pose True
```
## Training
For training on a sequence of Genebody, run:
```
python train_net.py --cfg_file configs/genebody_test/amanda.yaml exp_name amanda resume False
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
```

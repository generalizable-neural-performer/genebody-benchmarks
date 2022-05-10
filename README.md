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

## Training
For training on a sequence of Genebody, run:
```
python train_net.py --cfg_file configs/genebody_test/amanda.yaml exp_name amanda resume False
```

## Evaluation
We provide our pretained models [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EvajPkSOLxtOrnzbTTiJ8KkB8qFzUwG6Y_guPMfMLElHOg?e=kJtl61). To evaluate our model or your trained model on Genebody, run:

```
python run.py  --type evaluate --cfg_file configs/genebody_test/amanda.yaml exp_name amanda test_novel_pose True
```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@article{cheng2022generalizable,
    title={Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    author={Cheng, Wei and Xu, Su and Piao, Jingtan and Qian, Chen and Wu, Wayne and Lin, Kwan-Yee and Li, Hongsheng},
    journal={arXiv preprint arXiv:2204.11798},
    year={2022}
}

@inproceedings{peng2021neural,
  title={Neural Body: Implicit Neural Representations with Structured Latent Codes for Novel View Synthesis of Dynamic Humans},
  author={Peng, Sida and Zhang, Yuanqing and Xu, Yinghao and Wang, Qianqian and Shuai, Qing and Bao, Hujun and Zhou, Xiaowei},
  booktitle={CVPR},
  year={2021}
}
```

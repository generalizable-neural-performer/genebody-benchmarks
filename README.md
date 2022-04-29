# Genebody Benchmark - Multi-view Neural Human Rendering (NHR)

This repository borrows most of the codes from the original [Pytorch implementation of NHR](https://github.com/wuminye/NHR).

**Multi-view Neural Human Rendering** </br>
[Minye Wu](https://github.com/wuminye), [Yuehao Wang](https://github.com/yuehaowang), Qiang Hu, Jingyi Yu.</br>
In CVPR 2020.</br>





## Installation
The designed architecture follows this guide [PyTorch-Project-Template](https://github.com/L1aoXingyu/PyTorch-Project-Template), you can check each folder's purpose by yourself.

Dependencies:
- python 3.6
- [pytorch>=1.2](https://pytorch.org/)
- torchvision
- opencv
- [ignite=0.2.0](https://github.com/pytorch/ignite)
- [yacs](https://github.com/rbgirshick/yacs)
- [Antialiased CNNs](https://github.com/adobe/antialiased-cnns)
- [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)
- [apex](https://github.com/NVIDIA/apex)
- [PCPR](https://github.com/wuminye/PCPR)


## Datasets
Please first `cd data/`, and then download datasets into `data/`. The organization of the datasets should be the same as above.
```
├──genebody/
  ├──amanda/
  ├──barry/
```

#### (a) **GeneBody**
Download our data [Genebody](https://generalizable-neural-performer.github.io/genebody.html) from OneDrive for training and evaluation.

## Evaluation
First download our [pretrained models](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1) under the project `tools/output/${SUBJECT}` directory.

You can use `tools/test.py` to evaluate the pretrained model. For example, to obtain the PSNR, SSIM and LPIPS on the *amanda* scene on V-Sense dataset, run:
```
cd tools/
python test.py ../configs/redandblack.yml
``` 
Note that the `DATASETS.TRAIN` in the config file should be changed accordingly.


## Training
You can use the API in `tools/train_net.py` for training on V-Sense dataset.
```
cd tools/
python train_net_vsense.py 0 ../configs/redandblack.yml
```
Note that the `DATASETS.TRAIN` in the config file should be changed accordingly.

## Citation
```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
```


## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

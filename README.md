# Genebody-Benchmarks
This repository contains the training and evaluation code for [NeuralBody](https://zju3dv.github.io/neuralbody/), [NeuralVolumes](https://stephenlombardi.github.io/projects/neuralvolumes/), [NeuralHumanRendering](https://wuminye.github.io/NHR/), [NeuralTexture](https://github.com/SSRSGJYD/NeuralTexture), [A-NeRF](https://github.com/LemonATsu/A-NeRF/) and [IBRNet](https://ibrnet.github.io/) to perform novel-view synthesis on Genebody dataset. Following benchmark tables are also shown in the [paper](https://arxiv.org/pdf/2204.11798.pdf).

The code for each method is on the branches of this repository. To re-implement the results on GeneBody, please download the pretrained models in the [Model Zoo](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EpsK3TaBfGBBgtDlcWV6nxABoQeBmGUtFmDy-XQQE1jaiQ?e=QmNQbP) first, and prepare the environment and dataset based on the `README.md` on each branch.

## News
**[29/04/22]**: First version of benchmarks released, containing 5 case-specific methods and 1 generalizable methods.
## Benchmarks
### Case-specific Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  |
| [NV](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nv)| 19.86 |0.774 |  0.267 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EniK9r9UdbtGvYvtJITBGkIBlmxSHqaoEIiIgpYBGddCHQ?e=RbS0sG)|
| [NHR](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nhr)| 20.05  |0.800 |  0.155 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/EqQDNVch2j5DmyIDnHX0VgkBDdCksmT4Kfq2oPOMn6gfMg?e=dy6yUA)|
| [NT](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nt)| 21.68  |0.881 |   0.152 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/Etg3LW44m61OjZOgDp-f4TcB_rgm_32ve529z5EZgCmoGw?e=zGUadc)|
| [NB](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nb)| 20.73  |0.878 |  0.231 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fnb%2Fgenebody)|
| [A-Nerf](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/A-Nerf)| 15.57 |0.508 |  0.242 | [ckpts](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wchengad_connect_ust_hk/En56nksujH1Fn1qWiUJ-gpIBfzdHqHf66F-RvfzwTe2TBQ?e=Zz0EgX)|

(see detail why A-Nerf's performance is counterproductive in [issue](https://github.com/LemonATsu/A-NeRF/issues/8))
### Generalizable Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  |
| PixelNeRF (Our implemetation coming soon)| 24.15   |0.903 | 0.122 | |
| [IBRNet](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/ibrnet)| 23.61    |0.836 |  0.177 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fibrnet)|
## Citation
```
@article{cheng2022generalizable,
    title={Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    author={Cheng, Wei and Xu, Su and Piao, Jingtan and Qian, Chen and Wu, Wayne and Lin, Kwan-Yee and Li, Hongsheng},
    journal={arXiv preprint arXiv:2204.11798},
    year={2022}
}
```

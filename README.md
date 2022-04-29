# Genebody-Benchmarks
This repository contains the training and evaluation code for [NeuralBody](https://zju3dv.github.io/neuralbody/), [NeuralVolumes](https://stephenlombardi.github.io/projects/neuralvolumes/), [NeuralHumanRendering](https://wuminye.github.io/NHR/), [NeuralTexture](https://github.com/SSRSGJYD/NeuralTexture) and [IBRNet](https://ibrnet.github.io/) to perform novel-view synthesis on Genebody dataset. Following benchmark tables are also shown in the [paper](https://arxiv.org/pdf/2204.11798.pdf).

The code for each method is on the branches of this repository. To re-implement the results on GeneBody, please download the pretrained models in the [model zoo]() first, and prepare the environment and dataset based on the `README.md` on each branch.

## News
**[29/04/22]**: First version of benchmarks released, containing 4 case-specific methods and 2 generalizable methods.
## Benchmarks
### Case-specific Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  |
| [NV](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nv)| 19.86 |0.774 |  0.267 | [ckpts]()|
| [NHR](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nhr)| 20.05  |0.800 |  0.155 | [ckpts]()|
| [NT](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nt)| 21.68  |0.881 |   0.152 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fnt%20%281%29)|
| [NB](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/nb)| 20.73  |0.878 |  0.231 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fnb%2Fgenebody)|
| [A-Nerf](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/A-Nerf)| 15.57 |0.508 |  0.242 | [ckpts]()|

(see detail why A-Nerf's performance is bad in [issue](https://github.com/LemonATsu/A-NeRF/issues/8))
### Generalizable Methods on Genebody
| Model  | PSNR | SSIM |LPIPS| ckpts|
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  |
| [PixelNeRF](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/main)| 24.15   |0.903 | 0.122 | [ckpts]()|
| [IBRNet](https://github.com/generalizable-neural-performer/genebody-benchmarks/tree/ibrnet)| 23.61    |0.836 |  0.177 | [ckpts](https://hkustconnect-my.sharepoint.com/personal/wchengad_connect_ust_hk/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fwchengad%5Fconnect%5Fust%5Fhk%2FDocuments%2Fgenebody%2Dbenchmark%2Dpretrained%2Fibrnet)|
## Citation
```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }
```

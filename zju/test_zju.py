import argparse
import os
import sys
from os import mkdir

import numpy as np
import torch
import torch.nn.functional as F
<<<<<<< HEAD
sys.path.insert(0, os.getcwd())
=======

>>>>>>> 3c194e0482d789c61a8ee03bdf00e3996cb11737
# sys.path.append('..')
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss

from utils.logger import setup_logger
from utils.feats_pca import feats_map_pca_projection,feats_pca_projection
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2, imageio
from imageio_ffmpeg import write_frames
from pdb import set_trace as st
from tools.metrics import ssim, lpips, psnr
from tqdm import tqdm
from data.datasets.zju_total import ZJUTotalDataset
from data.datasets.genebody import image_cropping

torch.cuda.set_device(0)

if __name__ == "__main__":

    # parse arguments from command line
    cfg.merge_from_file(sys.argv[1])
    cfg.SOLVER.IMS_PER_BATCH = 1    # set batch size to 1
    dataset_dir = sys.argv[2]
    subject = sys.argv[3]
    cfg.OUTPUT_DIR = f'./logs/{subject}'
    cfg.DATASETS.TRAIN = dataset_dir
    cfg.DATASETS.SUBJECT = subject 
    cfg.DATASETS.SKIP_STEP = [1]   # set skip step to 15
    cfg.freeze()

    import glob
    logdir = cfg.OUTPUT_DIR
    # read pretrained model
    para_files = sorted([file_ for file_ in os.listdir(logdir) if 'nr_model_' in file_])
    epochs = [int(f[9:-4]) for f in para_files]
    epoch = epochs[-1]
    para_file = os.path.join(logdir, 'nr_model_%d.pth' % epoch)
    outdir = os.path.join(logdir, 'eval_%d' % epoch)
    os.makedirs(outdir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(logdir,'tensorboard_test'))
    Dataset = ZJUTotalDataset

    test_loader, dataset = make_data_loader(cfg, dataset=Dataset, is_train=False)

    model = build_model(cfg, isTrain = False)
    print('loading pretrained model from ', para_file)
    model.load_state_dict(torch.load(para_file, map_location='cpu'))
    model = model.cuda()
    model.eval()

    feature_maps = []
    tars = []

    SSIM = []
    PSNR = []
    LPIPS = []
    from pdb import set_trace as st
    print(f'testing {len(test_loader)} images.')
    i = 0
    for batch in tqdm(test_loader):
        in_points = batch[1].cuda()
        K = batch[2].cuda()
        T = batch[3].cuda()
        near_far_max_splatting_size = batch[5]
        num_points = batch[4]
        point_indexes = batch[0]
        target = batch[7].cuda()
        inds = batch[6].cuda()
        rgbs = batch[8].cuda()
        
        res,depth,features,dir_in_world,rgb,m_point_features = model(in_points, K, T,
                            near_far_max_splatting_size, num_points, rgbs, inds)
        i = i+1
        
        # extract images and masks
        gt_mask = target[0, 3:4, :, :].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        pred_mask = ((res[0,3:4,:,:].detach().cpu().permute(1, 2, 0).numpy()*255 > 0.5)).astype(np.uint8)
        img = np.clip(res[0,0:3,:,:].detach().cpu().permute(1, 2, 0).numpy(), 0, 255)
        gt = np.clip(target[0, 0:3, : ,:].cpu().detach().permute(1, 2, 0).numpy() * 255, 0, 255)

        # crop and resize the image
        # NOTE: To be noted that we failed to train NHR using cropped image, here we use raw image instead without cropping
        #       when evaluating, we crop and resize the image to be consistent with other SOTA methods
        top, left, bottom, right = image_cropping(gt_mask)
        pred = np.clip(img * 255, 0, 255)
        pred = cv2.resize((pred*pred_mask)[top:bottom, left:right].astype(np.uint8), (512, 512), cv2.INTER_CUBIC)
        gt = cv2.resize((gt*gt_mask)[top:bottom, left:right].astype(np.uint8), (512, 512), cv2.INTER_CUBIC)
        
        if cfg.DATASETS.MOVE_CAM == 0:
            imageio.imwrite(f'{outdir}/gt_{i:04d}.png', gt)
            imageio.imwrite(f'{outdir}/pred_{i:04d}.png', pred)
            SSIM.append(ssim(pred, gt))
            PSNR.append(psnr(pred, gt))
            LPIPS.append(lpips(pred, gt))
        else:
            imageio.imwrite(f'{outdir}/render_{i:04d}.png', pred)

    with open(os.path.join(outdir, 'eval.txt'), 'w') as f:
        f.write(f'ssim {np.mean(SSIM)}\npsnr {np.mean(PSNR)}\nlpips {np.mean(LPIPS)}\n')
        print(f'ssim {np.mean(SSIM)} psnr {np.mean(PSNR)} lpips {np.mean(LPIPS)}')

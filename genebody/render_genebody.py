import argparse
import os
import sys, re
from os import mkdir

import numpy as np
import torch
import torch.nn.functional as F

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
from metrics import ssim, lpips, psnr
from tqdm import tqdm
from data.datasets.genebody_total import GeneBodyTotalDataset
from data.datasets.genebody import image_cropping

torch.cuda.set_device(0)


def extract_float(text):
    flts = []
    for c in re.findall('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]+)',text):
        if c != '':
            try:
                flts.append(float(c))
            except ValueError as e:
                continue
    return flts
def natural_sort(files):
    return	sorted(files, key = lambda text: \
        extract_float(os.path.basename(text)) \
        if len(extract_float(os.path.basename(text))) > 0 else \
        [float(ord(c)) for c in os.path.basename(text)])


# para_file = 'nr_model_%d.pth' % epoch

if __name__ == "__main__":

    cfg.merge_from_file(sys.argv[1])
    cfg.SOLVER.IMS_PER_BATCH = 1
    dataset_dir = sys.argv[2]
    subject = sys.argv[3]
    cfg.OUTPUT_DIR = f'./logs/{subject}'
    cfg.DATASETS.TRAIN = dataset_dir
    cfg.DATASETS.SUBJECT = subject
    cfg.DATASETS.SKIP_STEP = [5]
    cfg.freeze()

    import glob
    logdir = cfg.OUTPUT_DIR
    # para_file = sorted(glob.glob(logdir + '/nr_model_*.pth'))[-1]
    para_files = natural_sort([file_ for file_ in os.listdir(logdir) if 'nr_model_' in file_])
    epochs = [int(f[9:-4]) for f in para_files]
    epoch = epochs[-1]
    para_file = os.path.join(logdir, 'nr_model_%d.pth' % epoch)
    outdir = os.path.join(logdir, 'white_%d' % epoch)
    os.makedirs(outdir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(logdir,'tensorboard_test'))
    Dataset = GeneBodyTotalDataset

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
        
        cur_mask = target[0, 3:4, :, :].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        pred = res[0,0:3,:,:].masked_fill_(res[0,3:4,:,:] <= 0.5, 1).clamp(0, 1)
        pred = pred.detach().cpu().permute(1, 2, 0).numpy()
        gt = target[0,0:3,:,:].masked_fill_(target[0,3:4,:,:] <= 0.5, 1).clamp(0, 1)
        gt = gt.detach().cpu().permute(1, 2, 0).numpy() 
        top, left, bottom, right = image_cropping(cur_mask)

        pred = cv2.resize((pred*255).astype(np.uint8)[top:bottom, left:right], (512, 512), cv2.INTER_CUBIC)
        gt = cv2.resize((gt*255).astype(np.uint8)[top:bottom, left:right], (512, 512), cv2.INTER_CUBIC)
        

        if cfg.DATASETS.MOVE_CAM == 0:
            imageio.imwrite(f'{outdir}/gt_{i:04d}.png', gt)
            imageio.imwrite(f'{outdir}/pred_{i:04d}.png', pred)
            # cv2.imwrite(f'{logdir}/{cfg.DATASETS.SUBJECT}_nhr_e{epoch}/raw_frame{i:04d}_{i}.jpg', pred)
            SSIM.append(ssim(pred, gt))
            PSNR.append(psnr(pred, gt))
            LPIPS.append(lpips(pred, gt))
        else:
            imageio.imwrite(f'{outdir}/render_{i:04d}.png', pred)

    with open(os.path.join(outdir, 'eval.txt'), 'w') as f:
        f.write(f'ssim {np.mean(SSIM)}\npsnr {np.mean(PSNR)}\nlpips {np.mean(LPIPS)}\n')
        print(f'ssim {np.mean(SSIM)} psnr {np.mean(PSNR)} lpips {np.mean(LPIPS)}')


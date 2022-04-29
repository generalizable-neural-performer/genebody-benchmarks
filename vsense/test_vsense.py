import argparse
import os
import sys
from os import mkdir

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('..')
from config import cfg
from data import make_data_loader

from torch.utils.tensorboard import SummaryWriter
import cv2
from pdb import set_trace as st
from tools.metrics import ssim, lpips, psnr

torch.cuda.set_device(0)


cfg_file, dataset_dir, subject = sys.argv[1], sys.argv[2], sys.argv[3]

cfg.merge_from_file(cfg_file)
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.freeze()

epoch = 30
model_path =cfg.OUTPUT_DIR

para_file = model_path + '/nr_model_%d.pth' % epoch
print(f'using {para_file}')
writer = SummaryWriter(log_dir=os.path.join(model_path,'tensorboard_test'))
test_loader, dataset = make_data_loader(cfg, is_train=False)

model = torch.load(para_file)
model.eval()
model = model.cuda()

feature_maps = []
tars = []

SSIM = []
PSNR = []
LPIPS = []
print(f'testing {len(test_loader)} images.')
i = 0
for batch in test_loader:
    print(i)
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
    
    cur_mask = res[:, 3, : ,:].repeat(1, 3, 1, 1)
    res[:,0:3,:,:] = res[:,0:3,:,:]*cur_mask
    img = res[0,0:3,:,:].detach().cpu().permute(1, 2, 0).numpy()[..., ::-1]
    os.makedirs(f'{cfg.DATASETS.SUBJECT}_nhr_e{epoch}_demo', exist_ok=True)
    os.makedirs(f'{cfg.DATASETS.SUBJECT}_nhr_e{epoch}', exist_ok=True)
    gt = np.clip(target[0, 0:3, : ,:].cpu().detach().permute(1, 2, 0).numpy()[..., ::-1] * 255, 0, 255).astype(np.uint8)
    pred = np.clip(img * 255, 0, 255).astype(np.uint8)
    if cfg.DATASETS.MOVE_CAM == 0:
        cv2.imwrite(f'{cfg.DATASETS.SUBJECT}_nhr_e{epoch}/gt_frame{i:04d}_{i}.jpg', gt)
        cv2.imwrite(f'{cfg.DATASETS.SUBJECT}_nhr_e{epoch}/pred_frame{i:04d}_{i}.jpg', pred)
        SSIM.append(ssim(pred, gt))
        PSNR.append(psnr(pred, gt))
        LPIPS.append(lpips(pred, gt))
    else:
        cv2.imwrite(f'{cfg.DATASETS.SUBJECT}_nhr_e{epoch}_demo/pred_frame{i:04d}_demo.jpg', pred)

print(f'ssim {np.mean(SSIM)} psnr {np.mean(PSNR)} lpips {np.mean(LPIPS)}')
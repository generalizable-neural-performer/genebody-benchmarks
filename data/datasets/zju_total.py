import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
from .zju import ZJUDataset

class ZJUTotalDataset(torch.utils.data.Dataset):

    def __init__(self, cfg, transforms, is_train = True):
        super(ZJUTotalDataset, self).__init__()

        self.dataset_paths = cfg.DATASETS.TRAIN
        self.frame_nums = cfg.DATASETS.FRAME_NUM
        self.skips = cfg.DATASETS.SKIP_STEP
        self.near_far = cfg.INPUT.NEAR_FAR_SIZE
        self.crop = cfg.DATASETS.CROP
        self.pc_upsample = cfg.DATASETS.PC_UPSAMPLE
        self.datasets = []
        self.nums = []

        Holes = ['None']*len(self.dataset_paths)

        if len(self.dataset_paths) == len(cfg.DATASETS.HOLES):
            Holes = cfg.DATASETS.HOLES

        for d in zip(self.dataset_paths, self.frame_nums, self.skips, self.near_far, Holes,cfg.DATASETS.IGNORE_FRAMES):

            if is_train:
                dataset = ZJUDataset(d[0],
                                d[1],
                                cfg.INPUT.SIZE_TRAIN,
                                cfg.DATASETS.MASK,
                            transforms, d[3], d[2],cfg.DATASETS.RANDOM_NOISY, d[4],d[5], is_train=is_train, subject=cfg.DATASETS.SUBJECT,
                            cropping=self.crop, pc_upsampling=self.pc_upsample)
            else:
                dataset = ZJUDataset(d[0],
                                d[1],
                                cfg.INPUT.SIZE_TEST,
                                cfg.DATASETS.MASK,
                            transforms, d[3], d[2],False, 0, is_train=is_train, subject=cfg.DATASETS.SUBJECT, move_cam=cfg.DATASETS.MOVE_CAM,
                            cropping=self.crop, pc_upsampling=self.pc_upsample)

            self.nums.append(len(dataset))
            self.datasets.append(dataset)


    def __len__(self):
        return np.sum(self.nums)

    def __getitem__(self, index, need_transform = True):
        cnt = 0
        while index >= self.nums[cnt]:
            index -= self.nums[cnt]
            cnt = cnt + 1
        
        res = list(self.datasets[cnt].__getitem__(index,need_transform))
        res[2] = cnt
        return tuple(res)


    def load_vs(self, cfg):
        for i,d in enumerate(self.datasets):
            d.vs  = torch.load(os.path.join(cfg.OUTPUT_DIR,'vs_%d.pth'%i))
    
    def save_vs(self, cfg):
        for i,d in enumerate(self.datasets):
            torch.save(d.vs,os.path.join(cfg.OUTPUT_DIR,'vs_%d.pth'%i))







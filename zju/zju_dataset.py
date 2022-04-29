import numpy as np
import os, sys
from PIL import Image
from torch.utils.data import Dataset
import torch
sys.path.insert(0, os.getcwd())

from util import img_transform, map_transform, natural_sort
import imageio, cv2

def rot2euler(R):
    phi = np.arctan2(R[1,2], R[2,2])
    theta = -np.arcsin(R[0,2])
    psi = np.arctan2(R[0,1], R[0,0])
    return np.array([phi, theta, psi])

def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0],
                [0, cos(phi), sin(phi)],
                [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
                [0, 1, 0],
                [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0],
                [-sin(psi), cos(psi), 0],
                [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R

def image_cropping(mask):
    a = np.where(mask != 0)
    h, w = list(mask.shape[:2])

    top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
    bbox_h, bbox_w = bottom - top, right - left

    # padd bbox
    bottom = min(int(bbox_h*0.1+bottom), h)
    top = max(int(top-bbox_h*0.1), 0)
    right = min(int(bbox_w*0.1+right), w)
    left = max(int(left-bbox_h*0.1), 0)
    bbox_h, bbox_w = bottom - top, right - left

    if bbox_h >= bbox_w:
        w_c = (left+right) / 2
        size = bbox_h
        if w_c - size / 2 < 0:
            left = 0
            right = size
        elif w_c + size / 2 >= w:
            left = w - size
            right = w
        else:
            left = int(w_c - size / 2)
            right = left + size
    else:   # bbox_w >= bbox_h
        h_c = (top+bottom) / 2
        size = bbox_w
        if h_c - size / 2 < 0:
            top = 0
            bottom = size
        elif h_c + size / 2 >= h:
            top = h - size
            bottom = h
        else:
            top = int(h_c - size / 2)
            bottom = top + size
    
    return top, left, bottom, right


class ZJUDataset(Dataset):

    def __init__(self, dir, uv, H, W, subject, is_train, eval_skip=1, view_direction=False):
        # self.idx_list = idx_list
        # self.dir = dir
        # self.crop_size = (H, W)
        # self.view_direction = view_direction
        self.dir = dir
        self.uv_path = uv
        self.subject = subject
        self.is_train = is_train
        self.eval_skip = eval_skip
        self.H, self.W = H, W
        self.file_path = os.path.join(self.dir, self.subject)

        self.all_frames = sorted(os.listdir(os.path.join(self.file_path, 'Camera_B1')))

        all_views = natural_sort([f for f in os.listdir(self.file_path) if 'Camera' in f])
        # self.train_views = [view for view in all_views_raw[1::3] if view in all_views]
        self.train_views = all_views
        if self.is_train:
            self.views = self.train_views
            self.frames = self.all_frames[:300]
        else:
            # self.views = sorted(list(set(all_views)-set(self.train_views)))
            self.views = all_views
            self.frames = self.all_frames[300:]
            self.frames = self.frames[::self.eval_skip]

        self.mask_path = [f for f in os.listdir(os.path.join(self.dir, self.subject)) if 'mask' in f][-1]
        
        param_path = os.path.join(self.dir, self.subject, 'annots.npy')
        annots = np.load(param_path, allow_pickle=True).reshape(-1)[0]
        self.Ts = []
        for i in self.views:
            idx = all_views.index(i)
            # Rt = np.array(annots['cams']['RT'][idx], dtype=np.float32)
            R = np.array(annots['cams']['R'][idx])
            t = np.array(annots['cams']['T'][idx])
            Rt = np.concatenate([ \
                    np.concatenate([R, t.reshape(3,1)/1000.],1), \
                    [[0,0,0,1]]], 0).astype(np.float32)
            c2w = np.linalg.inv(Rt)
            self.Ts.append(c2w)
        self.Ts = np.array(self.Ts)

    def __len__(self):
        return len(self.frames) * len(self.views)

    def __getitem__(self, idx):

        frame_idx = idx // len(self.views)
        view_idx = idx % len(self.views)
        frame = self.frames[frame_idx][:-4]

        img_dir = os.path.join(self.dir, self.subject, self.views[view_idx])
        img_paths = [dir_ for dir_ in sorted(os.listdir(img_dir)) if frame in dir_]
        img = imageio.imread(os.path.join(img_dir, img_paths[0]))

        msk_dir = os.path.join(self.dir, self.subject, self.mask_path, self.views[view_idx])
        msk_paths = [dir_ for dir_ in sorted(os.listdir(msk_dir)) if frame in dir_]
        msk = imageio.imread(os.path.join(msk_dir, msk_paths[0]))
        img = img * (msk[...,None] > 0).astype(np.uint8)

        uv_dir = os.path.join(self.uv_path, self.subject, 'smpl_uv', self.views[view_idx])
        uv_paths = [dir_ for dir_ in sorted(os.listdir(uv_dir)) if frame in dir_]
        uv_map = imageio.imread(os.path.join(uv_dir, uv_paths[0]))[...,:2]

        top, left, bottom, right = image_cropping(msk)
        img = cv2.resize(img[top:bottom, left:right].copy(), (self.W,self.H), cv2.INTER_CUBIC)
        img = img_transform(img).float()
        uv_map = cv2.resize(uv_map[top:bottom, left:right].copy(), (self.W,self.H), cv2.INTER_NEAREST)
        msk = cv2.resize(msk[top:bottom, left:right].copy(), (self.W,self.H), cv2.INTER_NEAREST)
        uv_map = map_transform(uv_map).float()
        uv_map = uv_map * (uv_map >= 0).float()


        mask = torch.from_numpy((msk>0).astype(np.float32)).unsqueeze(0)

        rotmat = self.Ts[view_idx][:3, :3]
        euler = rot2euler(rotmat)
        extrinsics = torch.from_numpy(euler).float()

        return img, uv_map, extrinsics, mask
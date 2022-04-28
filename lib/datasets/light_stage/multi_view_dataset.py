import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData

from pdb import set_trace as st


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



def save_ply2(fname, pts, alpha=None):
    fmt = '%.6f %.6f %.6f' if alpha is None else '%.6f %.6f %.6f %d %d %d'
    header = f'ply\nformat ascii 1.0\nelement vertex {pts.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nend_header' if alpha is None else \
                f'ply\nformat ascii 1.0\nelement vertex {pts.shape[0]}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header'
    if alpha is not None:
        r = (alpha.detach().cpu().numpy() * 255).astype(np.uint8)
        g = np.zeros_like(r)
        b = np.zeros_like(r)
        rgb = np.stack([r,g,b], -1)
        pts = np.concatenate([pts, rgb.view(-1,3)], -1)
        # pts = pts[:, ::8].reshape(-1,6)
    np.savetxt(fname, pts, fmt=fmt, comments='', header=(header))

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, eval_skip=1):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])
        test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        view = cfg.training_view if split == 'train' else test_view
        if len(view) == 0:
            view = [0]

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame
            if self.human == 'CoreView_390':
                i = 0

        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        # from pdb import set_trace as st
        if cfg.test_novel_pose:
            self.ims = self.ims[::eval_skip]
            self.cam_inds = self.cam_inds[::eval_skip]
        self.num_cams = len(view)

        self.nrays = cfg.N_rand

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        msk_cihp = imageio.imread(msk_path)
        msk = (msk_cihp != 0).astype(np.uint8)

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100

        return msk

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        try:
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                                        '{}.npy'.format(i))
            xyz = np.load(vertices_path).astype(np.float32)
        except:
            vertices_path = os.path.join(self.data_root, 'smpl',
                                        '{}.ply'.format(i))
            from lib.utils import data_utils
            xyz = data_utils.load_ply(vertices_path).astype(np.float32)

        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)
        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh']
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['Th'].astype(np.float32)
        # save_ply2('xyz.ply', xyz.reshape(-1, 3))

        xyz = np.dot(xyz - Th, R)

        # save_ply2('xyz_smpl.ply', xyz.reshape(-1, 3))

        # obtain the bounds for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        if cfg.big_box:
            min_xyz -= 0.05
            max_xyz += 0.05
        else:
            min_xyz[2] -= 0.05
            max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        # construct the coordinate
        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]
        voxel_size = np.array(cfg.voxel_size)
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

        # construct the output shape
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1
        return coord, out_sh, can_bounds, bounds, Rh, Th

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk = self.get_mask(index)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        top, left, bottom, right = image_cropping(msk)
        
        msk = msk[top:bottom, left:right]
        img = img[top:bottom, left:right, :]
        msk = cv2.resize(msk, (cfg.H,cfg.W), \
                            interpolation = cv2.INTER_NEAREST)
        img = cv2.resize(img, (cfg.H,cfg.W), \
                            interpolation = cv2.INTER_NEAREST)
        K[0,2] -= left
        K[1,2] -= top
        K[0,:] *= cfg.W / float(right - left)
        K[1,:] *= cfg.H / float(bottom - top)

        # reduce the image resolution by ratio
        # H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        # img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        elif self.human in ['CoreView_396']:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i
            i = i - 810
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            i)
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)


        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'rgb': rgb,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index - cfg.begin_ith_frame
        if cfg.test_novel_pose:
            latent_index = cfg.num_train_frame - 1
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)

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
import open3d as o3d

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



class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, eval_skip=1):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        num_cams = 48
        all_views = list(range(num_cams))
        # test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        if self.human == 'Tichinah_jervier' or self.human == 'dannier':
            all_views = list(set(all_views) - set([32]))
        elif self.human == 'wuwenyan':
            all_views = list(set(all_views)-set([34, 36]))
        elif self.human == 'joseph_matanda':
            all_views = list(set(all_views) - set([39, 40, 42, 43, 44, 45, 46, 47]))
        assert len(self.cams['K']) == len(all_views)
        all_views = sorted(all_views)
        test_view = sorted(list(set(all_views) - set(cfg.training_view)))

        view = cfg.training_view if split == 'train' else test_view
        if len(view) == 0:
            view = [0]
        self.all_views = all_views
        self.views = view

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame

        ims = sorted(os.listdir(os.path.join(self.data_root, self.human, 'smpl')))
        ims = ims[i:i + ni * i_intv][::i_intv]
        ims = ims[::eval_skip]
    
        self.ims = ims
        self.nrays = cfg.N_rand
        self.ni = ni
        self.num_cams = 1

        self.Ks, self.Rs, self.Ts, self.Ds = [], [], [], []
        for view in self.views:
            cam_ind = self.all_views.index(view)
            K = np.array(self.cams['K'][cam_ind], dtype=np.float32)
            Rt = np.array(self.cams['RT'][cam_ind], dtype=np.float32)
            Rt = np.linalg.inv(Rt)
            self.Rs.append(Rt[:3, :3])
            self.Ts.append(Rt[:3, 3:])
            self.Ks.append(K)


    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, self.human, 'smpl',
                                     self.ims[i][:-4]+'.ply')
        mesh = o3d.io.read_triangle_mesh(vertices_path)
        xyz = np.asarray(mesh.vertices).astype(np.float32)
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
        params_path = os.path.join(self.data_root, self.human, 'param',
                                   self.ims[i][:-4]+'.npy')
        params = np.load(params_path, allow_pickle=True).item()
        # Rh = params['Rh']
        Rh = params['pose'][:3]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # Th = params['Th'].astype(np.float32)
        Th = params['transl'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)
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

    def get_mask(self, index, view):
        # basename = self.ims[index].split('/')[-1][:-4]
        basename = self.ims[index][:-4]
        path = os.path.join(self.data_root, self.human, 'mask', '%02d' % view)
        mask_paths = sorted([os.path.join(path, p) for p in os.listdir(path) if basename in p])
        msk_cihp = imageio.imread(mask_paths[0])

        msk = (msk_cihp > 128).astype(np.uint8) # msk: 0 1

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100 # msk: 0 1 100
        return msk

    def prepare_inside_pts(self, pts, i):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        # for nv in range(self.ims.shape[1]):
        for i, nv in enumerate(self.views):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            RT = np.concatenate([self.Rs[i], self.Ts[i]], axis=1)
            pts2d = base_utils.project(pts3d_, self.Ks[i], RT)

            msk = self.get_mask(i, nv)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside

    def __getitem__(self, index):
        i = index
        latent_index = index
        frame_index = index + cfg.begin_ith_frame

        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            i)

        voxel_size = cfg.voxel_size
        x = np.arange(can_bounds[0, 0], can_bounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(can_bounds[0, 1], can_bounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(can_bounds[0, 2], can_bounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        inside = self.prepare_inside_pts(pts, i)

        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'pts': pts,
            'inside': inside
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'wbounds': can_bounds,
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.ni

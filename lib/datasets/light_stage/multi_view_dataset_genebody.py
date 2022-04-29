import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
import sys
import open3d as o3d

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir,'../../../'))



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

        # num_cams = len(self.cams['K'])
        num_cams = 48
        all_views = list(range(num_cams))
        # test_view = [i for i in range(num_cams) if i not in cfg.training_view]
        if self.human == 'Tichinah_jervier' or self.human == 'dannier':
            all_views = list(set(all_views) - set([32]))
        elif self.human == 'wuwenyan':
            all_views = list(set(all_views)-set([34, 36]))
        elif self.human == 'joseph_matanda':
            all_views = list(set(all_views) - set([39, 40, 42, 43, 44, 45, 46, 47]))
        all_views = sorted(all_views)
        # test_view = sorted(list(set(all_views) - set(cfg.training_view)))
        test_view = all_views
        # view = cfg.training_view if split == 'train' else test_view
        view = all_views
        if len(view) == 0:
            view = [0]
        self.all_views = all_views

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame

        if cfg.test_novel_pose:
            i = (i + cfg.num_train_frame) * i_intv
            ni = cfg.num_novel_pose_frame

        all_ims = []
        cam_ids = []
        for v in view:
            ims = sorted(os.listdir(os.path.join(self.data_root, self.human, 'image', f'{v:02d}')))
            ims = ims[i:i + ni * i_intv][::i_intv]
            ims = ims[::eval_skip]
            all_ims += [os.path.join(os.path.join(self.data_root, self.human, 'image', f'{v:02d}'), f) for f in ims]
            cam_ids += [v for f in ims]
        self.ims = all_ims
        self.cam_inds = cam_ids
        self.num_cams = len(view)

        self.nrays = cfg.N_rand

    def get_mask(self, index):
        basename = self.ims[index].split('/')[-1][:-4]

        msk_path = os.path.dirname(self.ims[index]).replace('image', 'mask') + f'/mask{int(basename):04d}' + '.png'
        msk_cihp = imageio.imread(msk_path)

        msk = (msk_cihp > 128).astype(np.uint8) # msk: 0 1

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 100 # msk: 0 1 100
        return msk

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, self.human, 'smpl',
                                     f'{i:04d}.ply')
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
                                   f'{i:04}.npy')
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

    def __getitem__(self, index):
        # img_path = os.path.join(self.data_root, self.ims[index])
        img_path = os.path.join(self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk = self.get_mask(index)
        size = msk.shape[:2]
        h, w = list(size)
        a = np.where(msk != 0)
        bbox = [[0, np.min(a[1])], [msk.shape[0], np.max(a[1])]]
        top, left, bottom, right = image_cropping(msk)
        
        msk = msk[top:bottom, left:right]
        img = img[top:bottom, left:right, :]
        msk = cv2.resize(msk, (cfg.H,cfg.W), \
                            interpolation = cv2.INTER_NEAREST)
        img = cv2.resize(img, (cfg.H,cfg.W), \
                            interpolation = cv2.INTER_NEAREST)


        cam_ind = self.cam_inds[index]
        cam_ind = self.all_views.index(cam_ind)
        K = np.array(self.cams['K'][cam_ind], dtype=np.float32)

        Rt = np.array(self.cams['RT'][cam_ind], dtype=np.float32)
        Rt = np.linalg.inv(Rt)
        R = Rt[:3, :3]
        T = Rt[:3, 3]
        T = T[..., None]
        K[0,2] -= left
        K[1,2] -= top
        K[0,:] *= cfg.W / float(right - left)
        K[1,:] *= cfg.H / float(bottom - top)

        # msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        # msk = msk[..., None]
        i = int(os.path.basename(img_path)[:-4])
        frame_index = i
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            i)
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, can_bounds, self.nrays, self.split, index=index)

        # except BaseException as e:
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
        latent_index = frame_index - cfg.frame_name_start
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



if __name__=='__main__':
    dataset = Dataset('data/vsense', 'Matis_obj', 'data/vsense/Matis_obj/annots.npy', 'train')
    # print(dataset[0])
    for i, data in enumerate(dataset):
        print(data)

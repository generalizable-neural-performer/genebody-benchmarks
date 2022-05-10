from lib.utils.render_utils import load_cam
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
from lib.utils import render_utils
from lib.utils import data_utils
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


def load_cam(ann_file):
    if ann_file.endswith('.json'):
        annots = json.load(open(ann_file, 'r'))
        cams = annots['cams']['20190823']
    else:
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']

    K = []
    RT = []
    lower_row = np.array([[0., 0., 0., 1.]])
    for i in range(len(cams.keys())):
        K.append(np.array(cams[i]['K']))
        K[i][:2] = K[i][:2] * cfg.ratio
        r = np.array(cams[i]['R'])
        t = np.array(cams[i]['T'])
        t = t[..., None]
        t /= 2.87 # # NOTE: to use our pretrained model, need to rescale it to the real-world scale
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return K, RT


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, eval_skip=1):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        render_annots = np.load(f'/mnt/lustre/share_data/chengwei/render_smpl/{self.human}/circle/{self.human}.npy', allow_pickle=True).item()['cams']
        # this K is for spiral traj
        self.K = render_annots['K']
        self.render_w2c = render_annots['RT']
        ann_file = os.path.join('/mnt/lustre/share_data/chengwei/camera_npy', self.human + '.npy')
        # this K is all the Ks of 48 cams
        K, RT = load_cam(ann_file)
        view = [1, 13,25, 37]
        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)
        self.RT_w2c = []
        for rt in self.RT:
            # our annotation is c2w, nb use w2c
            rt = np.linalg.inv(rt)
            self.RT_w2c.append(rt)
        self.RT = np.array(self.RT_w2c)
        self.all_views = view

        # prepare input images
        i = 0
        i_intv = 1
        ni = 150
        self.ni = ni
        all_ims = []
        for idx in range(150):
            _ims = []
            for v in view:
            # only four src views
                ims = sorted(os.listdir(os.path.join(self.data_root, self.human, 'image', f'{v:02d}')))
                im = ims[idx]
                _ims.append(os.path.join(os.path.join(self.data_root, self.human, 'image', f'{v:02d}'), im))
            all_ims.append(_ims)
        self.ims = all_ims

        self.nrays = cfg.N_rand

    def get_mask(self, index):
        ims = self.ims[index]
        msks = []
        for nv in range(len(ims)):
            im = ims[nv]
            basename = im.split('/')[-1][:-4]
            msk_path = os.path.dirname(im).replace('image', 'mask') + f'/mask{int(basename):04d}' + '.png'
            msk_cihp = imageio.imread(msk_path)
            msk = (msk_cihp > 128).astype(np.uint8) # msk: 0 1

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        msks = np.array(msks, dtype=np.uint8)
        return msks

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        vertices_path = os.path.join(self.data_root, self.human, 'smpl',
                                     f'{i:04d}.obj')
        mesh = o3d.io.read_triangle_mesh(vertices_path)
        xyz = np.asarray(mesh.vertices).astype(np.float32)
        xyz /= 2.87 # NOTE: to use our pretrained model, need to rescale it to the real-world scale

        nxyz = np.zeros_like(xyz).astype(np.float32)
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
        Rh = params['smplx']['global_orient'].numpy()
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        Th = params['smplx']['transl'].numpy()
        Th = Th.astype(np.float32) / 2.87 # NOTE: to use our pretrained model, need to rescale it to the real-world scale
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
        frame_index = index + cfg.begin_ith_frame
        latent_index = index
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            frame_index)
        msks = self.get_mask(index)
        resized_msks = []
        for msk in msks:
            # top, left, bottom, right = image_cropping(msk)
            # msk = msk[top:bottom, left:right]
            msk = cv2.resize(msk, (cfg.H,cfg.W), \
                                interpolation = cv2.INTER_NEAREST)
            # use dummy mask for debug
            msk = np.ones_like(msk)
            resized_msks.append(msk)
        msks = np.array(resized_msks)
        K = self.K
        cam_ind = index % len(self.render_w2c)

        ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
            np.linalg.inv(self.render_w2c[cam_ind]), K, can_bounds)
        # except BaseException as e:
        ret = {
            'coord': coord,
            'out_sh': out_sh,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = frame_index - cfg.frame_name_start
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': cam_ind,
            'msks': msks,
            'Ks': self.Ks,
            'RT': self.RT
        }
        ret.update(meta)
        return ret

    def __len__(self):
        # return len(self.ims)
        return self.ni
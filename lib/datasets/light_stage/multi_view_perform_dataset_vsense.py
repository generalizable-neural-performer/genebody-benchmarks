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
from lib.utils import data_utils
from lib.utils import render_utils
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir,'../../../'))


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

    for i in range(len(cams['K'])):
        K.append(np.array(cams['K'][i]))
        r = np.array(cams['R'][i])
        t = np.array(cams['T'][i])
        t = t[..., None]
        r_t = np.concatenate([r, t], 1)
        RT.append(np.concatenate([r_t, lower_row], 0))

    return K, RT

def get_cams_vsense(subject):
    all_extrinsics = []
    param_dir = f'/mnt/lustre/share_data/chengwei/vsense/{subject}/RENDER_PARAM'
    for cam in sorted(os.listdir(param_dir)):
        cam = os.path.join(param_dir, cam)
        param = np.load(cam, allow_pickle=True).item()
        scale = param.get('scale')
        ortho_ratio = param.get('ortho_ratio')
        intr_mat = param.get('intr')
        norm_mat = param.get('norm')
        view_mat = param.get('view')
        rot_mat = param.get('rot')
        focal = param.get('focal')
        transl_z = param.get('transl_z')
        center = param.get('center')
        height = param.get('height')
        gl2cv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        extrinsic = gl2cv @ view_mat @ rot_mat @ norm_mat
        K = intr_mat
        all_extrinsics.append(extrinsic)

    return all_extrinsics, K

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split
        K, RT = load_cam(ann_file)

        # this is for spiral render
        self.render_w2c, self.K = get_cams_vsense(human)

        view = cfg.training_view
        self.views = view
        self.all_views = view

        # prepare input images
        i = 0
        i_intv = 1
        ni = 299
        self.ni = ni
        all_ims = []
        ims = sorted(os.listdir(os.path.join(self.data_root, self.human, 'SMPL')))
        ims = ims[i:i + ni * i_intv][::i_intv]
        for im in ims:
            view_ims = []
            for vid in view:
                view_ims += [os.path.join(self.data_root, self.human, 'RENDER', f'cam_{vid:02d}', im[:-4] + '.png')]
            all_ims.append(view_ims)
        self.ims = all_ims
        self.num_cams = 1

        self.nrays = cfg.N_rand

        # this is for src views
        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)



    def get_mask(self, i):
        ims = self.ims[i]
        msks = []
        for nv in range(len(ims)):
            im = ims[nv]
            msk_path = im.replace('RENDER', 'MASK')
            msk_cihp = imageio.imread(msk_path)
            msk = (msk_cihp != 0).astype(np.uint8) # msk: 0 1

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 0 # msk: 0 1 100
            msks.append(msk)

        return msks

    def prepare_input(self, i):
        # read xyz, normal, color from the ply file
        import trimesh
        vertices_path = os.path.join(self.data_root, self.human, 'SMPL',
                                     f'{i:06d}.obj')
        # mesh = o3d.io.read_triangle_mesh(vertices_path)
        mesh = trimesh.load_mesh(vertices_path)
        xyz = np.asarray(mesh.vertices).astype(np.float32)
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
        params_path = os.path.join(self.data_root, self.human, 'SMPL_PARAM',
                                   f'{i:06d}.npy')
        params = np.load(params_path, allow_pickle=True).item()
        # Rh = params['Rh']
        Rh = params['pose'][:3]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        # Th = params['Th'].astype(np.float32)
        Th = params['transl'].astype(np.float32)
        xyz = np.dot(xyz - Th, R)
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
        # st()
        x = 32
        out_sh = (out_sh | (x - 1)) + 1
        return coord, out_sh, can_bounds, bounds, Rh, Th


    def prepare_mesh_param(self, i):
        img_name = self.ims[i][0]
        cam_param_path = img_name.replace('RENDER', 'PARAM') + '.npy'
        param = np.load(cam_param_path, allow_pickle=True).item()
        scale = param.get('scale')
        ortho_ratio = param.get('ortho_ratio')
        center = param.get('center')
        spatial_freq = scale / ortho_ratio
        mesh_param = {'center': center, 'spatial_freq': spatial_freq}
        return mesh_param


    def __getitem__(self, index):
        latent_index = index
        frame_index = index + cfg.begin_ith_frame

        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            frame_index)
        resized_msks = []
        msks = self.get_mask(index)
        for msk in msks:
            msk = cv2.resize(msk, (cfg.H,cfg.W), \
                                interpolation = cv2.INTER_NEAREST)
            # use dummy mask for debug
            msk = np.ones_like(msk)
            resized_msks.append(msk)

        msks = np.array(resized_msks)
        K = self.K
        cam_ind = index % len(self.render_w2c)
        ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
            self.render_w2c[cam_ind], K, can_bounds)
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
        latent_index = frame_index - cfg.begin_ith_frame
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'wbounds': can_bounds,
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
        return len(self.ims)



if __name__=='__main__':
    dataset = Dataset('data/vsense', 'Matis_obj', 'data/vsense/Matis_obj/annots.npy', 'train')
    # print(dataset[0])
    for i, data in enumerate(dataset):
        print(data)
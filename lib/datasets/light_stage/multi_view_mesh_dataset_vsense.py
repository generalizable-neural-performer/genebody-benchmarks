from numpy.core.fromnumeric import argmax
from numpy.core.overrides import verify_matching_signatures
from numpy.ma.core import _extrema_operation
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
from plyfile import PlyData
import sys
import open3d as o3d
from pdb import set_trace as st

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir,'../../../'))

from pdb import set_trace as st


def cam_pose_vis(output_path, camera_poses, cds='gl', pose_type='w2c', rgbs=None, 
                  use_tex=True, camera_ids=None):
    """ Visualize camera poses
        output_path: Output path in obj format
        camera_poses: List or array of camera poses with size [Nx4x4]
        cds: Camera cordinate system, whether the camera cordinate system is 
             'gl' (xyz->right,up,backward) or 'cv' (xyz->right,down,forward)
        pose_type: Camera pose type, whether the camera pose is 
                   'w2c' world to camera or 'c2w' camera to world
        rgbs: Desired rgb value for each camera with size [Nx3], None if no desired value
        use_tex: If True, outputs file with camera id texture
        camera_ids: Camera ids, None with default ranking id, or [N] array with specific id
    """
    # path
    if output_path[-4:] is not '.obj':
        if '.' not in output_path:
            output_path += '.obj'
        else:
            output_path = os.path.splitext(output_path)[0] + '.obj'
    # convert to c2w
    if pose_type == 'w2c':
        c2ws = np.linalg.inv(np.array(camera_poses))
    else:
        c2ws = np.array(camera_poses)
    # scaling the camera pose
    tex_cir_rad = 40
    transl = c2ws[:, :3, 3]
    min_, max_ = np.min(transl, axis=0), np.max(transl, axis=0)
    scale = np.mean(max_-min_) * 0.1
    camera_num = len(camera_poses)
    # defining camera vertices, faces and tex
    cam_verts = np.array([
        [0, 0, 0], [.5, .5, -1], [-.5, .5, -1],
        [-.5, -.5, -1], [.5, -.5, -1], [.5, .6, -1],
        [-.5, .6, -1], [0, .8, -1]
    ])*scale                            # camera vertex coordinate
    # convert cv camera coordinate to gl (default camera system in meshlab is gl)
    if cds == 'cv':
        cam_verts = cam_verts * np.array([1,-1,-1])
    face_map = np.array([
        [1,2,3], [1,3,4], [1,4,5], [1,5,2],
        [4,3,2], [2,5,4], [6,7,8]
    ])                                  # faces by vertex index
    tex_map = np.array([
        [1,0], [0.5,0.5], [0,1], [0.5,1.5],
        [1,2], [1.5,1.5], [2,1], [1.5,0.5]
    ])                                  # vertex texture coordinate
    tex_face_map = np.array([
        [1,8,2], [3,2,4], [5,4,6], [7,6,8],
        [6,8,2], [2,4,6], [1,8,2]
    ])                                  # faces by texture index
    with open(os.path.join(output_path), 'w') as f:
        # if use texture, prepare material file and texture image
        if use_tex:
            mtl_file = output_path[:-4] + '.mtl'
            mtl_base = os.path.basename(mtl_file)
            tex_file = output_path[:-4] + '.png'
            tex_base = os.path.basename(tex_file)
            f.write(f'mtllib {mtl_base}\n')
            n_row = int(np.ceil(np.sqrt(camera_num)))
            im_size = n_row * tex_cir_rad * 2
            tex_im = np.zeros([im_size, im_size, 3], dtype=np.uint8)
        # write vertices
        for i in range(camera_num):
            verts = np.concatenate([cam_verts, np.ones((len(cam_verts),1))], axis=1)
            for j in range(verts.shape[0]):
                p = np.dot(c2ws[i], np.transpose(verts[j]))[:3]
                rgb = list(rgbs[i]) if rgbs is not None else [0, 0, (i+1)/camera_num]
                if not use_tex:
                    f.write('v %f %f %f %f %f %f\n' % tuple(list(p) + rgb))  # vertex coloring
                else:
                    x, y = i % n_row, i // n_row
                    cam_id = i if camera_ids is None else camera_ids[i]
                    cx, cy = int((x*2+1)*tex_cir_rad), int((y*2+1)*tex_cir_rad)
                    tex_im = cv2.circle(tex_im, (cx, cy), tex_cir_rad, [int(c*255) for c in rgb],  cv2.FILLED)
                    tex_im = cv2.putText(tex_im, '%02d'%(cam_id), (int((x*2+0.64)*tex_cir_rad), int((y*2+1.2)*tex_cir_rad)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, [255, 255, 255], thickness=2)
                    f.write('v %f %f %f\n' % tuple(list(p)))
        # write texture
        if use_tex:
            for view_id in range(camera_num):
                x, y = view_id % n_row, view_id // n_row
                for tex in tex_map:
                    tex = ((np.array([x,y]) * 2 + tex) * tex_cir_rad) / im_size
                    tex[1] = 1 - tex[1]
                    f.write('vt %f %f\n' % tuple(list(tex)))
            f.write('usemtl mymtl\n')
            cv2.imwrite(tex_file, tex_im)
            with open(mtl_file, 'w') as f_mtl:
                f_mtl.write('newmtl mymtl\n')
                f_mtl.write('map_Kd {}\n'.format(tex_base))
        # write faces
        for i in range(camera_num):
            face_step = i * cam_verts.shape[0]
            tex_step = i * tex_map.shape[0]
            for j in range(face_map.shape[0]):
                face = face_map[j] + face_step
                if not use_tex:
                    f.write('f %d %d %d\n' % tuple(list(face)))
                else:
                    tex_face = tex_face_map[j] + tex_step
                    face = np.stack([face, tex_face], axis=0).T.reshape(-1)
                    f.write('f %d/%d %d/%d %d/%d\n' % tuple(list(face)))

class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']
        num_cams = len(self.cams['K'])
        view = cfg.training_view
        self.views = view
        self.all_views = view

        # prepare input images
        i = 0
        i = i + cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame
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
        # from pdb import set_trace as st
        self.Ks, self.Rs, self.Ts, self.Ds = [], [], [], []
        for view in self.views:
            cam_ind = self.all_views.index(view)
            K = np.array(self.cams['K'][cam_ind], dtype=np.float32)
            R = np.array(self.cams['R'][cam_ind], dtype=np.float32)
            T = np.array(self.cams['T'][cam_ind], dtype=np.float32)
            self.Rs.append(R[:3, :3])
            self.Ts.append(T)
            self.Ks.append(K)

    def get_mask(self, index, view):
        basename = os.path.basename(self.ims[index][view][:-4])
        msk_path = os.path.join(self.data_root,
                                self.human, 'MASK',
                                f'cam_{view:02d}',
                                basename + '.png')
        
        msk_cihp = imageio.imread(msk_path)
        msk = (msk_cihp != 0).astype(np.uint8) # msk: 0 1

        border = 5
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(msk.copy(), kernel)
        msk_dilate = cv2.dilate(msk.copy(), kernel)
        msk[(msk_dilate - msk_erode) == 1] = 0 # msk: 0 1 100
        return msk

    def prepare_inside_pts(self, pts, i):
        sh = pts.shape
        pts3d = pts.reshape(-1, 3)

        inside = np.ones([len(pts3d)]).astype(np.uint8)
        # for nv in range(self.ims.shape[1]):
        for vidx, nv in enumerate(self.views):
            ind = inside == 1
            pts3d_ = pts3d[ind]

            # RT = np.concatenate([self.Rs[i], self.Ts[i]], axis=1)
            RT = np.concatenate([self.Rs[vidx], self.Ts[vidx][..., None]], axis=1)

            pts2d = base_utils.project(pts3d_, self.Ks[vidx], RT)

            msk = self.get_mask(i, vidx)
            H, W = msk.shape
            pts2d = np.round(pts2d).astype(np.int32)
            pts2d[:, 0] = np.clip(pts2d[:, 0], 0, W - 1)
            pts2d[:, 1] = np.clip(pts2d[:, 1], 0, H - 1)
            msk_ = msk[pts2d[:, 1], pts2d[:, 0]]

            inside[ind] = msk_

        inside = inside.reshape(*sh[:-1])

        return inside
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
        i = index
        latent_index = index
        frame_index = index + cfg.begin_ith_frame

        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(
            frame_index)

        mesh_param = self.prepare_mesh_param(i)

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
        print(f'latent_index: {latent_index}')
        meta = {
            'wbounds': can_bounds,
            'bounds': bounds,
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'center': mesh_param['center'],
            'spatial_freq': mesh_param['spatial_freq'],
            'voxel_size': voxel_size[0]
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
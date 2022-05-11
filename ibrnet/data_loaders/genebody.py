import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from .ply_utils import load_ply, load_obj_mesh
# from pdb import set_trace  as st

class GeneBodyDataset(Dataset):
    def __init__(self, args, mode,
                 scenes=(), **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/genebody')
        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val
        self.is_train = self.mode == 'train'
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip if not self.is_train else 1
        self.train_n_frames = args.train_n_frames
        self.scenes = scenes
        print("loading {} scenes for {}".format(len(scenes), mode))
        self.render_rgb_files = []
        self.mask_files = []
        self.render_poses = []
        self.render_intrinsics = []
        self.near_far_list = []
        self.mesh_param_list = []
        self.Ks = {}
        self.Rts = {}
        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            annotation = np.load(os.path.join(self.scene_path, 'annots.npy'), allow_pickle=True).item()['cams']
            rgb_files, mask_files, intrinsics, poses = self.read_cameras(self.scene_path, annotation)
            self.Ks[scene] = np.array([annotation[k]['K'] for k in annotation.keys()]).astype(np.float32)
            self.Rts[scene] = np.array([annotation[k]['c2w'] for k in annotation.keys()]).astype(np.float32)
            self.Rts[scene][:, :3, 3] /= 2.87
            self.render_rgb_files.extend(rgb_files)
            self.render_poses.extend(poses)
            self.render_intrinsics.extend(intrinsics)
            self.mask_files.extend(mask_files)
        print(f'dataset loaded: {len(self.render_rgb_files)} images loaded.')

    def __len__(self):
        return len(self.render_rgb_files)


    def get_near_far(self, smpl_verts, w2c):
        vp = smpl_verts.dot(w2c[:3,:3].T) + w2c[:3,3:].T
        vmin, vmax = vp.min(0), vp.max(0)
        near, far = vmin[2], vmax[2]
        near, far = near-(far-near)/2, far+(far-near)/2
        return near, far

    def get_realworld_scale(self, smpl_verts, w2c, K):
        smpl_min, smpl_max = smpl_verts.min(0), smpl_verts.max(0)
        # reprojected smpl verts
        vp = smpl_verts.dot(w2c[:3,:3].T) + w2c[:3,3:].T
        vp = vp.dot(K[:3, :3].T)
        vp = vp[:,:2] / (vp[:,2:]+1e-8)
        vmin, vmax = vp.min(0), vp.max(0)
        # compare with bounding box
        bbox_h = 512
        bbox_w = 512
        long_axis = bbox_h/(vmax[1]-vmin[1])*(smpl_max[1]-smpl_min[1]) if bbox_h > bbox_w else bbox_w/(vmax[0]-vmin[0])*(smpl_max[0]-smpl_min[0])
        spatial_freq = 180/long_axis/0.5

        return spatial_freq


    def read_cameras(self, scene_dir, annotation):
        rgb_files = []
        mask_files = []
        c2w_mats = []
        intrinsic_list = []
        all_views = list(range(48))
        if 'joseph' in scene_dir:
            all_views = list(set(all_views) - set([39, 40, 42, 43, 44, 45, 46, 47]))
        for cam_idx, cam_id in enumerate(all_views):
            cnt = 0
            image_dir = os.path.join(scene_dir, 'image', f'{cam_id:02d}')
            for frame_idx, frame_name in enumerate(sorted(os.listdir(image_dir))[0::self.testskip]):
                if not os.path.exists(image_dir):
                    print(image_dir)
                    continue
                cnt += 1
                if self.is_train and cnt >= self.train_n_frames:
                    break
                elif not self.is_train and cnt <= (self.train_n_frames // self.testskip):
                    continue
                rgb_file = os.path.join(image_dir, frame_name)
                mask_file = os.path.join(image_dir.replace('image', 'mask'), f'mask{int(frame_name[:-4]):04d}.png')
                rgb_files.append(rgb_file)
                mask_files.append(mask_file)
                intrinsic = np.eye(4)
                K = annotation[f'{cam_idx:02d}']['K'].astype(np.float32)
                intrinsic[0][0] = K[0][0]
                intrinsic[1][1] = K[1][1]
                intrinsic[0][2] = K[0][2]
                intrinsic[1][2] = K[1][2]
                intrinsic_list.append(intrinsic)
                c2w = annotation[f'{cam_idx:02d}']['c2w'].astype(np.float32)
                c2w[:3, 3] /= 2.87
                c2w_mats.append(c2w)
        return rgb_files, mask_files, intrinsic_list, c2w_mats

                

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        mask_file = self.mask_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx].copy()
        vis_cam_poses = []
        vis_cam_poses.append(render_pose)

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        mask = imageio.imread(mask_file).astype(np.float32) / 255.
        mask = (mask > 0.5).astype(np.uint8)
        rgb = rgb * mask[..., None]
        # crop image and resize
        a = np.where(mask != 0)
        size = mask.shape[:2]
        h, w = list(size)
        bbox = [[0, np.min(a[1])], [mask.shape[0], np.max(a[1])]]
        top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
        bbox_h, bbox_w = bottom - top, right - left
        if bbox_h > bbox_w:
            crop_bottom = min(int(bbox_h*0.1+bottom), h)
            crop_top = max(int(top-bbox_h*0.1), 0)
            crop_size = crop_bottom - crop_top
            crop_right = min(int((right + left)/2 + crop_size/2), w)
            crop_left = max(int((right + left)/2 - crop_size/2), 0)
        else:
            crop_right = min(int(bbox_w*0.1+right), w)
            crop_left = max(int(left-bbox_w*0.1), 0)
            crop_size = crop_right - crop_left
            crop_bottom = min(int((top + bottom)/2 + crop_size/2), h)
            crop_top = max(int((top + bottom)/2 - crop_size/2), 0)

        mask = mask[crop_top:crop_bottom, crop_left:crop_right]
        rgb = rgb[crop_top:crop_bottom, crop_left:crop_right]
        rgb = cv2.resize(rgb, (512 ,512), \
                            interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (512 ,512), \
                            interpolation = cv2.INTER_NEAREST)
        render_intrinsics[0,2] -= crop_left
        render_intrinsics[1,2] -= crop_top
        render_intrinsics[0,:] *= 512 / float(crop_right - crop_left)
        render_intrinsics[1,:] *= 512  / float(crop_bottom - crop_top)
        scene_name_ref, foo, cam_id_ref, bar = self.render_rgb_files[idx].split('/')[-4:]

        # read near far by K, RT, smpl
        mesh_param = {}
        frame_name = rgb_file.split('/')[-1]
        smpl_path = os.path.join(self.folder_path, scene_name_ref, 'smpl',f'{int(frame_name[:-4]):04d}.obj')
        vert, face = load_obj_mesh(smpl_path)
        vert /= 2.87
        center = (vert.max(0)+vert.min(0))/2
        w2c = np.linalg.inv(render_pose)
        spatial_freq = self.get_realworld_scale(vert, w2c, render_intrinsics)
        mesh_param = {'center': center, 'spatial_freq': spatial_freq}
        near, far = self.get_near_far(vert, w2c)

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), render_intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)
        prefix = '/' + os.path.join(*self.render_rgb_files[idx].split('/')[:-4])
        src_rgb_paths = []
        src_mask_paths = []

        for cam_id in ['01', '13', '25', '37']: # four source views as GNR
            src_rgb_path = os.path.join(prefix, scene_name_ref, foo, cam_id, bar)
            mask_path = os.path.join(prefix, scene_name_ref, 'mask', cam_id, f'mask{int(bar[:-4]):04d}.png')
            src_rgb_paths.append(src_rgb_path)
            src_mask_paths.append(mask_path)
        src_rgbs = []
        src_cameras = []
        for src_rgb_path, src_mask_path in zip(src_rgb_paths, src_mask_paths):
            scene_name_ref, foo, cam_id, bar = src_rgb_path.split('/')[-4:]
            src_rgb = imageio.imread(src_rgb_path).astype(np.float32) / 255.
            mask = imageio.imread(src_mask_path).astype(np.float32) / 255.
            mask = (mask > 0).astype(np.uint8)
            src_rgb = src_rgb * mask[..., None]
            # crop image and resize
            a = np.where(mask != 0)
            size = mask.shape[:2]
            h, w = list(size)
            bbox = [[0, np.min(a[1])], [mask.shape[0], np.max(a[1])]]
            top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(a[0]), np.max(a[1])
            bbox_h, bbox_w = bottom - top, right - left
            if bbox_h > bbox_w:
                crop_bottom = min(int(bbox_h*0.1+bottom), h)
                crop_top = max(int(top-bbox_h*0.1), 0)
                crop_size = crop_bottom - crop_top
                crop_right = min(int((right + left)/2 + crop_size/2), w)
                crop_left = max(int((right + left)/2 - crop_size/2), 0)
            else:
                crop_right = min(int(bbox_w*0.1+right), w)
                crop_left = max(int(left-bbox_w*0.1), 0)
                crop_size = crop_right - crop_left
                crop_bottom = min(int((top + bottom)/2 + crop_size/2), h)
                crop_top = max(int((top + bottom)/2 - crop_size/2), 0)

            mask = mask[crop_top:crop_bottom, crop_left:crop_right]
            src_rgb = src_rgb[crop_top:crop_bottom, crop_left:crop_right]
            src_rgb = cv2.resize(src_rgb, (512 ,512), \
                                interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (512 ,512), \
                                interpolation = cv2.INTER_NEAREST)
            pose = self.Rts[scene_name_ref][int(cam_id)]
            intri = self.Ks[scene_name_ref][int(cam_id)].copy()
            vis_cam_poses.append(pose)
            intri[0,2] -= crop_left
            intri[1,2] -= crop_top
            intri[0,:] *= 512 / float(crop_right - crop_left)
            intri[1,:] *= 512  / float(crop_bottom - crop_top)
            # this repo use 4*4 intrinsic
            _intri = np.eye(4).astype(np.float32)
            _intri[:3, :3] = intri
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), _intri.flatten(),
                                              pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)
        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        near_depth = near
        far_depth = far

        depth_range = torch.tensor([near_depth, far_depth])
     
        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'mesh_param': mesh_param
                }


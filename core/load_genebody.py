from ast import Param
from asyncore import write
from cgi import test
import imp
import os
import h5py
import torch
import imageio
import numpy as np
from smplx import SMPL, SMPLX
import smplx
import torch.nn as nn


from .dataset import BaseH5Dataset
from .process_spin import SMPL_JOINT_MAPPER, SMPLX_JOINT_MAPPER, write_to_h5py
from .utils.skeleton_utils import *


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)




def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))



class SMPLx:
    def __init__(self, model_path, use_openpose=False):
        if use_openpose:
            joint_mapper = JointMapper(smpl_to_openpose("smplx", use_hands=True,
                                    use_face=True,
                                    use_face_contour=True,
                                    openpose_format="coco25"))
        else:
            joint_mapper = None
        model_params = dict(model_path=model_path,
                            model_type='smplx',
                            joint_mapper=joint_mapper,
                            ext='npz',
                            gender='neutral',
                            create_global_orient=True,
                            create_body_pose=True,
                            create_betas=True,
                            create_left_hand_pose=True,
                            create_right_hand_pose=True,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=False,
                            use_face_contour=True,
                            dtype=torch.float32,
                            use_pca=False)
        self.smpl = smplx.create(**model_params)

    def __call__(self, model_params):
        out = self.smpl(**model_params)
        return out.vertices, out.joints

    def get_faces(self):
        return self.smpl.faces

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

def get_views(subject):
    all_views_raw = list(range(48))
    if subject == 'Tichinah_jervier' or subject == 'dannier':
        all_views = list(set(all_views_raw) - set([32]))
    elif subject == 'wuwenyan':
        all_views = list(set(all_views_raw)-set([34, 36]))
    elif subject == 'joseph_matanda':
        all_views = list(set(all_views_raw) - set([39, 40, 42, 43, 44, 45, 46, 47]))
    else:
        all_views = all_views_raw
    
    return all_views

def get_frames(root_dir, subject):
    frame_list = []
    frame_list = os.listdir(os.path.join(root_dir, subject, 'image', '00'))
    frame_list = sorted(frame_list)
    return frame_list

def get_annot(self, subject):
    return np.load(os.path.join(self.rootdir, subject, 'annots.npy'), allow_pickle=True).item()['cams']

def get_data(self, subject, frame_list, all_views, annots, frame_id, views):
    """
    Fetch one frame of multiview data from database with cropping
    subject: name of subject
    frame_list: all frames of the subject <- self.get_frames(subject)
    all_views: all views of subject <- self.get_views(subject)
    annots: camera parameters <- self.get_annot(subject)
    frame_id: eg. 1
    views: list of views to fetch, eg. load sourceviews through self.sourceviews,
            or all view through self.get_views(subject)
    """
    subject_dir = os.path.join(self.rootdir, subject)
    
    Ks, Rts, images, masks = [], [], [], []
    for view in views:
        img = imageio.imread(os.path.join(subject_dir, 'image', '{:02d}'.format(view), frame_list[frame_id]))
        msk = imageio.imread(os.path.join(subject_dir, 'mask', '{:02d}'.format(view), f'mask{frame_list[frame_id][:-4]}.png'))
        # crop the human out from raw image            
        top, left, bottom, right = image_cropping(msk)
        img = img * (msk > 128)[...,None]
        # resize to uniform resolution
        img = cv2.resize(img[top:bottom, left:right].copy(), (self.loadsize, self.loadsize), cv2.INTER_CUBIC)
        images.append(img)
        msk = cv2.resize(msk[top:bottom, left:right].copy(), (self.loadsize, self.loadsize), cv2.INTER_NEAREST)
        masks.append(msk)
        # retrive the view index from all views
        i = all_views.index(view)
        # adjust the camera intrinsic parameter because of the cropping and resize
        # Note that there is no need to adjust extrinsic or distortation coefficents
        K, Rt = annots['K'][i].copy(), annots['RT'][i].copy()
        K[0,2] -= left
        K[1,2] -= top
        K[0,:] *= self.loadsize / float(right - left)
        K[1,:] *= self.loadsize / float(bottom - top)
        Ks.append(K.astype(np.float32))
        # Rt is a camera to world homogenous rotation
        Rts.append(Rt.astype(np.float32))

    return images, masks, Ks, Rts

def get_mask(path, img_path, erode_border=False):
    '''
    Following NeuralBody repo
    https://github.com/zju3dv/neuralbody/blob/master/lib/datasets/light_stage/can_smpl.py#L46    '''

    mask_path = os.path.join(path, 'mask', img_path[:-4] + '.png')
    mask = None
    if os.path.exists(mask_path):
        mask = imageio.imread(mask_path)
        mask = (mask != 0).astype(np.uint8)

    mask_path = os.path.join(path, 'mask_cihp', img_path[:-4] + '.png')
    mask_cihp = None
    if os.path.exists(mask_path):
        mask_cihp = imageio.imread(mask_path)
        mask_cihp = (mask_cihp != 0).astype(np.uint8)
    
    if mask is not None and mask_cihp is not None:
        mask = (mask | mask_cihp).astype(np.uint8)
    elif mask_cihp is not None:
        mask = mask_cihp
    
    border = 5
    kernel = np.ones((border, border), np.uint8)
    #mask_erode = cv2.erode(mask.copy(), kernel)
    sampling_mask = cv2.dilate(mask.copy(), kernel, iterations=3)

    #sampling_mask = mask.copy()
    #sampling_mask[(mask_dilate - mask_erode) == 1] = 100
    if erode_border:
        dilated = cv2.dilate(mask.copy(), kernel) 
        eroded = cv2.erode(mask.copy(), kernel) 
        sampling_mask[(dilated - eroded) == 1] = 0
    #eroded = cv2.erode(mask.copy(), kernel, iterations=1) 
    #mask = eroded.copy()

    return mask, sampling_mask

@torch.no_grad()
def get_smpls(path, kp_idxs, gender='neutral', ext_scale=1.0, scale_to_ref=True,
              ref_pose=smpl_rest_pose, param_path=None, model_path=None, vertices_path=None):
    '''
    Note: it's yet-another-smpl-coordinate system
    bodies: the vertices from ZJU dataset
    '''

    if param_path is None:
        param_path = 'new_param'
    if model_path is None:

        model_path = '../../smplx-model/'

    bones, betas, root_bones, root_locs = [], [], [], []
    joints_all = []
    mysmplx = SMPLx('../../smplx-model/', use_openpose=False)

    for kp_idx in kp_idxs:
        params = np.load(os.path.join(path, param_path, ('%04d.npy' % kp_idx)), allow_pickle=True).item()

        for key, val in params.items():
            if key not in ['betas', 'left_hand_pose', 'right_hand_pose']:
                params[key] = torch.from_numpy(val).unsqueeze(0)
            else:
                params[key] = torch.from_numpy(val)

        verts, joints = mysmplx(params)
        verts = verts.squeeze(0).numpy() / 2.87
        joints = joints.squeeze(0).numpy() / 2.87

        joints = joints[:22]
        joints_all.append(joints)


        bone = params['body_pose'].reshape(-1, 21, 3)
        add_hand = np.concatenate((params['left_hand_pose'][0], params['right_hand_pose'][0]), axis=0)
        bone = np.concatenate((np.zeros((1, 1, 3)), bone), axis=1)
        bone = np.concatenate((bone, add_hand.reshape(-1, 2, 3)), axis=1)
        beta = params['betas']

        root_bones.append(params['global_orient'])
        root_locs.append(params['transl'])
        bones.append(bone)
        betas.append(beta)


    #print(len(betas))
    bones = np.concatenate(bones, axis=0).reshape(-1, 3)
    betas = np.concatenate(betas, axis=0).reshape(-1, 10)
    root_bones = np.concatenate(root_bones, axis=0).reshape(-1, 3)

    betas = torch.FloatTensor(betas)


    # 2. get the rest pose
    dummy = {'body_pose': torch.zeros((1, 21, 3))}
    v, j = mysmplx(dummy)
    pose_scale = 1 / 2.87
    j = j.squeeze(0).numpy() * pose_scale
    rest_pose = j[:22]

    rest_pose -= rest_pose[0] # center rest pose

    
    joints_all = np.array(joints_all)

    root_locs = joints_all[:, 0]
    bones = bones.reshape(-1, 24, 3)
    bones[:, 0] = root_bones
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose=rest_pose, skel_type=SMPLSkeleton) for bone in bones])
    l2ws[..., :3, -1] += root_locs[:, None]

    kp3d = l2ws[..., :3, -1]

    skts = np.array([np.linalg.inv(l2w) for l2w in l2ws])

    
    return betas, kp3d, bones, skts, rest_pose, pose_scale




def process_genebody_data(data_path, subject='zhuna', training_view=None,
                          i_intv=1, split='train',
                          ext_scale=0.001, res=None, skel_type=SMPLXSkeleton):
    '''
    ni, i_intv, intv, begin_i: setting from NeuralBody
    '''
    assert ext_scale == 0.001 # TODO: deal with this later
    # TODO: might have to check if this is true for all

    loadsize = H = W = 512 # default image size.
    ni = 75
    begin_i = 0

    if res is not None:
        H = int(H * res)
        W = int(W * res)

    subject_path = os.path.join(data_path, subject)
    annot_path = os.path.join(subject_path, "annots.npy")
    annots = np.load(annot_path, allow_pickle=True).item()
    cams = annots['cams']
    num_cams = len(cams['K'])
    i = begin_i
    img_paths = []
    mask_paths = []


    frames_list = get_frames(data_path, subject)
    views = get_views(subject)

    if split == 'train':
        frames_list = frames_list[:ni]
    else:
        frames_list = frames_list[ni:]

    cam_idxs = views * len(frames_list)


    for frame_id in frames_list:
        img_paths.append([os.path.join(subject_path, 'image', '{:02d}'.format(view), frame_id) for view in views])
        mask_paths.append([os.path.join(subject_path, 'mask', '{:02d}'.format(view), f'mask0{frame_id[:-4]}.png') for view in views])
    img_paths = np.array(img_paths).ravel()
    mask_paths = np.array(mask_paths).ravel()


    imgs = np.zeros((len(img_paths), H, W, 3), dtype=np.uint8)
    masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    sampling_masks = np.zeros((len(img_paths), H, W, 1), dtype=np.uint8)
    c2ws, focals, centers = [], [], []

    kp_idxs = []
    for i, (img_path, msk_path, cam_idx) in enumerate(zip(img_paths, mask_paths, cam_idxs)):

        if i % 50 == 0:
            print(f'{i+1}/{len(img_paths)}')

        idx = views.index(cam_idx)

        img = imageio.imread(img_path)
        msk = imageio.imread(msk_path)

        # crop the human out from raw image            
        top, left, bottom, right = image_cropping(msk)
        img = img * (msk > 128)[...,None]
        # resize to uniform resolution
        img = cv2.resize(img[top:bottom, left:right].copy(), (H, W), cv2.INTER_CUBIC)

        msk = cv2.resize(msk[top:bottom, left:right].copy(), (H, W), cv2.INTER_NEAREST)
        msk = np.expand_dims(msk, axis=2)
        sampling_msk = msk.copy()
        msk[msk <= 128] = 0
        msk[msk > 128] = 1

        # retrive the view index from all views

        # adjust the camera intrinsic parameter because of the cropping and resize
        # Note that there is no need to adjust extrinsic or distortation coefficents
        K= cams['K'][idx].copy()

        Rt = np.array(cams['RT'][idx], dtype=np.float32)
        # Rt = np.linalg.inv(Rt)
        R = Rt[:3, :3]
        T = Rt[:3, 3]
        T = T[..., None]

        K[0,2] -= left
        K[1,2] -= top
        K[0,:] *= loadsize / float(right - left)
        K[1,:] *= loadsize / float(bottom - top)

        # get camera-to-world matrix from extrinsic
        ext = np.concatenate([R, T], axis=-1)
        ext = np.concatenate([ext, np.array([[0, 0, 0., 1.]])], axis=0)

        c2ws.append(ext)

        # save intrinsic data
        if res is not None:
            K[:2] = K[:2] * res
        focals.append([K[0, 0], K[1, 1]])
        centers.append(K[:2, -1])


        kp_idx = int(os.path.basename(img_path)[:-4])

        imgs[i] = img
        masks[i] = msk
        sampling_masks[i] = sampling_msk
        kp_idxs.append(kp_idx)


    focals = np.array(focals)
    centers = np.array(centers)
    c2ws = np.array(c2ws).astype(np.float32)
    c2ws = swap_mat(c2ws) # to NeRF format

    unique_cams = np.unique(cam_idxs) ###[0, 6, 12, 18]
    bkgds = np.zeros((num_cams, H, W, 3), dtype=np.uint8)


    # get pose-related data
    betas, kp3d, bones, skts, rest_pose, pose_scale = get_smpls(subject_path, np.unique(kp_idxs),
                                                                          scale_to_ref=False)

    #print(kp3d)
    skeletons = draw_skeletons_3d(imgs[480:], kp3d.reshape(-1, 1, 22, 3).repeat(48, 1).reshape(-1, 22, 3),
                            c2ws=c2ws, H=H, W=W, focals=focals, centers=centers, skel_type=SMPLXSkeleton)
    for i, skel in enumerate(skeletons):
        basedir = 'test_smpl_cal'
        imageio.imwrite(os.path.join(basedir, 'skel', f'{i:05d}.png'), skel)

    cyls = get_kp_bounding_cylinder(kp3d,
                                    ext_scale=ext_scale,
                                    skel_type=skel_type,
                                    extend_mm=300,
                                    top_expand_ratio=1.00,
                                    bot_expand_ratio=0.25,
                                    head='-y')## question why 3 * 5 represent a cylinder

    kp_idxs = np.array(kp_idxs) - kp_idxs[0]

    return {'imgs': np.array(imgs),
            'bkgds': np.array(bkgds),
            'bkgd_idxs': np.array(cam_idxs),
            'masks': np.array(masks).reshape(-1, H, W, 1),
            'sampling_masks': np.array(sampling_masks).reshape(-1, H, W, 1),
            'c2ws': c2ws.astype(np.float32),
            'img_pose_indices': np.array(cam_idxs),
            'kp_idxs': np.array(kp_idxs),
            'centers': centers.astype(np.float32),
            'focals': focals.astype(np.float32),
            'kp3d': kp3d.astype(np.float32),
            'betas': betas.numpy().astype(np.float32),
            'bones': bones.astype(np.float32),
            'skts': skts.astype(np.float32),
            'cyls': cyls.astype(np.float32),
            'rest_pose': rest_pose.astype(np.float32),
            'views': np.array(views)
            }



class GenebodyDataset(BaseH5Dataset):

    N_render = 15
    render_skip = 63

    def __init__(self, *args, **kwargs):
        super(GenebodyDataset, self).__init__(*args, **kwargs)

    def init_meta(self):
        if self.split == 'test':
            self.h5_path = self.h5_path.replace('train', 'test')
        super(GenebodyDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]

        if self.split == 'test':
            n_unique_cam = len(np.unique(self.cam_idxs))
            self.kp_idxs = self.kp_idxs // n_unique_cam

        self.skel_type = SMPLXSkeleton
        print('WARNING: genebody does not support pose refinement for now (_get_subset_idxs is not implemented)')
        dataset.close()

    def get_meta(self):
        data_attrs = super(GenebodyDataset, self).get_meta()
        return data_attrs

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.kp_idxs[idx], q_idx

    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        #return self.cam_idxs[idx], q_idx
        return idx, q_idx


    def _get_subset_idxs(self, render=False):
        '''
        get the part of data that you want to train on
        '''
        if self._idx_map is not None:
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))
        else:
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs




    
if __name__ == '__main__':
    #from renderer import Renderer
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for masks extraction')
    parser.add_argument("-d", "--dataset", type=str, default='genebody')
    parser.add_argument("-s", "--subject", type=str, default="zhuna",
                        help='subject to extract')
    parser.add_argument("--split", type=str, default="test",
                        help='split to use')
    args = parser.parse_args()
    dataset = args.dataset
    subject = args.subject
    split = args.split


    data_path = '../data/genebody_origin/'
    write_path = '../data/genebody/'
    print(f"Processing {subject}_{split}...")
    data = process_genebody_data(data_path, subject, split=split, res=1.0)
    process_spin.write_to_h5py(os.path.join(write_path, f"{subject}_{split}.h5"), data)
    write_to_h5py(os.path.join(data_path, f"{subject}_{split}.h5"), data)


import os, sys
from types import CoroutineType
import numpy as np
import torch
import torch.nn as nn
import smplx
import cv2
import imageio


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

def persp_proj(pts, w2c, K):
    pts = pts.reshape(-1,3)
    vp = pts.dot(w2c[:3,:3].T) + w2c[:3,3:].T
    vp = vp / (vp[:, 2:] + 1e-8)
    vp = vp.dot(K.T)
    return vp[:, :2]

if __name__ == '__main__':

    datadir = '../data/genebody'
    subject = 'zhuna'
    frame = 0
    param_paths = sorted(os.listdir(os.path.join(datadir, subject, 'new_param')))
    param = np.load(os.path.join(datadir, subject, 'new_param', param_paths[frame]), allow_pickle=True).item()
    for key, val in param.items():
        if key not in ['betas', 'left_hand_pose', 'right_hand_pose']:
            param[key] = torch.from_numpy(val).unsqueeze(0)
        else:
            param[key] = torch.from_numpy(val)

    mysmplx = SMPLx('../..//smplx-model/', use_openpose=False)
    verts, joints = mysmplx(param)
    verts = verts.squeeze(0).numpy() / 2.87
    joints = joints.squeeze(0).numpy() / 2.87
    # print(joints.shape)
    joints = joints[:22]
    print(joints)
    annot = np.load(os.path.join(datadir, subject, 'annots.npy'), allow_pickle=True).item()['cams']
    views = sorted(os.listdir(os.path.join(datadir, subject, 'image')))
    imgname = sorted(os.listdir(os.path.join(datadir, subject, 'image', '00')))[frame]
    for K, c2w, view in zip(annot['K'], annot['RT'], views):
        
        img = imageio.imread(os.path.join(datadir, subject, 'image', view, imgname))
        msk = imageio.imread(os.path.join(datadir, subject, 'mask', view, 'mask0' + imgname[:-4] + '.png'))
        # crop the human out from raw image            
        top, left, bottom, right = image_cropping(msk)
        img = img * (msk > 128)[...,None]
        # resize to uniform resolution
        img = cv2.resize(img[top:bottom, left:right].copy(), (512, 512), cv2.INTER_CUBIC)

        msk = cv2.resize(msk[top:bottom, left:right].copy(), (512, 512), cv2.INTER_NEAREST)
        msk = np.expand_dims(msk, axis=2)
        sampling_msk = msk.copy()
        msk[msk <= 128] = 0
        msk[msk > 128] = 1

        K[0,2] -= left
        K[1,2] -= top
        K[0,:] *= 512 / float(right - left)
        K[1,:] *= 512 / float(bottom - top)


        joints_2d = persp_proj(joints, np.linalg.inv(c2w), K)
        joints_2d[:, 0] = np.clip(joints_2d[:, 0], 0, img.shape[1])
        joints_2d[:, 1] = np.clip(joints_2d[:, 1], 0, img.shape[1])
        for j in joints_2d:
            img = cv2.circle(img, tuple(j.astype(np.int32)), 5, [0, 255 ,0], thickness=5)

        imageio.imwrite(f'./logs_crop/{view}.png', img)
        print(f'./logs/{view}.png')
    # print(joints.shape)
    
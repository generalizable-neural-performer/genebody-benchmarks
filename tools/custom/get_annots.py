import cv2
import numpy as np
import glob
import os
import json
import sys
import json

import pickle

calib_file = sys.argv[1]


def get_cams():
    intri = cv2.FileStorage('intri.yml', cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage('extri.yml', cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for i in range(23):
        cams['K'].append(intri.getNode('K_Camera_B{}'.format(i + 1)).mat())
        cams['D'].append(
            intri.getNode('dist_Camera_B{}'.format(i + 1)).mat().T)
        cams['R'].append(extri.getNode('Rot_Camera_B{}'.format(i + 1)).mat())
        cams['T'].append(extri.getNode('T_Camera_B{}'.format(i + 1)).mat() * 1000)
    return cams

def get_cams_aistpp():
    with open(calib_file) as f:
        calib_data = json.load(f)
    cams_data = {}
    for cam_data in calib_data:
        cams_data[cam_data['name']] = {}
        cams_data[cam_data['name']]['K'] = cam_data['matrix']
        cams_data[cam_data['name']]['D'] = cam_data['distortions']
        cams_data[cam_data['name']]['R'] = cam_data['rotation']
        cams_data[cam_data['name']]['T'] = cam_data['translation']

    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for i in range(2, 10):
        cams['K'].append(cams_data[f'c{i:02d}']['K'])
        cams['D'].append(cams_data[f'c{i:02d}']['D'])
        cams['R'].append(cams_data[f'c{i:02d}']['R'])
        cams['T'].append(cams_data[f'c{i:02d}']['T'])

    return cams


def get_img_paths():
    all_ims = []
    for i in range(1, 9):
        i = i + 1
        data_root = f'/mnt/lustre/share_data/xusu/AISTpp/ch01/frames/gWA_sFM_c{i:02d}_d25_mWA0_ch01'
        ims = glob.glob(os.path.join(data_root, '*.png'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


def get_2dkpts():
    all_kpts = []
    with open('/mnt/lustre/share_data/xusu/AISTpp/annos/aist_plusplus_final/keypoints2d/gWA_sFM_cAll_d25_mWA0_ch01.pkl', 'rb') as f:
        data = pickle.load(f)
    for i in range(2, 10):
        kpts = data['keypoints2d']
        kpts_cam = kpts[i - 1]
        all_kpts.append(kpts_cam)
    num_kpts = min([len(ims) for ims in all_kpts])
    all_kpts = [kpt[:num_kpts] for kpt in all_kpts]
    all_kpts = np.stack(all_kpts, axis=1)
    return all_kpts




        

    

cams = get_cams_aistpp()
img_paths = get_img_paths()
kpts2d = get_2dkpts()

annot = {}
annot['cams'] = cams

ims = []
for img_path, kpt in zip(img_paths, kpts2d):
    data = {}
    data['ims'] = img_path.tolist()
    data['kpts2d'] = kpt.tolist()
    ims.append(data)
    
annot['ims'] = ims
# annot['kpts2d'] = kpt

np.save('annots.npy', annot)
# np.save('annots_python2.npy', annot, fix_imports=True)

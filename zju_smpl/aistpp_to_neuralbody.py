import os
import sys
import json
import pickle

import numpy as np
import torch
sys.path.append("../")
from smplmodel.body_model import SMPLlayer
from pdb import set_trace as st

# easymocap_params = json.load(open('zju_smpl/example.json'))[0]
# easymocap_params = json.load(open('setting10_2.json'))
aistpp_motion_path = sys.argv[1]
model_dir  = sys.argv[2]
dst_dir = sys.argv[3]


with open(aistpp_motion_path, 'rb') as f:
    aistpp_motion_params = pickle.load(f)
poses = np.array(aistpp_motion_params['smpl_poses'])
# Rh = np.array(aistpp_motion_params['Rh'])
Rh = np.identity(3)
Th = np.array(aistpp_motion_params['smpl_trans'])
# shapes = np.array(aistpp_motion_params['shapes'])
shapes = torch.zeros((1, 10))


# the params of neural body
params = {'poses': poses, 'Rh': Rh, 'Th': Th, 'shapes': shapes}
# st()
import open3d as o3d


for obj_file in os.listdir(model_dir):
    # st()
    frame_id = int(obj_file.split('.')[0])
    from copy import deepcopy
    param = deepcopy(params)
    param['poses'] = param['poses'][frame_id - 1]
    np.save(os.path.join(dst_dir, "params", f"{frame_id}.npy"), param)
    mesh = o3d.io.read_triangle_mesh(os.path.join(model_dir, obj_file))
    verts = np.asarray(mesh.vertices)
    np.save(os.path.join(dst_dir, "vertices", f"{frame_id}.npy"), verts)

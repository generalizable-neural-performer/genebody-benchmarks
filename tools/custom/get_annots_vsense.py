from textwrap import dedent
import cv2
import numpy as np
import glob
import os
import json
import sys
import json

import pickle
from pdb import set_trace as st
# calib_file = sys.argv[1]



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
    if len(camera_poses.shape) == 2:
        camera_poses = np.expand_dims(camera_poses, 0)
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

def get_cams_vsense(subject):
    all_cams = []
    cams = {'K': [], 'R': [], 'T': []}
    for cam_idx in range(36):
        cam_param_dir = f'data/vsense/{subject}/PARAM/cam_{cam_idx:02d}'
        sample = os.path.join(cam_param_dir, '000001.png.npy')
        param = np.load(sample, allow_pickle=True).item()
        # print(param)
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
        all_cams.append(extrinsic)
        R = extrinsic[:3, :3]
        T = extrinsic[:3, 3]
        K = intr_mat

        cams['R'].append(R)
        cams['T'].append(T)
        cams['K'].append(K)

    return cams, all_cams


def get_img_paths(subject):
    all_ims = []
    for i in range(36):
        data_root = f'data/vsense/{subject}/RENDER/cam_{i:02d}'
        ims = glob.glob(os.path.join(data_root, '*.png'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


# subjects = ['Matis_obj', 'longdress', 'loot', 'redandblack', 'soldier']
subjects = ['loot']

for subject in subjects:
    cams, all_cams = get_cams_vsense(subject)
    cam_pose_vis(f'{subject}_cams.obj', np.array(all_cams), cds='cv')
    img_paths = get_img_paths(subject)

    annot = {}
    annot['cams'] = cams

    ims = []
    for img_path in img_paths:
        data = {}
        data['ims'] = img_path.tolist()
        ims.append(data)
        
    annot['ims'] = ims
    # annot['kpts2d'] = kpt
    np.save(f'data/vsense/{subject}/annots.npy', annot)
# np.save('annots_python2.npy', annot, fix_imports=True)

import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torch.distributions as tdist

from pdb import set_trace as st
import sys, re
import struct
from tqdm import tqdm, trange
import imageio
import torchvision.transforms.functional as F



def extract_float(text):
	flts = []
	for c in re.findall('(-?[0-9]*\.?[0-9]*[eE]?[-\+]?[0-9]+)',text):
		if c != '':
			try:
				flts.append(float(c))
			except ValueError as e:
				continue
	return flts
def natural_sort(files):
	return	sorted(files, key = lambda text: \
		extract_float(os.path.basename(text)) \
		if len(extract_float(os.path.basename(text))) > 0 else \
		[float(ord(c)) for c in os.path.basename(text)])

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

def rot2euler(R):
    phi = np.arctan2(R[1,2], R[2,2])
    theta = -np.arcsin(R[0,2])
    psi = np.arctan2(R[0,1], R[0,0])
    return np.array([phi, theta, psi])

def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0],
                [0, cos(phi), sin(phi)],
                [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)],
                [0, 1, 0],
                [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0],
                [-sin(psi), cos(psi), 0],
                [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R


def circle_path_from_smpl(smpl_verts, intrinsic, img_size, num_views):
    # smpl_min, smpl_max = smpl_verts.min(0), smpl_verts.max(0)
    intrinsic = intrinsic.numpy()
    smpl_verts = smpl_verts.numpy()
    img_size = img_size[0]
    smpl_min, smpl_max = np.min(smpl_verts), np.max(smpl_verts)
    center = (smpl_min + smpl_max) / 2
    render_poses = []
    fx, fy = intrinsic[0], intrinsic[1]
    height = (smpl_max - smpl_min)
    rad = height / (img_size * 0.8) * fy
    gl2cv = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    for theta in np.linspace(0, np.pi*2, num_views+1)[:-1]:
        euler = [0, theta+np.pi/2, 0]
        R = euler2rot(euler)
        t = np.array([np.cos(theta), 0, -np.sin(theta)]) * rad + center
        Rt = np.eye(4)
        Rt[:3,:3] = gl2cv@R
        Rt[:3, 3] = t
        # render_poses.append(torch.from_numpy(np.linalg.inv(Rt)).float())        
        render_poses.append(torch.from_numpy(Rt).float())        

    return render_poses


def load_obj_mesh(mesh_file, with_normal=False, with_texture=False, with_texture_image=False):
    vertex_data = []
    norm_data = []
    uv_data = []
    vertex_color = []

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue

        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
            if len(values) >= 7:
                v_c = list(map(float, values[4:7]))
                vertex_color.append(v_c)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)

        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
            
            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
        elif 'mtllib' in line.split():
            mtlname = line.split()[-1]
            mtlfile = os.path.join(os.path.dirname(mesh_file), mtlname)
            with open(mtlfile, 'r') as fmtl:
                mtllines = fmtl.readlines()
                for mtlline in mtllines:
                    # if mtlline.startswith('map_Kd'):
                    if 'map_Kd' in mtlline.split():
                        texname = mtlline.split()[-1]
                        texfile = os.path.join(os.path.dirname(mesh_file), texname)
                        texture_image = cv2.imread(texfile)
                        texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)
                        break

    vertices = np.array(vertex_data)
    faces = np.array(face_data) - 1
    if len(vertex_color) >0:
        colors = np.array(vertex_color)
        return vertices, faces, colors

    return vertices, faces

def merge_holes(pc1,pc2):

    # change point color here

    return np.concatenate([pc1, pc2], axis=0)

def pointcloud_upsampling(verts, faces, rgbs, ratio=5):
    if isinstance(faces, np.ndarray):
        faces = torch.from_numpy(faces)
    coeff = torch.rand((faces.shape[0], ratio, 3))
    coeff = coeff / (torch.sum(coeff, dim=-1, keepdim=True) + 1e-8)
    idx1 = faces[:, 0].long()
    idx2 = faces[:, 1].long()
    idx3 = faces[:, 2].long()
    coeff1 = coeff[:,:,0:1]
    coeff2 = coeff[:,:,1:2]
    coeff3 = coeff[:,:,2:3]
    new_verts = (verts[idx1, None, :] * coeff1 +  verts[idx2, None, :] * coeff2 +  verts[idx3, None, :] * coeff3)
    new_verts = new_verts.view(-1,3)
    new_verts = torch.cat([verts, new_verts], dim=0)

    new_rgbs = (rgbs[idx1, None, :] * coeff1 +  rgbs[idx2, None, :] * coeff2 +  rgbs[idx3, None, :] * coeff3)
    new_rgbs = new_rgbs.view(-1,3)
    new_rgbs = torch.cat([rgbs, new_rgbs], dim=0)

    return new_verts, new_rgbs

class ZJUDataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_path, annot_path, pc_path, frame_num, size, use_mask, transforms, near_far_size, 
                    skip_step, random_noisy,holes,ignore_frames=[], is_train=True, subject='loot', 
                    move_cam=0, cropping=False, pc_upsampling=0):
        super(ZJUDataset, self).__init__()
        self.move_cam = move_cam if not is_train else 0
        self.frame_num = frame_num
        self.data_folder_path = data_folder_path
        self.annot_path = annot_path
        self.pc_path = pc_path
        self.use_mask = use_mask
        self.skip_step = skip_step
        self.random_noisy  =random_noisy
        self.holes = holes
        self.ignore_frames = ignore_frames
        self.is_train = is_train
        self.subject = subject
        self.height, self.width = size
        self.cropping = cropping
        self.pc_upsampling = pc_upsampling

        self.file_path = os.path.join(data_folder_path, subject)
        self.ims = sorted(os.listdir(os.path.join(self.file_path, 'Camera_B1')))
        tot_frame_num = len(self.ims)
        self.all_ims = [name[:-4]for name in self.ims]

        if self.is_train:
            self.start_idx = 0
            self.end_idx = self.frame_num
        else:
            self.start_idx = self.frame_num
            self.end_idx = tot_frame_num
        # st()
        print(f'from {self.start_idx} to {self.end_idx}')
        self.ims = self.all_ims[self.start_idx : self.end_idx]
        self.ims = self.ims[::self.skip_step]
        # self.frame_num = self.end_idx - self.start_idx
        self.frame_num = len(self.ims)
        self.vs = []
        self.vs_rgb = []
        self.vs_num = []
        self.vs_index =[]

        sum_tmp = 0
        for im in tqdm(self.ims):
            verts, faces, rgbs = load_obj_mesh(os.path.join(self.pc_path, subject, 'smpl_color', f'{self.all_ims.index(im)}.obj'))
            verts, rgbs = torch.Tensor(verts).float(), torch.Tensor(rgbs).float()
            self.vs_index.append(sum_tmp)
            if self.pc_upsampling > 0:
                verts, rgbs = pointcloud_upsampling(verts, faces, rgbs, self.pc_upsampling)
            self.vs.append(verts)
            self.vs_rgb.append(rgbs)
            self.vs_num.append(verts.shape[0])
            sum_tmp = sum_tmp + verts.shape[0]

        self.vs = torch.cat( self.vs, dim=0 )
        self.vs_rgb = torch.cat( self.vs_rgb, dim=0 )

        if random_noisy>0:
            n = tdist.Normal(torch.tensor([0.0, 0.0,0.0]), torch.tensor([random_noisy,random_noisy,random_noisy]))
            kk = torch.min((torch.max(self.vs,dim = 1)[0] - torch.min(self.vs,dim = 1)[0])/500)
            self.vs = self.vs + kk*n.sample((self.vs.size(0),))

        # camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        # anno_path = os.path.join(data_folder_path, subject, 'annots.npy')
        # camposes = np.load(anno_path, allow_pickle=True).item()['cams']
        all_views = natural_sort([f for f in os.listdir(self.file_path) if 'Camera' in f])

        # self.train_cameras = [view for view in all_views_raw[1::3] if view in all_views]
        self.train_cameras = all_views

        if self.is_train:
            self.camposes_inds = self.train_cameras
        else:
            # self.camposes_inds = sorted(list(set(all_views)-set(self.train_cameras)))
            self.camposes_inds = all_views

        self.cam_num = len(self.camposes_inds)

        param_path = os.path.join(self.file_path, 'annots.npy')
        annots = np.load(param_path, allow_pickle=True).item()
        
        Ks, Ts = [], []
        for i in self.camposes_inds:
            idx = all_views.index(i)
            K = np.array(annots['cams']['K'][idx], dtype=np.float32)
            R = np.array(annots['cams']['R'][idx])
            t = np.array(annots['cams']['T'][idx])
            Rt = np.concatenate([ \
                    np.concatenate([R, t.reshape(3,1)/1000.],1), \
                    [[0,0,0,1]]], 0).astype(np.float32)
            Ks.append(torch.from_numpy(K))
            Ts.append(torch.from_numpy(np.linalg.inv(Rt)))

        self.Ks = torch.stack(Ks, dim=0)
        self.Ts = torch.stack(Ts, dim=0)
        # self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))
        '''
        for i in range(self.Ks.size(0)):
            if self.Ks[i,0,2] > 1100:
                self.Ks[i] = self.Ks[i] * 2048.0/2448.0
                self.Ks[i] = self.Ks[i] / (2048.0/800)
            else:
                self.Ks[i] = self.Ks[i] / (2048.0/800)

        self.Ks[:,2,2] = 1
        '''

        self.transforms = transforms
        self.near_far_size = torch.Tensor(near_far_size)

        #self.black_list = [625,747,745,738,62,750,746,737,739,762]

        self._all_imgs = None
        self._all_Ts = None
        self._all_Ks = None
        self._all_width_height = None

        print('dataset initialed.')

    def __len__(self):
        return self.cam_num * self.frame_num

    def __getitem__(self, index, need_transform = True):
        frame_idx = index // self.cam_num
        cam_idx = index % self.cam_num

        frame_id = self.ims[frame_idx]
        # cam_id = int(self.cams[cam_idx][-2:])
        # cam_id = all_views.index(self.camposes_inds[cam_idx])
        cam_id = self.camposes_inds[cam_idx]
        # img = Image.open(os.path.join(self.file_path,'%d/img_%04d.jpg' % ( frame_id, cam_id)))
        img_path = os.path.join(self.file_path, cam_id)
        img_paths = [os.path.join(img_path, dir_) for dir_ in os.listdir(img_path) if frame_id in dir_]
        img = imageio.imread(img_paths[0])
        h, w = img.shape[0], img.shape[1]

        mask_dir = [dir_ for dir_ in os.listdir(self.file_path) if 'mask' in dir_][0]
        msk_path = os.path.join(self.file_path, mask_dir, cam_id)
        msk_paths = [os.path.join(msk_path, dir_) for dir_ in os.listdir(msk_path) if frame_id in dir_]
        img_mask = imageio.imread(msk_paths[0])

        if self.cropping:
            top, left, bottom, right = image_cropping(img_mask)
            img = img * (img_mask[...,None] > 0).astype(np.uint8)
            img = cv2.resize(img[top:bottom, left:right].copy(), (self.width, self.height), cv2.INTER_CUBIC)
            img_mask = cv2.resize(img_mask[top:bottom, left:right].copy(), (self.width, self.height), cv2.INTER_NEAREST)

            ROI = F.to_tensor(Image.fromarray(np.ones_like(img)))[:1]
            img = F.to_tensor(Image.fromarray(img))
            img_mask = F.to_tensor(Image.fromarray((img_mask> 0).astype(np.uint8)))
        else:
            img = Image.fromarray(img)
            img_mask = Image.fromarray(((img_mask> 0).astype(np.float32)*255).astype(np.uint8))


        T = self.Ts[cam_idx]
        K = self.Ks[cam_idx]

        if self.cropping:
            K[0,2] -= left
            K[1,2] -= top
            K[0,:] *= self.width / float(right - left)
            K[1,:] *= self.height / float(bottom - top)
        else:
            img,K,T,img_mask, ROI = self.transforms(img,K,T,img_mask)
        # if self.move_cam != 0:
        #     T = circle_path_from_smpl(self.vs[self.vs_index[frame_idx]:self.vs_index[frame_idx]+self.vs_num[frame_idx],:],
        #                             K,
        #                             img.size(),
        #                             self.move_cam
        #                             )[index % self.move_cam]
        #     st()
        img = torch.cat([img,img_mask[0:1,:,:]], dim=0)

        img = torch.cat([img,ROI], dim=0)


        return img, self.vs[self.vs_index[frame_idx]:self.vs_index[frame_idx]+self.vs_num[frame_idx],:], index, T, K, self.near_far_size, self.vs_rgb[self.vs_index[frame_idx]:self.vs_index[frame_idx]+self.vs_num[frame_idx],:]

    def get_vertex_num(self):
        return torch.Tensor(self.vs_num)






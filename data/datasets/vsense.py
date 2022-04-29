import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision
import torch.distributions as tdist

from pdb import set_trace as st
import sys
import struct
from tqdm import tqdm, trange



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


def load_ply(file_name):
	try:
		fid = open(file_name, 'r')
		head = fid.readline().strip()
		readl = lambda f: f.readline().strip()
	except UnicodeDecodeError as e:
		fid = open(file_name, 'rb')
		readl =	(lambda f: str(f.readline().strip())[2:-1]) \
			if sys.version_info[0] == 3 else \
			(lambda f: str(f.readline().strip()))
		head = readl(fid)
	form = readl(fid).split(' ')[1]
	line = readl(fid)
	vshape = fshape = [0]
	while line != 'end_header':
		s = [i for i in line.split(' ') if len(i) > 0]
		if len(s) > 2 and s[0] == 'element' and s[1] == 'vertex':
			vshape = [int(s[2])]
			line = readl(fid)
			s = [i for i in line.split(' ') if len(i) > 0]
			while s[0] == 'property' or s[0][0] == '#':
				if s[0][0] != '#':
					vshape += [s[1]]
				line = readl(fid)
				s = [i for i in line.split(' ') if len(i) > 0]
		elif len(s) > 2 and s[0] == 'element' and s[1] == 'face':
			fshape = [int(s[2])]
			line = readl(fid)
			s = [i for i in line.split(' ') if len(i) > 0]
			while s[0] == 'property' or s[0][0] == '#':
				if s[0][0] != '#':
					fshape = [fshape[0],s[2],s[3]]
				line = readl(fid)
				s = [i for i in line.split(' ') if len(i) > 0]
		else:
			line = readl(fid)
	if form.lower() == 'ascii':
		v = []
		for i in range(vshape[0]):
			s = [i for i in readl(fid).split(' ') if len(i) > 0]
			if s[0][0] != '#':
				v += [[float(i) for i in s]]
		v = np.array(v, np.float32)
		tri = []
		for i in range(fshape[0]):
			s = [i for i in readl(fid).split(' ') if len(i) > 0]
			if s[0][0] != '#':
				tri += [[int(s[1]),int(s[i-1]),int(s[i])] \
					for i in range(3,len(i))]
		tri = np.array(tri, np.int64)
	else:
		maps = {'float':('f',4), 'double':('d',8), \
			'uint': ('I',4), 'int':   ('i',4), \
			'ushot':('H',2), 'short': ('h',2), \
			'uchar':('B',1), 'char':  ('b',1)}
		if 'little' in form.lower():
			fmt = '<' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
		else:
			fmt = '>' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
		l = sum([maps[i][1] for i in vshape[1:]]) * vshape[0]
		v = struct.unpack(fmt, fid.read(l))
		v = np.array(v).reshape(vshape[0],-1).astype(np.float32)
		tri = []
		for i in range(fshape[0]):
			l = struct.unpack(fmt[0]+maps[fshape[1]][0], fid.read(maps[fshape[1]][1]))
			l = l[0]
			f = struct.unpack(fmt[0]+maps[fshape[2]][0]*l, \
				fid.read(l*maps[fshape[2]][1]))
			tri += [[f[0],f[i-1],f[i]] for i in range(2,len(f))]
		tri = np.array(tri).reshape(fshape[0],-1).astype(np.int64)
	fid.close()
	return v, tri

def merge_holes(pc1,pc2):

    # change point color here

    return np.concatenate([pc1, pc2], axis=0)


class VsenseDataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_path, frame_num, use_mask, transforms, near_far_size, skip_step, random_noisy,holes,ignore_frames=[], is_train=True, subject='loot', move_cam=0):
        super(VsenseDataset, self).__init__()
        self.move_cam = move_cam if not is_train else 0
        self.frame_num = frame_num
        self.data_folder_path = data_folder_path
        self.use_mask = use_mask
        self.skip_step = skip_step
        self.random_noisy  =random_noisy
        self.holes = holes
        self.ignore_frames = ignore_frames
        self.is_train = is_train
        self.subject = subject

        self.file_path = os.path.join(data_folder_path, subject, 'RENDER')
        tot_frame_num = len(os.listdir(os.path.join(self.file_path, 'cam_01')))
        self.ims = sorted(os.listdir(os.path.join(self.file_path, 'cam_01')))
        self.ims = [name[:-4]for name in self.ims]

        self.cams = sorted(os.listdir(os.path.join(data_folder_path, subject, 'RENDER')))
        if self.is_train:
            self.start_idx = 1
            self.end_idx = self.frame_num
        else:
            self.start_idx = self.frame_num
            self.end_idx = tot_frame_num
        # st()
        print(f'from {self.start_idx} to {self.end_idx}')
        self.ims = self.ims[self.start_idx : self.end_idx]
        self.frame_num = self.end_idx - self.start_idx
        self.vs = []
        self.vs_rgb = []
        self.vs_num = []
        self.vs_index =[]

        sum_tmp = 0
        for i in trange(self.start_idx, self.end_idx):
            #tmp = np.loadtxt(os.path.join(data_folder_path,'pointclouds/frame%d.obj' % (i+1)), usecols = (1,2,3,4,5,6))
            # tmp = np.load(os.path.join(data_folder_path,'pointclouds/frame%d.npy' % (i+1)))
            tmp, _ = load_ply(os.path.join(data_folder_path, subject, 'GEO/OBJ', f'{i:06d}.ply'))
            num_verts = tmp.shape[0]
            random_inds = np.random.randint(num_verts, size=int(0.01 * num_verts))
            tmp = tmp[random_inds]
            vs_tmp = tmp[:,0:3] 
            vs_rgb_tmp = tmp[:,3:6]
            self.vs_index.append(sum_tmp)
            self.vs.append(torch.Tensor(vs_tmp))
            self.vs_rgb.append(torch.Tensor(vs_rgb_tmp))
            self.vs_num.append(vs_tmp.shape[0])
            sum_tmp = sum_tmp + vs_tmp.shape[0]
            
            if i%50 == 0:
                print(i,'/',self.frame_num)


        self.vs = torch.cat( self.vs, dim=0 )
        self.vs_rgb = torch.cat( self.vs_rgb, dim=0 )

        if random_noisy>0:
            n = tdist.Normal(torch.tensor([0.0, 0.0,0.0]), torch.tensor([random_noisy,random_noisy,random_noisy]))
            kk = torch.min((torch.max(self.vs,dim = 1)[0] - torch.min(self.vs,dim = 1)[0])/500)
            self.vs = self.vs + kk*n.sample((self.vs.size(0),))
        
        

        # camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        # anno_path = os.path.join(data_folder_path, subject, 'annots.npy')
        # camposes = np.load(anno_path, allow_pickle=True).item()['cams']
        self.all_cameras = list(range(36))
        if self.move_cam != 0:
            self.camposes_inds = list(range(0 ,36))
        elif self.is_train:
            self.camposes_inds = list(set(list(range(0, 36, 2))))
        else:
            self.camposes_inds = list(set(list(range(1, 36, 2))))

        # self.cams = [self.cams[idx] for idx in camposes_inds]
        # self.Ks = []
        # self.Ts = []
        # for cam_idx in camposes_inds:
        #     K = camposes['K'][cam_idx]
        #     R = camposes['R'][cam_idx]
        #     T = camposes['T'][cam_idx]
        #     self.Ks.append(np.float32(K))
        #     Rt = np.concatenate([R, T[:, None]], axis=1)
        #     w2c = np.concatenate([Rt, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
        #     c2w = np.linalg.inv(w2c)
        #     self.Ts.append(np.float32(c2w))

        # # st()
        # self.Ks = np.array(self.Ks)
        # self.Ts = np.array(self.Ts)
        # # st()
        # self.Ks = torch.from_numpy(self.Ks)
        # self.Ts = torch.from_numpy(self.Ts)

        # self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        # self.cam_num = self.Ts.size(0)
        self.cam_num = len(self.camposes_inds)
        
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
        if self.move_cam == 0:
            return self.cam_num *  (self.frame_num//self.skip_step) 
        else:
            return 1 * (self.frame_num//self.skip_step)

    def __getitem__(self, index, need_transform = True):
        if self.move_cam == 0:
            frame_idx = ((index // self.cam_num) * self.skip_step) %self.frame_num
        else:
            frame_idx = index % self.frame_num

        if self.move_cam == 0:
            cam_idx = index % self.cam_num
        else:
            cam_idx = int((1.0 - float(index) / float(self.frame_num)) * (len(self.cams) - 1))

        frame_id = int(self.ims[frame_idx])
        # cam_id = int(self.cams[cam_idx][-2:])
        cam_id = self.all_cameras.index(self.camposes_inds[cam_idx])
        # img = Image.open(os.path.join(self.file_path,'%d/img_%04d.jpg' % ( frame_id, cam_id)))
        img = Image.open(os.path.join(self.file_path, f'cam_{cam_id:02d}', f'{frame_id:06d}.png'))
        #if need_transform:
        #    if img.size[0]>2100:
        #        img = torchvision.transforms.functional.crop(img, 0, 0, 1836, 2448)

            #img = self.transforms(img)

        # K = self.Ks[cam_idx]

        # if self.use_mask:
        img_mask = Image.open(os.path.join(self.file_path.replace('RENDER', 'MASK'), f'cam_{cam_id:02d}', f'{frame_id:06d}.png'))
        #if need_transform:
        #    if img_mask.size[0]>2100:
        #        img_mask = torchvision.transforms.functional.crop(img_mask, 0, 0, 1836, 2448)

            #img_mask = self.transforms(img_mask)
        #print(img.size(),img_mask.size())
        #img = torch.cat([img,img_mask[0:1,:,:]], dim=0)
        param_path = os.path.join(self.file_path.replace('RENDER', 'PARAM'), f'cam_{cam_id:02d}', f'{frame_id:06d}.png.npy')
        param = np.load(param_path, allow_pickle=True).item()
        intr_mat = param.get('intr')
        norm_mat = param.get('norm')
        view_mat = param.get('view')
        rot_mat = param.get('rot')
        gl2cv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

        w2c = gl2cv @ view_mat @ rot_mat @ norm_mat
        T = torch.from_numpy(np.linalg.inv(w2c)).float()
        K = torch.from_numpy(intr_mat).float()

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






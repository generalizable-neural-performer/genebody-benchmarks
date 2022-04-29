from tqdm import tqdm
import numpy as np
import argparse
import struct
import sys
import cv2, imageio
import os
import re
from multiprocessing import Queue, Lock, Process
from util import natural_sort

base_dir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--datatype', type = str, default = 'ghr')
parser.add_argument('--datadir', type = str, default = '/data/ZJU_Mocap/CoreView_313')
parser.add_argument('--outdir', type = str, default = os.path.join(base_dir,'..','..','data'))
parser.add_argument('--annotdir', type = str, default = os.path.join(base_dir,'..','..','data'))
parser.add_argument('--smpl', type = str, default = '/data/smpl/model/smpl/SMPL_NEUTRAL.pkl')
parser.add_argument('--smpl_uv', type = str, default = './smpl_t_pose/smplx.obj')
parser.add_argument('--workers', type = int, default = 8)

def load_obj_mesh(mesh_file, with_normal=False, with_texture=False, with_texture_image=False):
	vertex_data = []
	norm_data = []
	uv_data = []

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

	if with_texture and with_normal:
		uvs = np.array(uv_data)
		face_uvs = np.array(face_uv_data) - 1
		norms = np.array(norm_data)
		if norms.shape[0] == 0:
			norms = compute_normal(vertices, faces)
			face_normals = faces
		else:
			norms = normalize_v3(norms)
			face_normals = np.array(face_norm_data) - 1
		if with_texture_image:
			return vertices, faces, norms, face_normals, uvs, face_uvs, texture_image
		else:
			return vertices, faces, norms, face_normals, uvs, face_uvs

	if with_texture:
		uvs = np.array(uv_data)
		face_uvs = np.array(face_uv_data) - 1
		return vertices, faces, uvs, face_uvs

	if with_normal:
		# norms = np.array(norm_data)
		# norms = normalize_v3(norms)
		# face_normals = np.array(face_norm_data) - 1
		norms = np.array(norm_data)
		if norms.shape[0] == 0:
			norms = compute_normal(vertices, faces)
			face_normals = faces
		else:
			norms = normalize_v3(norms)
			face_normals = np.array(face_norm_data) - 1
		return vertices, faces, norms, face_normals

	return vertices, faces



def load_ply(file_name):
	v = []; tri = []
	try:
		fid = open(file_name, 'r')
		head = fid.readline().strip()
		readl= lambda f: f.readline().strip()
	except UnicodeDecodeError as e:
		fid = open(file_name, 'rb')
		readl =	(lambda f: str(f.readline().strip())[2:-1]) \
			if sys.version_info[0] == 3 else \
			(lambda f: str(f.readline().strip()))
		head = readl(fid)
	if head.lower() != 'ply':
		return	v, tri
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
		for i in range(vshape[0]):
			s = [i for i in readl(fid).split(' ') if len(i) > 0]
			if len(s) > 0 and s[0][0] != '#':
				v += [[float(i) for i in s]]
		v = np.array(v, np.float32)
		for i in range(fshape[0]):
			s = [i for i in readl(fid).split(' ') if len(i) > 0]
			if len(s) > 0 and s[0][0] != '#':
				tri += [[int(s[1]),int(s[i-1]),int(s[i])] \
					for i in range(3,len(s))]
		tri = np.array(tri, np.int64)
	else:
		maps = {'float': ('f',4), 'double':('d',8), \
			'uint':  ('I',4), 'int':   ('i',4), \
			'ushort':('H',2), 'short': ('h',2), \
			'uchar': ('B',1), 'char':  ('b',1)}
		if 'little' in form.lower():
			fmt = '<' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
		else:
			fmt = '>' + ''.join([maps[i][0] for i in vshape[1:]]*vshape[0])
		l = sum([maps[i][1] for i in vshape[1:]]) * vshape[0]
		v = struct.unpack(fmt, fid.read(l))
		v = np.array(v).reshape(vshape[0],-1).astype(np.float32)
		v = v[:,:3]
		tri = []
		for i in range(fshape[0]):
			l = struct.unpack(fmt[0]+maps[fshape[1]][0], \
				fid.read(maps[fshape[1]][1]))[0]
			f = struct.unpack(fmt[0]+maps[fshape[2]][0]*l, \
				fid.read(l*maps[fshape[2]][1]))
			tri += [[f[0],f[i-1],f[i]] for i in range(2,len(f))]
		tri = np.array(tri).reshape(fshape[0],-1).astype(np.int64)
	fid.close()
	return	v, tri
'''import torch
class Rodrigues(torch.autograd.Function):
	@staticmethod
	def forward(ctx, rvec, eps = 1e-6):
		ctx.eps = min(abs(eps), 1)
		sz = [int(i) for i in rvec.shape]
		x,y,z =	rvec.view(-1,sz[-1])[:,:1], \
			rvec.view(-1,sz[-1])[:,1:2],\
			rvec.view(-1,sz[-1])[:,2:3]
		x2, y2, z2 = x*x, y*y, z*z
		xy, yz, zx = x*y, y*z, z*x
		r2 = x2 + y2 + z2
		r = torch.sqrt(r2)
		c = torch.cos(r)
		s = torch.sin(r)
		sc = torch.where(r <= ctx.eps, 1 - r2/6, s/r)
		cc = torch.where(r <= ctx.eps,.5 - r2/24,(1-c)/r2)
		ctx.save_for_backward(c, sc, cc, r, rvec)
		return	torch.cat([cc*x2+c, cc*xy-sc*z, cc*zx+sc*y, \
				cc*xy+sc*z, cc*y2 + c,  cc*yz-sc*x, \
				cc*zx-sc*y, cc*yz+sc*x, cc*z2 + c], 1) \
				.view(sz[:-1] + [3,3])
	@staticmethod
	def backward(ctx, dR):
		dr = None
		if ctx.needs_input_grad[0]:
			c, sc, cc, r, rvec = ctx.saved_tensors
			sz = [int(i) for i in rvec.shape]
			x2, y2, z2 = x*x, y*y, z*z
			xy, yz, zx = x*y, y*z, z*x
			r2 = x2 + y2 + z2
			dR = dR.view(-1,3,3)
			dcc = torch.where(r <= ctx.eps,-1./12+ r2/180,(sc-2*cc)/r2)
			dsc = torch.where(r <= ctx.eps,-1./3 + r2/30, (c -  sc)/r2)
			dRdr = torch.cat([dcc*x2-sc,  dcc*xy-dsc*z, dcc*zx+dsc*y, \
					dcc*xy+dsc*z, dcc*y2 - sc,  dcc*yz-dsc*x, \
					dcc*zx-dsc*y, dcc*yz+dsc*x, dcc*z2 - sc], 1)
			rc = rvec.view(-1,sz[-1])[:,:3] * cc
			dr = (dR * dRdr).sum([1,2]).unsqueeze(1)
			dr = torch.cat([(rc *(dR[:,0,:] + dR[:,:,0])).sum(1,keepdim=True) + \
					dr * x + sc *(dR[:,2,1:2] - dR[:,1,2:3]), \
					(rc *(dR[:,1,:] + dR[:,:,1])).sum(1,keepdim=True) + \
					dr * y + sc *(dR[:,0,2:3] - dR[:,2,0:1]), \
					(rc *(dR[:,2,:] + dR[:,:,2])).sum(1,keepdim=True) + \
					dr * z + sc *(dR[:,1,0:1] - dR[:,0,1:2])], 1)
			dr = torch.cat([dr,torch.zeros([len(dr),sz[-1]-3], \
				dtype = dr.dtype, device = dr.device)], 1).view(sz)
		return dr if len(ctx.needs_input_grad) == 1 else (dr, None)
class SMPL(torch.nn.Module):
	def __init__(self, pkl_name, dtype = torch.float32, device = 'cpu'):
		super(SMPL, self).__init__()
		import pickle
		import scipy
		with open(pkl_name, 'rb') as f:
			data = pickle.load(f, encoding = 'latin1')
		for k in [	'v_template',  \
				'J_regressor', \
				'shapedirs', \
				'posedirs', \
				'weights', \
				'f']:
			if isinstance(data[k], scipy.sparse.csc.csc_matrix):
				data[k] = data[k].toarray()
			if data[k].dtype in [np.float16,np.float32,np.float64,np.float128]:
				self.register_buffer(k, \
					torch.from_numpy(data[k]) \
					.type(dtype).to(device))
			else:
				self.register_buffer(k, \
					torch.from_numpy(data[k].astype(np.int64)) \
					.to(device))
		self.tree = data['kintree_table'].T.astype(np.int64)
	def forward(self, pose = None, beta = None):
		batch =(max(len(pose),len(beta)) if beta is not None else \
			len(pose))if pose is not None else ( \
			len(beta) if beta is not None else 1)
		if beta is not None:
			v_shaped = torch.matmul(beta, \
				self.shapedirs.view(-1,self.shapedirs.shape[-1]).t()) \
				.view(len(beta),*self.shapedirs.shape[:2]) + \
				self.v_template.unsqueeze(0)
		else:
			v_shaped = self.v_template.unsqueeze(0).expand(batch,-1,-1)
		joints = torch.matmul(self.J_regressor.unsqueeze(0), v_shaped)
		R = [torch.eye(3,dtype=joints.dtype,device=joints.device).unsqueeze(0)]
		if pose is not None:
			pose = pose.view(len(pose),-1,3)
			R += [Rodrigues.apply(pose[:,i]) for i in range(pose.shape[1])]
			p = torch.cat([(r - R[0]).view(-1,9) for r in R[1:]],-1)
			v_posed = torch.matmul(p, \
				self.posedirs.view(-1,self.posedirs.shape[-1]).t()) \
				.view(-1, *self.posedirs.shape[:2]) + \
				v_shaped
		else:
			R *= int(joints.shape[1])
			v_posed = v_shaped
		T = [torch.cat([R[0],joints[:,0,:].unsqueeze(-1)],-1)]
		for p, i in self.tree[1:]:
			T += [torch.cat([ \
				torch.matmul(T[p][:,:3,:3], R[i]), \
				torch.matmul(T[p][:,:3,:3], \
					(joints[:,i]-joints[:,p]).unsqueeze(-1)) + \
					T[p][:,:3,3:]], -1)]
		v = 0
		for i in range(len(T)):
			v = v + self.weights[np.newaxis,:,i:i+1] * torch.matmul( \
				v_posed,T[i][:,:3,:3].permute(0,2,1)) + \
				T[i][:,:3,3:4].permute(0,2,1)
		return	v'''
def load_smpl(file_name, smpl_model):
	param = np.reshape(np.load(file_name, allow_pickle = True),-1)[0]
	pose = param['poses'].reshape(-1)
	beta = param['shapes'].reshape(-1)
	T = np.concatenate([ \
		cv2.Rodrigues(param['Rh'].reshape(-1))[0], \
		np.reshape(param['Th'], [3,1])], 1)
	pose = torch.from_numpy(pose).type(torch.float32).unsqueeze(0)
	beta = torch.from_numpy(beta).type(torch.float32).unsqueeze(0)
	if pose.shape[1] == 3 * smpl_model.J_regressor.shape[0]:
		pose = pose[:,3:]
	v = smpl_model(pose, beta)
	v = v[0].detach().cpu().numpy()
	v = v.dot(T[:3,:3].T) + T[:3,3:4].T
	tri = smpl_model.f.detach().cpu().numpy()
	return	v, tri
def distortPoints(p, dist):
	dist = np.reshape(dist,-1) \
		if dist is not None else []
	k1 = dist[0] if len(dist) > 0 else 0
	k2 = dist[1] if len(dist) > 1 else 0
	p1 = dist[2] if len(dist) > 2 else 0
	p2 = dist[3] if len(dist) > 3 else 0
	k3 = dist[4] if len(dist) > 4 else 0
	k4 = dist[5] if len(dist) > 5 else 0
	k5 = dist[6] if len(dist) > 6 else 0
	k6 = dist[7] if len(dist) > 7 else 0
	x, y = p[...,0], p[...,1]
	x2 = x * x; y2 = y * y; xy = x * y
	r2 = x2 + x2
	c =	(1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / \
		(1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
	x_ = c*x + p1*2*xy + p2*(r2+2*x2)
	y_ = c*y + p2*2*xy + p1*(r2+2*y2)
	p[...,0] = x_
	p[...,1] = y_
	return p
def check_proj(v, K, img, w2c = np.identity(4), dist = None):
	v_rot = v.dot(w2c[:3,:3].T) + w2c[:3,3:].T
	v_proj= v_rot[:,:2] / v_rot[:,2:3]
	if dist is not None:
		v_proj = distortPoints(v_proj, dist)
	v_proj= v_proj[:,:2].dot(K[:2,:2].T) + K[:2,2:3].T
	for p in v_proj:
		x = int(p[0])
		y = int(p[1])
		img = cv2.circle(img, (x,y), 1,(0,255,0), -1)
	return	img


def rasterize(v, tri, uv, uv_tri, size, K = np.identity(3), \
		dist = None, persp = True, eps = 1e-6, use_white_bkgd=False):
	h, w = size
	zbuf = np.ones([h, w], v.dtype) * float('inf')
	img = np.ones([h, w, uv.shape[-1]], uv.dtype) * -1
	if dist is not None:
		valid = np.where(v[:,2] >= eps)[0] \
			if persp else np.arange(len(v))
		v_proj = v[valid,:2] / v[valid,2:]
		v_proj = distortPoints(v_proj, dist)
		v[valid,:2]= v_proj * v[valid,2:]
	v_proj = v.dot(K.T)[:,:2] / np.maximum(v[:,2:], eps) \
		if persp else v.dot(K.T)[:,:2]
	va = v_proj[tri[:,0],:2]
	vb = v_proj[tri[:,1],:2]
	vc = v_proj[tri[:,2],:2]
	front = np.cross(vc - va, vb - va)
	umin = np.maximum(np.ceil (np.vstack((va[:,0],vb[:,0],vc[:,0])).min(0)), 0)
	umax = np.minimum(np.floor(np.vstack((va[:,0],vb[:,0],vc[:,0])).max(0)),w-1)
	vmin = np.maximum(np.ceil (np.vstack((va[:,1],vb[:,1],vc[:,1])).min(0)), 0)
	vmax = np.minimum(np.floor(np.vstack((va[:,1],vb[:,1],vc[:,1])).max(0)),h-1)
	umin = umin.astype(np.int32)
	umax = umax.astype(np.int32)
	vmin = vmin.astype(np.int32)
	vmax = vmax.astype(np.int32)
	front = np.where(np.logical_and(np.logical_and( \
			umin <= umax, vmin <= vmax), front > 0))[0]
	for t in front:
		A = np.concatenate((vb[t:t+1]-va[t:t+1], vc[t:t+1]-va[t:t+1]),0)
		x, y = np.meshgrid(	range(umin[t],umax[t]+1), \
					range(vmin[t],vmax[t]+1))
		u = np.vstack((x.reshape(-1),y.reshape(-1))).T
		coeff = (u.astype(v.dtype) - va[t:t+1,:]).dot(np.linalg.pinv(A))
		coeff = np.concatenate((1-coeff.sum(1).reshape(-1,1),coeff),1)
		if persp:
			z = coeff.dot(v[tri[t], 2])
		else:
			z = 1 / np.maximum((coeff/v[tri[t],2:3].T).sum(1), eps)
		c_ = coeff.dot(uv[uv_tri[t], :])
		for i, (x, y) in enumerate(u):
			if  coeff[i,0] >= -eps \
			and coeff[i,1] >= -eps \
			and coeff[i,2] >= -eps \
			and zbuf[y,x] > z[i]:
				zbuf[y,x] = z[i]
				img[y, x, :] = c_[i, :]
	return	img, zbuf


def render_view(intri, dists, c2ws, plys, uv, uv_face, params, view, i):

	out = os.path.join(args.outdir, os.path.basename(view))
	if not os.path.isdir(out):
		os.makedirs(out, exist_ok=True)
	else:
		completed = len(os.listdir(out))
		# a completed folder
		if completed > 0 and completed % 150 == 0:
			return
	imgs = [os.path.join(view, f) for f in os.listdir(view) \
		if f[-4:] in ['.jpg','.png']]
	imgs = sorted(imgs) if len(imgs) > 1 else imgs
	for k in tqdm(range(len(imgs))):
		img = cv2.imread(imgs[k])
		try:
			if i < len(plys) and plys[k][-4:] == '.ply':
				v, tri = load_ply(plys[k])
			elif i < len(plys) and plys[k][-4:] == '.npy':
				v = np.load(plys[k])
			elif i < len(plys) and plys[k][-4:] == '.obj':
				v, tri = load_obj_mesh(plys[k])
			elif i < len(params):
				v, tri = load_smpl(params[k], smpl_model)
			else:
				continue
		except:
			continue
		if len(intri.shape) == 3:
			K = intri[k] 
			dist = dists[k] if dists is not None else None
			c2w  = np.concatenate([c2ws[k],[[0,0,0,1]]], 0)
		else:
			K = intri
			dist = dists if dists is not None else None
			c2w = np.concatenate([c2ws,[[0,0,0,1]]], 0)
		w2c = np.linalg.inv(c2w)
		v_= v.dot(w2c[:3,:3].T) + w2c[:3,3:].T
		img, z = rasterize(v_, tri, uv, uv_face, img.shape[:2], K, dist)
		z[z == float('inf')] = 0
		img = np.concatenate([img, z[...,None]], axis=-1).astype(np.float32)
		imageio.imwrite(os.path.join(out, \
				os.path.basename(imgs[k])[:-4]+'.exr'), img)
		# np.save(os.path.join(out, \
		# 		os.path.basename(imgs[k])[:-4]+'.npy'), img)

class Worker(Process):

	def __init__(self, queue, lock):
		super(Worker, self).__init__()
		self.queue = queue
		self.lock = lock

	def run(self):
		while True:
			self.lock.acquire()
			if self.queue.empty():
				self.lock.release()
				break
			else:
				kwargs = self.queue.get()
				queue_len = self.queue.qsize()
				self.lock.release()
				print("started {}, {} jobs left".format(kwargs["view"], queue_len))
				render_view(**kwargs)


def render_ghr(args):
	views = [os.path.join(args.datadir,'image', f) \
		for f in os.listdir(os.path.join(args.datadir, 'image'))]
	annot = np.load(os.path.join(args.annotdir), allow_pickle = True)
	annot = np.reshape(annot,-1)[0]
	intri = np.array(annot['cams']['K'], np.float32)
	dists = np.array(annot['cams']['D'], intri.dtype)
	c2ws  = np.concatenate([ \
			annot['cams']['R'], \
			annot['cams']['T'][:,:,None]],-1).astype(intri.dtype)
	if args.outdir == '': args.outdir = '.'
	if not os.path.isdir(args.outdir):
		os.mkdir(args.outdir)
	if os.path.exists(os.path.join(args.datadir,'new_smpl')):
		plys = [os.path.join(args.datadir,'new_smpl',f) \
			for f in os.listdir(os.path.join(args.datadir,'new_smpl')) \
			if f[-4:] == '.ply']
		plys = natural_sort(plys)
		params = []
	elif os.path.exists(os.path.join(args.datadir,'smpl')):
		plys = [os.path.join(args.datadir,'smpl',f) \
			for f in os.listdir(os.path.join(args.datadir,'smpl')) \
			if f[-4:] == '.ply']
		plys = natural_sort(plys)
		params = []
	elif os.path.exists(os.path.join(args.datadir,'new_vertices')):
		plys = [os.path.join(args.datadir,'new_vertices',f) \
			for f in os.listdir(os.path.join(args.datadir,'new_vertices')) \
			if f[-4:] == '.npy']
		tri = np.loadtxt(os.path.join(base_dir,'tri.txt')).astype(np.int64)
		plys = natural_sort(plys)
		params = []
	elif os.path.exists(os.path.join(args.datadir,'vertices')):
		plys = [os.path.join(args.datadir,'vertices',f) \
			for f in os.listdir(os.path.join(args.datadir,'vertices')) \
			if f[-4:] == '.npy']
		_, tri = load_obj_mesh('./smpl_t_pose/smplx.obj')
		tri = tri.astype(np.int64)
		plys = natural_sort(plys)
		params = [] 
	else:
		plys  = []
		smpl_model = SMPL(args.smpl)
		params= [os.path.join(args.datadir,'params',f) \
			for f in os.listdir(os.path.join(args.datadir,'params')) \
			if f[-4:] == '.npy']
		params = natural_sort(params)

	_, _, uv, uv_face = load_obj_mesh(args.smpl_uv, with_texture=True)

	queue = Queue()
	lock = Lock()

	for i, view in enumerate(natural_sort(views)):

		queue.put({
			'intri': intri[i], 
			'dists': dists[i], 
			'c2ws': c2ws[i], 
			'plys': plys, 
			'params': params, 
			'view': view, 
			'i': i,
			'uv': uv, 
			'uv_face': uv_face
		})

	import time
	pool = [Worker(queue, lock) for _ in range(args.workers)]
	for worker in pool: 
		worker.start()
		time.sleep(0.1)
	for worker in pool: 
		worker.join()
		time.sleep(0.1)

def read_vsense_param(param_path):
	# loading calibration data
	param = np.load(param_path, allow_pickle=True).item()
	intr_mat = param.get('intr')
	norm_mat = param.get('norm')
	view_mat = param.get('view')
	rot_mat = param.get('rot')
	gl2cv = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
	w2c = gl2cv @ view_mat @ rot_mat @ norm_mat
	c2w = np.linalg.inv(w2c)
	return c2w, intr_mat

def render_vsense(args):
	views = [os.path.join(args.datadir,'RENDER', f) \
		for f in os.listdir(os.path.join(args.datadir, 'RENDER'))]
	# annot = np.load(os.path.join(args.annotdir), allow_pickle = True)
	# annot = np.reshape(annot,-1)[0]
	# intri = np.array(annot['cams']['K'], np.float32)
	# dists = np.array(annot['cams']['D'], intri.dtype)
	# c2ws  = np.concatenate([ \
	# 		annot['cams']['R'], \
	# 		annot['cams']['T'][:,:,None]],-1).astype(intri.dtype)
	views = natural_sort(views)
	c2ws, intri = [], []
	for view in views:
		view = view.replace('RENDER', 'PARAM')
		for frame in natural_sort(os.listdir(view)):
			c2w, K = read_vsense_param(os.path.join(view, frame))
			c2ws.append(c2w.copy()[:3])
			intri.append(K.copy())
	c2ws = np.stack(c2ws).reshape(len(views),-1,3,4).astype(np.float32)
	intri = np.stack(intri).reshape(len(views),-1,3,3).astype(np.float32)
	if args.outdir == '': args.outdir = '.'
	if not os.path.isdir(args.outdir):
		os.mkdir(args.outdir)

	plys = [os.path.join(args.datadir,'SMPL',f) \
		for f in os.listdir(os.path.join(args.datadir,'SMPL')) \
		if f[-4:] == '.obj']
	plys = natural_sort(plys)
	params = []


	_, _, uv, uv_face = load_obj_mesh(args.smpl_uv, with_texture=True)

	queue = Queue()
	lock = Lock()

	for i, view in enumerate(views):

		queue.put({
			'intri': intri[i], 
			'dists': None, 
			'c2ws': c2ws[i], 
			'plys': plys, 
			'params': params, 
			'view': view, 
			'i': i,
			'uv': uv, 
			'uv_face': uv_face
		})

	import time
	pool = [Worker(queue, lock) for _ in range(args.workers)]
	for worker in pool: 
		worker.start()
		time.sleep(0.1)
	for worker in pool: 
		worker.join()
		time.sleep(0.1)


if __name__ == '__main__':
	args  = parser.parse_args()
	if args.datatype == 'ghr':
		render_ghr(args)
	elif args.datatype == 'vsense':
		render_vsense(args)
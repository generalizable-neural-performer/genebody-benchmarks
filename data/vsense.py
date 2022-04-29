# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import os, sys
from PIL import Image
import imageio, cv2
from .utils import load_ply
from .mesh_utils import load_obj_mesh
import torch.utils.data

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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir, subject, keyfilter,
            fixedcammean=0., fixedcamstd=1., imagemean=0., 
            imagestd=1., subsampletype=None, subsamplesize=0,
            loadSize=512):
        # krtpath = "/data/experiments/dryice1/data/KRT"
        # krt = load_krt(krtpath)
        # get options
        self.subject = subject
        self.datadir = datadir
        self.all_cameras = self.get_allcameras()
        self.cameras = range(36)
        self.cameras = [cam for cam in self.cameras if cam in self.all_cameras]
        self.fixedcameras = [0, 4, 9, 13, 18, 22, 27, 31]
        self.fixedcameras = [cam for cam in self.fixedcameras if cam in self.all_cameras]
        self.loadSize = loadSize
        self.annots = np.load(os.path.join(self.datadir, self.subject, 'annots.npy'), allow_pickle=True).item()['cams']
        self.ninput = len(self.fixedcameras)
        self.is_train = subsampletype != None
        if self.is_train:
            print('Constructing training dataset.')
        else:
            print('Constructing testing dataset.')
        self.framelist = sorted(os.listdir(os.path.join(self.datadir, self.subject, 'SMPL')))
        self.framelist = self.framelist[:len(self.framelist)//2] if self.is_train else self.framelist[len(self.framelist)//2::30]
        self.framecamlist = [(x[:-4], cam)
        for x in self.framelist
        for cam in (self.cameras if len(self.cameras) > 0 else [None])]
        self.keyfilter = keyfilter
        self.fixedcammean = fixedcammean
        self.fixedcamstd = fixedcamstd
        self.imagemean = imagemean
        self.imagestd = imagestd
        self.subsampletype = subsampletype
        self.subsamplesize = subsamplesize
        self.cropping = 'cropping' in self.keyfilter
        self.worldscale = 1. / 600.
        # immitate dryice dataloader which is basically a cv to gl transformation
        self.transf = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]], dtype=np.float32)
        print(f'dataset size {len(self.framecamlist)}')

    def get_allcameras(self):

        all_cameras_raw = list(range(36))
        
        return all_cameras_raw

    def load_image(self, frame, cam):
        imagedir = os.path.join(self.datadir, self.subject, 'RENDER', 'cam_%02d' % cam)
        imagepaths = sorted([os.path.join(imagedir, dir_) for dir_ in os.listdir(imagedir) if frame in dir_])
        image = imageio.imread(imagepaths[0])
        maskdir = os.path.join(self.datadir, self.subject, 'MASK', 'cam_%02d' % cam)
        maskpaths = sorted([os.path.join(maskdir, dir_) for dir_ in os.listdir(maskdir) if frame in dir_])
        mask = imageio.imread(maskpaths[0])
        mask = (mask > 127).astype(np.uint8)
        image = image * mask[...,None]

        image = cv2.resize(image.copy(), (self.loadSize, self.loadSize), cv2.INTER_CUBIC)
        image = image.transpose((2,0,1)).astype(np.float32)

        idx = self.all_cameras.index(cam)
        K = self.annots['K'][idx].astype(np.float32).copy()
        R = self.annots['R'][idx].astype(np.float32).copy()
        T = self.annots['T'][idx].astype(np.float32).copy()
        RT = np.concatenate([R, T.reshape(3, 1)], axis=1)
        w2c = np.concatenate([RT, np.array([0, 0, 0, 1]).reshape(1, 4).astype(np.float32)], axis=0)
        c2w = np.linalg.inv(w2c)

        return image, K, c2w

    def load_smpl(self, frame):
        from pdb import set_trace as st
        smpldir = os.path.join(self.datadir, self.subject, 'SMPL')
        smplpaths = sorted([os.path.join(smpldir, dir_) for dir_ in os.listdir(smpldir) if frame in dir_])
        verts, faces= load_obj_mesh(smplpaths[0])
        verts = verts.astype(np.float32)
        verts_min, verts_max = verts.min(0), verts.max(0)
        center = (verts_min + verts_max) / 2
        size = (verts_max - verts_min).max()

        paramdir = os.path.join(self.datadir, self.subject, 'SMPL_PARAM')
        parampaths = sorted([os.path.join(paramdir, dir_) for dir_ in os.listdir(paramdir) if frame in dir_])
        param = np.load(parampaths[0], allow_pickle=True).item()
        pose = param['pose'].reshape(-1, 3)
        scale = param['body_scale']
        smplrot = cv2.Rodrigues(pose[:1])[0]

        return center, scale, smplrot


    def __len__(self):
        return len(self.framecamlist)

    def __getitem__(self, idx):
        frame, cam = self.framecamlist[idx]
        result = {}

        validinput = True

        # fixed camera images
        if "fixedcamimage" in self.keyfilter:
            fixedcamimage = []
            for i in range(self.ninput):
                img, K, Rt = self.load_image(frame, self.fixedcameras[i])
                fixedcamimage.append(img)
            fixedcamimage = np.concatenate(fixedcamimage, axis=0)
            fixedcamimage[:] -= self.imagemean
            fixedcamimage[:] /= self.imagestd
            result["fixedcamimage"] = fixedcamimage

        result["validinput"] = np.float32(1.0 if validinput else 0.0)
        # image data
        if cam is not None:
            if "camera" in self.keyfilter or "image" in self.keyfilter:
                image, K, Rt = self.load_image(frame, cam)
                center, scale, smplrot = self.load_smpl(frame)
                # camera data
                # w2c @ trasf * worldscale
                result["camrot"] = (np.linalg.inv(Rt) @ self.transf)[:3,:3] * self.worldscale 
                # result["campos"] = Rt[:3, 3] - center
                result["campos"] = np.dot(self.transf[:3, :3].T, Rt[:3, 3] - center) * self.worldscale


                result["focal"] = np.diag(K[:2, :2])
                result["princpt"] = K[:2, 2]
                result["camindex"] = self.all_cameras.index(cam)

                result["image"] = image
                result["imagevalid"] = np.float32(1.0)

            if "pixelcoords" in self.keyfilter:
                if self.subsampletype == "patch":
                    indx = np.random.randint(0, self.loadSize - self.subsamplesize + 1)
                    indy = np.random.randint(0, self.loadSize - self.subsamplesize + 1)

                    px, py = np.meshgrid(
                            np.arange(indx, indx + self.subsamplesize).astype(np.float32),
                            np.arange(indy, indy + self.subsamplesize).astype(np.float32))
                elif self.subsampletype == "random":
                    px = np.random.randint(0, self.loadSize, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    py = np.random.randint(0, self.loadSize, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                elif self.subsampletype == "random2":
                    px = np.random.uniform(0, self.loadSize - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                    py = np.random.uniform(0, self.loadSize - 1e-5, size=(self.subsamplesize, self.subsamplesize)).astype(np.float32)
                else:
                    px, py = np.meshgrid(np.arange(self.loadSize).astype(np.float32), np.arange(self.loadSize).astype(np.float32))

                result["pixelcoords"] = np.stack((px, py), axis=-1)

        return result

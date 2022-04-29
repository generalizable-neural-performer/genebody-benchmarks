import os, sys
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # datadir = '/data/zju'
    # subjects = [subject for subject in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,subject)) and 'CoreView_' in subject]
    # for subject in tqdm(subjects):
    #     uvdir = os.path.join(datadir, subject, 'smpl_uv')
    #     if os.path.isdir(uvdir):
    #         reffiles = sorted(os.listdir(os.path.join(uvdir, 'Camera_B1')))
    #         for cam in tqdm(os.listdir(uvdir)):
    #             files = sorted(os.listdir(os.path.join(uvdir, cam)))
    #             for ref, name in zip(reffiles, files):
    #                 src = os.path.join(uvdir, cam, name)
    #                 tar = os.path.join(uvdir, cam, ref)
    #                 os.system(f'mv {src} {tar}')
    datadir = '../smpl-nerf/data/zju'
    subjects = [subject for subject in os.listdir(datadir) if os.path.isdir(os.path.join(datadir,subject)) and 'CoreView_' in subject]
    for subject in tqdm(subjects):
        uvdir = os.path.join(datadir, subject, 'smpl_uv')
        if os.path.isdir(uvdir):
            reffiles = sorted(os.listdir(os.path.join(uvdir, 'Camera_B1')))
            for cam in tqdm(os.listdir(uvdir)):
                files = sorted(os.listdir(os.path.join(uvdir, cam)))
                if len(reffiles) != len(files):
                    print(subject, cam, len(files))
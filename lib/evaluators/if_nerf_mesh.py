import numpy as np
from lib.config import cfg
import os

def chamfer(x_verts, gt_verts, x_normals=None, gt_normals=None):

    searcher = KDTree(gt_verts)
    dists, inds = searcher.query(x_verts)

    if x_normals is None or gt_normals is None:
        return dists
    elif x_normals is not None and gt_normals is not None:
        cosine_dists = 1 - np.sum(x_normals * gt_normals[inds], axis=1)
        return dists, cosine_dists
    else:
        raise Exception("provide normals for both point sets")


class Evaluator:
    def evaluate(self, output, batch):
        cube = output['cube']
        cube = cube[10:-10, 10:-10, 10:-10]

        pts = batch['pts'][0].detach().cpu().numpy()
        pts = pts[cube > cfg.mesh_th]

        i = batch['i'].item()
        result_dir = os.path.join(cfg.result_dir, 'pts')
        os.system('mkdir -p {}'.format(result_dir))
        result_path = os.path.join(result_dir, '{}.npy'.format(i))
        np.save(result_path, pts)

    def summarize(self):
        return {}

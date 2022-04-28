from pdb import set_trace as st
from selectors import BaseSelector
import sys
sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from torch.utils.data import DataLoader
from pdb import set_trace as st

def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = {scene_name: {}}
    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        cam_id =  rgb_path.split('/')[-2]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        try:
            mesh_param = data['mesh_param']
        except BaseException as e:
            print(e)
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            bb_min = [-256, -256,-256]
            bb_max = [256, 256, 256]
            linspaces = [np.linspace(bb_min[i], bb_max[i], 256) for i in range(len(bb_min))]
            grids = np.stack(np.meshgrid(linspaces[0], linspaces[1], linspaces[2], indexing='ij'), -1)
            sh = grids.shape
            pts = grids.reshape(-1,3)
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
            pts = torch.from_numpy(pts).cuda().float()
            from collections import OrderedDict
            all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                                ('outputs_fine', OrderedDict())])

            pts = pts.reshape(-1, args.N_samples, 3)
            N_rays = pts.shape[0]
            chunk_size = 2000
            all_sigmas = []
            for i in range(0, N_rays, chunk_size):
                chunk = OrderedDict()
                for k in ray_batch:
                    if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras', 'mesh_param']:
                        chunk[k] = ray_batch[k]
                    elif ray_batch[k] is not None:
                        chunk[k] = ray_batch[k][i:i+chunk_size]
                    else:
                        chunk[k] = None
                rgb_feat_sampled, ray_diff, mask = projector.compute(pts[i:i + chunk_size].cuda(),
                                                                ray_batch['camera'].cuda(),
                                                                ray_batch['src_rgbs'].cuda(),
                                                                ray_batch['src_cameras'].cuda(),
                                                                featmaps=featmaps[1].cuda(),
                                                                ori_pts=pts[i:i + chunk_size].cuda())
                ret = model.net_coarse(rgb_feat_sampled, ray_diff, mask)
                sigma = ret[:, :, -1]
                all_sigmas.append(sigma.reshape(-1, 1).cpu().numpy())
            all_sigmas = np.concatenate(all_sigmas, axis=0)
            all_sigmas = all_sigmas.reshape(256, 256, 256)
            from skimage import measure
            verts, faces, normals, _ = measure.marching_cubes_lewiner(all_sigmas, 0.0)
            save_obj_mesh(f'{out_scene_dir}/test_{file_id}.obj', verts, faces)
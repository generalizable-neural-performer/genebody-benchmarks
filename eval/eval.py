from pdb import set_trace as st
import sys

sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
from torch.utils.data import DataLoader
from metrics import ssim, psnr, lpips, masked_psnr

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

    for scene_name in args.eval_scenes:
        out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
        os.makedirs(out_scene_dir, exist_ok=True)

        test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=[scene_name])
        print(f'scene: {scene_name}, num test frame {len(test_dataset)}')
        save_prefix = scene_name
        test_loader = DataLoader(test_dataset, batch_size=1)
        total_num = len(test_loader)
        results_dict = {scene_name: {}}
        LPIPS = []
        SSIM = []
        PSNR = []
        MASKED_PSNR = []

        for i, data in enumerate(test_loader):
            rgb_path = data['rgb_path'][0]

            file_id = os.path.basename(rgb_path).split('.')[0]
            cam_id = rgb_path.split('/')[-2]
            src_rgbs = data['src_rgbs'][0].cpu().numpy()

            averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),
                            averaged_img)
            model.switch_to_eval()
            with torch.no_grad():
                ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
                ray_batch = ray_sampler.get_all()
                featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
                if 'mesh_param' in ray_batch.keys() and 'mesh_param' in data.keys():
                    ray_batch['mesh_param'] = data['mesh_param']
                ret = render_single_image(ray_sampler=ray_sampler,
                                        ray_batch=ray_batch,
                                        model=model,
                                        projector=projector,
                                        chunk_size=args.chunk_size,
                                        det=True,
                                        N_samples=args.N_samples,
                                        inv_uniform=args.inv_uniform,
                                        N_importance=args.N_importance,
                                        white_bkgd=args.white_bkgd,
                                        featmaps=featmaps)

                gt_rgb = data['rgb'][0]
                coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
                coarse_pred_rgb_np = np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
                gt_rgb_np = gt_rgb.numpy()
                gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_{}_gt_rgb.png'.format(file_id, cam_id)), gt_rgb_np_uint8)
                coarse_pred_rgb_np = (255 * np.clip(coarse_pred_rgb_np, a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_{}_pred_coarse.png'.format(file_id, cam_id)), coarse_pred_rgb_np[0])
                if ret['outputs_fine'] is not None:
                    fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                    fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy(), a_min=0., a_max=1.)
                    LPIPS.append(lpips((fine_pred_rgb_np * 255).astype(np.uint8), (gt_rgb_np * 255).astype(np.uint8)))
                    PSNR.append(psnr((fine_pred_rgb_np * 255).astype(np.uint8), (gt_rgb_np * 255).astype(np.uint8)))
                    MASKED_PSNR.append(masked_psnr((fine_pred_rgb_np * 255).astype(np.uint8), (gt_rgb_np * 255).astype(np.uint8)))
                    SSIM.append(ssim((fine_pred_rgb_np * 255).astype(np.uint8), (gt_rgb_np * 255).astype(np.uint8)))
                    fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
                    fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                    imageio.imwrite(os.path.join(out_scene_dir, '{}_{}_pred_fine.png'.format(file_id, cam_id)), fine_pred_rgb)
                else:
                    pass

        results_string = f'psnr {np.mean(PSNR)} ssim {np.mean(SSIM)} lpips {np.mean(LPIPS)} masked_psnr {np.mean(MASKED_PSNR)}'
        f = open("{}/psnr_{}_{}.txt".format(extra_out_dir, save_prefix, model.start_step), "w")
        f.write(results_string)
        f.close()


import numpy as np
from lib.config import cfg
from skimage.measure import compare_ssim
import os
import cv2
from termcolor import colored
import lpips
import torch

from pdb import set_trace as st
class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []
        self.lpips = []
        self.lpips_net = lpips.LPIPS(net='alex', verbose=False)
        self.rank = int(os.environ['RANK']) if 'RANK' in os.environ.keys() else 0

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch):
        if not cfg.eval_whole_img:
            mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
            H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
            mask_at_box = mask_at_box.reshape(H, W)
            # crop the object region
            x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
            img_pred = img_pred[y:y + h, x:x + w]
            img_gt = img_gt[y:y + h, x:x + w]

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        if self.rank == 0:
            cv2.imwrite(
                '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                    view_index),
                (img_pred[..., [2, 1, 0]] * 255))
            cv2.imwrite(
                '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                        view_index),
                (img_gt[..., [2, 1, 0]] * 255))

        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def lpips_metric(self, img_pred, img_gt):
        x = torch.from_numpy(img_pred).float()
        gt = torch.from_numpy(img_gt).float()
        if torch.max(gt) > 128:
            # [0, 255]
            x = x / 255. * 2 - 1
            gt = gt / 255. * 2 - 1
        elif torch.min(gt) >= 0 and torch.max(gt) <= 1:
            # [0, 1]
            x = x * 2 - 1
            gt = gt * 2 - 1
        x = x.permute([2, 0, 1]).unsqueeze(0)
        gt = gt.permute([2, 0, 1]).unsqueeze(0)
        with torch.no_grad():
            loss = self.lpips_net.forward(x, gt)
        return loss.item()

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image
        white_bkgd = int(cfg.white_bkgd)
        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt
        # fill holes
        img_pred[np.sum(img_pred, axis=-1) < 1e-2] = 255
        if cfg.eval_whole_img:
            rgb_pred = img_pred
            rgb_gt = img_gt

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        rgb_pred = img_pred
        rgb_gt = img_gt
        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        self.lpips.append(self.lpips_metric(rgb_pred, rgb_gt))

    def summarize(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        if self.rank == 0:
            result_path = os.path.join(cfg.result_dir, 'metrics.npy')
            os.system('mkdir -p {}'.format(os.path.dirname(result_path)))
            metrics = {'psnr': self.psnr, 'ssim': self.ssim, 'lpips': self.lpips}
            np.save(result_path, metrics)
            print('mse: {}'.format(np.mean(self.mse)))
            print('psnr: {}'.format(np.mean(self.psnr)))
            print('ssim: {}'.format(np.mean(self.ssim)))
            print('lpips: {}'.format(np.mean(self.lpips)))

        self.mse = []
        self.psnr = []
        self.ssim = []

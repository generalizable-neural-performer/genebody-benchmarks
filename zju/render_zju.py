import argparse
import cv2
import numpy as np
import os
from skimage import img_as_ubyte
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
sys.path.insert(0, os.getcwd())
import config
from zju_dataset import ZJUDataset
from model.pipeline import PipeLine
import metrics
import imageio
from util import natural_sort

parser = argparse.ArgumentParser()
parser.add_argument('--loadSize', type=int,  default=512)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
parser.add_argument('--uv', type=str, help='directory to annotation')
parser.add_argument('--subject', type=str, help='directory to data')
parser.add_argument('--test', default=config.TEST_SET, help='index list of test uv_maps')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save checkpoint')
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--out_mode', type=str, default='image', choices=('video', 'image'))
parser.add_argument('--fps', type=int, default=30)
args = parser.parse_args()


if __name__ == '__main__':
    subject = args.subject
    log_dir = os.path.join(args.logdir, subject)
    if args.uv is None:
        args.uv = args.data

    checkpoint_files = natural_sort([os.path.join(log_dir, pt_file) for pt_file in os.listdir(log_dir) if '.pt' in pt_file])

    dataset = ZJUDataset(args.data, args.uv, args.loadSize, args.loadSize, subject, False, 
                        view_direction=args.view_direction, eval_skip=15)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # typically epoch 100 has the best performance on unseen pose
    recommand_checkpoint = os.path.join(log_dir, 'epoch_100.pt')
    checkpoint = checkpoint_files[-1] if recommand_checkpoint not in checkpoint_files else recommand_checkpoint
    print('loading ', checkpoint)
    model = torch.load(checkpoint, map_location='cpu')
    model = model.to('cuda')
    model.eval()
    torch.set_grad_enabled(False)
    out_dir = os.path.join(log_dir, 'eval_'+os.path.basename(checkpoint)[6:-3])
    os.makedirs(out_dir, exist_ok=True)

    if args.out_mode == 'video':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(os.path.join(out_dir, 'render.mp4'), fourcc, 16,
                                     (dataset.width, dataset.height), True)
    print('Evaluating started')
    idx = 0
    lpips_list, psnr_list, ssim_list = [], [], []
    lpips, psnr, ssim = metrics.LPIPS().cuda(), metrics.psnr, metrics.SSIM().cuda()
    with torch.no_grad():
        print(len(dataloader))
        for i, samples in tqdm(enumerate(dataloader)):
            images, uv_maps, extrinsics, masks = samples
            images, uv_maps, extrinsics, masks = images.cuda(), uv_maps.cuda(), extrinsics.cuda(), masks.cuda()
            RGB_texture, preds = model(uv_maps, extrinsics)
            preds, pred_masks = torch.split(preds, [3,1], dim=1)
            
            gt = images.masked_fill_(masks <= 0, 0).clamp(0, 1)
            x = preds.masked_fill_(pred_masks <= 0.55, 0).clamp(0, 1)
            lpips_list.append(lpips(x, gt).item())
            ssim_list.append(ssim(x, gt).item())
            psnr_list.append(psnr(x, gt).item())
            # gt = gt[0].cpu().numpy().transpose((1,2,0))
            # x = x[0].cpu().numpy().transpose((1,2,0))
            
            preds = preds.cpu()
            preds.masked_fill_(pred_masks.cpu() <= 0.55, 0) # fill invalid with 0

            # save result
            if args.out_mode == 'video':
                preds = preds.numpy()
                preds = np.clip(preds, -1.0, 1.0)
                for i in range(len(preds)):
                    image = img_as_ubyte(preds[i])
                    image = np.transpose(image, (1,2,0))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    writer.write(image)
            else:
                for i in range(len(preds)):
                    image = transforms.ToPILImage()(preds[i])
                    image.save(os.path.join(out_dir, '{:03d}_render.png'.format(idx)))
                idx += 1

        lpips_value = np.array(lpips_list).mean()
        ssim_value = np.array(ssim_list).mean()
        psnr_value = np.array(psnr_list).mean()
        with open(os.path.join(out_dir, 'eval.txt'), 'w') as f:
            f.write('psnr: %f\n' % psnr_value)
            f.write('ssim: %f\n' % ssim_value)
            f.write('lpips: %f\n' % lpips_value)
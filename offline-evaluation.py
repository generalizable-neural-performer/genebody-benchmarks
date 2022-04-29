import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import lpips
import os
import imageio
from tqdm import tqdm

class LPIPS(torch.nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.net = lpips.LPIPS(net='alex', verbose=False)

    def forward(self, x, gt):
        if torch.max(gt) > 128:
            # [0, 255]
            x = x / 255. * 2 - 1
            gt = gt / 255. * 2 - 1
        elif torch.min(gt) >= 0 and torch.max(gt) <= 1:
            # [0, 1]
            x = x * 2 - 1
            gt = gt * 2 - 1
        with torch.no_grad():
            loss = self.net.forward(x, gt)
        # return loss.item()
        return loss

def psnr(x, gt):
    """
    x: np.uint8, HxWxC, 0 - 255
    gt: np.uint8, HxWxC, 0 - 255
    """
    if torch.max(gt) > 128:
        # [0, 255]
        x = x / 255
        gt = gt / 255
    elif torch.min(gt) < -1:
        # [0, 1]
        x = (x+1)/2
        gt = (gt+1)/2

    mse = torch.mean((x - gt) ** 2)
    psnr = -10. * torch.log10(mse)
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).\

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        if len(list(img1.shape)) < 4:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim_(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True, default='jinyutong',
                        help='subject to evaluate')
    parser.add_argument('--basedir', type=str, required=False, default='render_output/',
                        help='path to saved img')
    parser.add_argument('--outputdir', type=str, required=False, default='data/eval',
                        help='path to save score result')


    return parser

def run_evaluate():
    args = config_parser().parse_args()
    subject = args.subject
    basedir = os.path.join(args.basedir, 'ghr_test_' + subject)
    imgs_dir = os.path.join(basedir, 'image')
    gt_dir = os.path.join(basedir, 'gt')
    lpips_calculator = LPIPS()
    lpips_all = []
    ims = os.listdir(imgs_dir)
    for i, frame_name in enumerate(tqdm(ims)):
        img_path = os.path.join(imgs_dir, frame_name)
        gt_path = os.path.join(gt_dir, frame_name)
        img = imageio.imread(img_path)
        gt = imageio.imread(gt_path)
        img = torch.tensor(img)
        gt = torch.tensor(gt)
        img = img.reshape(3, 512, 512)
        gt = gt.reshape(3, 512, 512)
        #print(img.shape, gt.shape)
        lpip = lpips_calculator(img, gt)
        lpips_all.append(np.array(lpip.cpu()))
    os.makedirs(os.path.join(args.outputdir, subject), exist_ok=True)
    with open(os.path.join(args.outputdir, subject, 'score_lpips' + '.txt'), 'w') as f:
        avg = np.mean(lpips_all)
        f.write(f'lpips: {avg}\n')


if __name__ == '__main__':
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    run_evaluate()
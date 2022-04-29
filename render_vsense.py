# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Render object from training camera viewpoint or novel viewpoints."""
import argparse
import importlib
import importlib.util
import os
import sys
import time
from metrics import psnr, masked_psnr, lpips, ssim
sys.dont_write_bytecode = True
from pdb import set_trace as st
import torch
import torch.utils.data
import numpy as np
import cv2
torch.backends.cudnn.benchmark = True # gotta go fast!

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Render", help='config profile')
    parser.add_argument('--datadir', type=str, default="./data/vsense", help='directory for data')
    parser.add_argument('--subject', type=str, default="loot", help='subject to train')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.join('logs', f'{args.subject}_{os.path.basename(args.experconfig)[:-3]}')
    print(" ".join(sys.argv))
    print("Output path:", outpath)
    torch.multiprocessing.set_start_method('spawn')

    # load config
    experconfig = import_module(args.experconfig, "config_vsense")
    # print({k: v for k, v in vars(args).items()})
    profile = getattr(experconfig, args.profile)(subject=args.subject, showtarget=True, showdiff=True, viewtemplate=False)

    # load datasets
    dataset = profile.get_dataset(args.datadir, args.subject)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    # data writer
    writer = profile.get_writer()

    # build autoencoder
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    # load
    # state_dict = ae.state_dict()
    # trained_state_dict = torch.load("{}/aeparams.pt".format(outpath))
    # trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in state_dict}
    # state_dict.update(trained_state_dict)
    # ae.module.load_state_dict(state_dict, strict=False)
    ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)

    # eval
    iternum = 0
    itemnum = 0
    starttime = time.time()
    rgbs2write = []
    with torch.no_grad():
        P = []
        L = []
        S = []
        MP = []
        for i, data in enumerate(dataloader):
            b = next(iter(data.values())).size(0)
            # forward
            output = ae(iternum, [], **{k: x.to("cuda") for k, x in data.items()}, **profile.get_ae_args())
            writer.batch(iternum, itemnum + torch.arange(b), **data, **output)
            for batch_idx in range(len(output['irgbrec'])):
                pred =  output["irgbrec"][batch_idx].data.to("cpu").numpy().transpose((1, 2, 0))[..., ::-1].astype(np.uint8).copy()
                rgbs2write.append(pred)
                gt = data["image"][batch_idx].data.to("cpu").numpy().transpose((1, 2, 0))[..., ::-1].copy()
                cv2.imwrite(os.path.join(outpath, f'test_gt_{i:05d}.jpg'), gt)
                cv2.imwrite(os.path.join(outpath, f'test_pred_{i:05d}.jpg'), pred)
                P.append(psnr(pred, gt))
                L.append(lpips(pred, gt))
                S.append(ssim(pred, gt))
                MP.append(masked_psnr(pred, gt))
            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()

            iternum += 1
            itemnum += b
    import numpy as np
    # cleanup
    writer.finalize()
    outstring = (f'psnr:\t {np.mean(P)} \n'
                 f'lpips:\t {np.mean(L)} \n'
                 f'ssim:\t {np.mean(S)} \n'
                 f'masked psnr:\t {np.mean(MP)} \n')
    with open(f'eval_{args.experconfig[:-3]}_{args.subject}', 'w') as f:
        f.write(outstring)
    import imageio
    imageio.mimwrite(os.path.join(outpath, f'nv_{args.subject}_demo.mp4'), rgbs2write, quality=8, fps=30)
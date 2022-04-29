import argparse
import numpy as np
import os, sys
import random
from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
import config
from genebody_dataset import GeneBodyDataset
from model.pipeline import PipeLine
from tqdm import tqdm, trange
import imageio
from util import natural_sort

parser = argparse.ArgumentParser()
# parser.add_argument('--texturew', type=int, default=config.TEXTURE_W)
# parser.add_argument('--textureh', type=int, default=config.TEXTURE_H)
parser.add_argument('--loadSize', type=int,  default=512)
parser.add_argument('--texture_dim', type=int, default=config.TEXTURE_DIM)
parser.add_argument('--use_pyramid', type=bool, default=config.USE_PYRAMID)
parser.add_argument('--view_direction', type=bool, default=config.VIEW_DIRECTION)
parser.add_argument('--data', type=str, default=config.DATA_DIR, help='directory to data')
# parser.add_argument('--annot', type=str, default=config.DATA_DIR, help='directory to annotation')
parser.add_argument('--uv', type=str, help='directory to annotation')
parser.add_argument('--checkpoint', type=str, default=config.CHECKPOINT_DIR, help='directory to save checkpoint')
parser.add_argument('--logdir', type=str, default=config.LOG_DIR, help='directory to save checkpoint')
parser.add_argument('--subject', default='fuzhizhi', type=str)
parser.add_argument('--epoch', type=int, default=config.EPOCH)
parser.add_argument('--batch', type=int, default=config.BATCH_SIZE)
parser.add_argument('--lr', type=float, default=config.LEARNING_RATE)
parser.add_argument('--betas', type=str, default=config.BETAS)
parser.add_argument('--l2', type=str, default=config.L2_WEIGHT_DECAY)
parser.add_argument('--eps', type=float, default=config.EPS)
parser.add_argument('--load', type=str, default=config.LOAD)
parser.add_argument('--load_step', type=int, default=config.LOAD_STEP)
parser.add_argument('--epoch_per_checkpoint', type=int, default=config.EPOCH_PER_CHECKPOINT)
args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, original_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch <= 5:
        lr = original_lr * 0.2 * epoch
    elif epoch < 50:
        lr = original_lr
    elif epoch < 100:
        lr = 0.1 * original_lr
    else:
        lr = 0.01 * original_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    subject = args.subject
    named_tuple = time.localtime()
    # time_string = time.strftime("%m_%d_%Y_%H_%M", named_tuple)
    log_dir = os.path.join(args.logdir, subject)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    if args.uv is None:
        args.uv = args.data

    dataset = GeneBodyDataset(args.data, args.uv, args.loadSize, args.loadSize, subject, True, view_direction=args.view_direction)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=5)

    # if args.load:
    pts = natural_sort([dir_ for dir_ in os.listdir(log_dir) if '.pt' in dir_])
    if len(pts) > 0:
        print('Loading Saved Model')
        model = torch.load(os.path.join(log_dir, pts[-1]))
        step = args.load_step
    else:
        model = PipeLine(args.loadSize, args.loadSize, args.texture_dim, args.use_pyramid, args.view_direction)
        step = 0

    l2 = args.l2.split(',')
    l2 = [float(x) for x in l2]
    betas = args.betas.split(',')
    betas = [float(x) for x in betas]
    betas = tuple(betas)

    optimizer = Adam([
        {'params': model.texture.layer1, 'weight_decay': l2[0], 'lr': args.lr},
        {'params': model.texture.layer2, 'weight_decay': l2[1], 'lr': args.lr},
        {'params': model.texture.layer3, 'weight_decay': l2[2], 'lr': args.lr},
        {'params': model.texture.layer4, 'weight_decay': l2[3], 'lr': args.lr},
        {'params': model.unet.parameters(), 'lr': 0.1 * args.lr}],
        betas=betas, eps=args.eps)
    model = model.to(0)
    model.train()
    torch.set_grad_enabled(True)
    criterion = nn.L1Loss()
    print('Training started')
    for i in trange(1, 1+args.epoch):
        tqdm.write('Epoch {}'.format(i))
        adjust_learning_rate(optimizer, i, args.lr)
        for samples in tqdm(dataloader):
            images, uv_maps, extrinsics, masks = samples
            images, uv_maps, extrinsics, masks = images.cuda(), uv_maps.cuda(), extrinsics.cuda(), masks.cuda()
            step += images.shape[0]
            optimizer.zero_grad()
            RGB_texture, preds = model(uv_maps, extrinsics)
            preds, pred_masks = torch.split(preds, [3,1], dim=1)

            images = masks * images
            RGB_texture = RGB_texture * masks
            preds = preds * masks

            loss1 = criterion(RGB_texture, images)
            loss2 = criterion(preds, images)
            loss3 = criterion(pred_masks, masks) * 0.1
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            if step % (args.batch * 20) == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                tqdm.write(f'texture loss: {loss1.item():.3e} unet loss: {loss2.item():.3e} mask loss: {loss3.item():.3e}')
        # save checkpoint
        if i % args.epoch_per_checkpoint == 0:
            tqdm.write('Saving checkpoint')
            torch.save(model, log_dir+'/epoch_{}.pt'.format(i))

if __name__ == '__main__':
    main()

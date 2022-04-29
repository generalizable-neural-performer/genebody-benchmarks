#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
from os import mkdir
# from apex import amp
import shutil
from pdb import set_trace as st

import torch.nn.functional as F

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(ROOT_PATH))# st()
sys.path.append(os.path.abspath(os.path.join(ROOT_PATH, 'third_party', 'PCPR')))
sys.path.append(os.path.abspath(os.path.join(ROOT_PATH, 'third_party', 'antialiased-cnns')))
from config import cfg
from data import make_data_loader
from data.datasets.genebody_total import GeneBodyTotalDataset
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR, build_scheduler 
from layers import make_loss

from utils.logger import setup_logger

from torch.utils.tensorboard import SummaryWriter
# from tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    torch.cuda.set_device(0)
    cfg_file, dataset_dir, subject = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg.merge_from_file(cfg_file)
    cfg.OUTPUT_DIR = f'./logs/{subject}'
    cfg.DATASET.TRAIN = dataset_dir
    cfg.DATASETS.SUBJECT = subject
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    writer = SummaryWriter(log_dir=os.path.join(output_dir,'tensorboard'))

    logger = setup_logger("rendering_model", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))

    # shutil.copy(sys.argv[1], os.path.join(cfg.OUTPUT_DIR,'configs.yml'))
    os.system(f'cp {sys.argv[1]} {cfg.OUTPUT_DIR}/configs.yml')

    train_loader, dataset = make_data_loader(cfg, dataset=GeneBodyTotalDataset, is_train=True,is_center = cfg.DATASETS.CENTER)
    model = build_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    #scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS, cfg.SOLVER.LR_SCALE)

    loss_fn = make_loss(cfg)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    do_train(
            cfg,
            model,
            train_loader,
            None,
            optimizer,
            scheduler,
            loss_fn,
            writer
        )







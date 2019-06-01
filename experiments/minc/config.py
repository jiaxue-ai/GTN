import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

C = edict()
config = C

C.seed = 0

"""please config ROOT_dir and user when u first using"""
C.repo_name = 'GTN'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]
C.log_dir = osp.abspath(osp.join(C.root_dir, 'log', C.this_dir))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

"""Data Dir and Weight Dir"""
C.dataset_path = "/home/jia/Downloads/database/minc-2500/"
C.img_root_folder = C.dataset_path
C.gt_root_folder = C.dataset_path
C.train_source = osp.join(C.dataset_path, "train1.txt")
C.eval_source = osp.join(C.dataset_path, "test1.txt")
C.test_source = osp.join(C.dataset_path, "test1.txt")
C.is_test = False

"""Path Config"""

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(C.root_dir)

"""Image Config"""
C.num_classes = 23
C.image_mean = np.array([0.485, 0.456, 0.406])
C.image_std = np.array([0.229, 0.224, 0.225])

"""Train Config"""
C.lr = 1e-2
C.lr_decay = 40
C.momentum = 0.9
C.weight_decay = 1e-4
C.batch_size = 64
C.nepochs = 100
C.aux_weight = 0.2

"""Eval Config"""
C.eval_iter = 30

"""Display Config"""
C.snapshot_iter = 50
C.record_info_iter = 20
C.display_iter = 50

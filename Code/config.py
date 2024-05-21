from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function



import os.path as osp
import sys
import time
import numpy as np
import argparse
from easydict import EasyDict as edict


C = edict()
config = C
cfg = C

C.seed = 888

C.abs_dir = osp.dirname(osp.realpath(__file__))
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.repo_name = 'bc_agent'


C.log_dir = osp.abspath(osp.join(C.abs_dir, 'log'))
C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_dir + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.this_dir, 'models'))

"""Data Config"""
C.EPISODE_FOLDER = 'episode_%d_synchronized'
C.TRAIN_REPLAY_STORAGE_DIR = '<path-to-training-data>'
C.TEST_REPLAY_STORAGE_DIR = '<path-to-test-data>'

"""Logging Config"""
C.wandb_activate = False
C.data_dir = "<path-to-dataset>"
C.train_split = 12
C.interfaces =  [3]
C.tasks = ['2'] #, '2', '3', '4','5', '6', '7','8'] #,'2', '5','6', '4', '8']

"""Scene Config"""
C.SCENE_BOUNDS = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
C.CAMERAS = ['wrist', 'left_shoulder', 'right_shoulder']
C.BATCH_SIZE = 8
C.VOXEL_SIZES = [100]
C.IMAGE_SIZE = 128
C.device = 'cuda'
C.rotation_resolution = 5


C.LOG_FREQ = 1
C.SAVE_FREQ = 5000
C.TRAINING_ITERATIONS = 10000
C.TEST_ITERATIONS = 1000

"""Model Config"""
C.agent = 'diffuser_agent'
C.optimizer = 'adamW'
C.lr = 1e-4
C.initial_dim=512,
C.low_dim_size=64

C.save_path = f'./{C.agent}/checkpoint'


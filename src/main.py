import os
import argparse
import numpy as np
import torch

from utils.config import get_cfg
from manager import DLManager

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='/data/hjw/events/Bi_CMPStereo/config_lt.yaml')
parser.add_argument("--data_root", type=str, default='/data/hjw/events/DSEC')
parser.add_argument("--save_root", type=str, default='/data/hjw/events/Bi_CMPStereo/save/train_all')
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--save_term", type=int, default=1)
##############################################################################
parser.add_argument("--is_continue_train", type=bool, default=True)
parser.add_argument("--continue_path", type=str, default='/data/hjw/events/Bi_CMPStereo/save/train_all/weights/final.pth')
parser.add_argument("--cross_modality", type=bool, default=True)
parser.add_argument("--image_location", type=str, default='left')
parser.add_argument("--discriminator", type=bool, default=False)
parser.add_argument("--double_img", type=bool, default=False)

args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

cfg = get_cfg(args.config_path)

np.random.seed(35)
torch.manual_seed(35)

exp_manager = DLManager(args, cfg)
exp_manager.test_all()

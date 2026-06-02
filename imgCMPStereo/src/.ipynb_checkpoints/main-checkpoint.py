import os
import argparse
import numpy as np
import torch

from utils.config import get_cfg
from manager import DLManager

parser = argparse.ArgumentParser()
# 配置文件路径
parser.add_argument("--config_path", type=str, default='/autodl-fs/data/DSEC/New/configs/config_lt.yaml')
# 数据集根路径
parser.add_argument("--data_root", type=str, default='/root/autodl-tmp/DSEC-ssh')
# log 训练数据存储路径
parser.add_argument("--save_root", type=str, default='/autodl-fs/data/DSEC/New/save/1')
# 多少线程 dataloader
parser.add_argument("--num_workers", type=int, default=4)
# 多少 epoch 后存储一次 checkpoint
parser.add_argument("--save_term", type=int, default=25)
##############################################################################
# 是否从 checkpoint 重新训练
parser.add_argument("--is_continue_train", type=bool, default=True)
# parser.add_argument("--continue_path", type=str, default='/data/hjw/events/DSEC/EF-CMS_att/save/1/weights/final.pth')
parser.add_argument("--continue_path", type=str, default='/root/autodl-fs/EF-CMS/checkpoint.pth')
# 是否跨模态，以及左右目对应的事件--图像
parser.add_argument("--cross_modality", type=bool, default=True)
parser.add_argument("--image_location", type=str, default='left')
parser.add_argument("--discriminator", type=bool, default=False)


#解析参数
args = parser.parse_args()

args.is_distributed = False
args.is_master = True
args.world_size = 1
args.local_rank = 0

#检查参数，配置文件是否为文件，数据集是否为目录
assert os.path.isfile(args.config_path)
assert os.path.isdir(args.data_root)

#读取配置文件，生成具体配置
cfg = get_cfg(args.config_path)

np.random.seed(35)
torch.manual_seed(35)

exp_manager = DLManager(args, cfg)
if not args.discriminator:
    exp_manager.train_test()
else:
    exp_manager.train_discriminator()

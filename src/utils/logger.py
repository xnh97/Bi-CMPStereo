import os
import shutil
import torch

from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

# 由文件路径 (txt/yaml) 打开文件，写入信息
class Log:
    def __init__(self, log_path):
        self._log_path = log_path

    # 信息 print 并写入文件，end 以回车'\n' 等方式结尾，mode 以追加'a'/重写'w' 等方式打开文件
    def write(self, log, mode='a', end='\n', is_print=True):
        # 信息 print 到终端
        if is_print:
            print(log, end=end)
        # 写入文件
        with open(self._log_path, mode=mode) as f:
            f.write(log + end)

        
# log 记录训练过程
class ExpLogger:
    _FILE_NAME = {
        'args': 'args.txt',
        'config': 'config.yaml',
        'model': 'model.txt',
        'optimizer': 'optimizer.txt',
        'train': 'train_log.txt',
        'validation': 'validation_log.txt',
        'test': 'test_log.txt',
    }
    _DIR_NAME = {
        'src': 'src',
        'weight': 'weights',
        'visualize': 'visualize',
    }
    
    # 记录器初始化，新建 save 文件夹
    def __init__(self, save_root, mode='train'):
        assert mode in ['train', 'validation', 'test']
        self._save_root = save_root
        self._mode = mode
        self._tensor_log = None
        # 新建 save 文件夹
        os.makedirs(self._save_root, exist_ok=True)
    
    # 设置 mode，主要用于创建 'train'/'test' 文件
    def train(self):
        self._mode = 'train'

    # 设置 mode，主要用于创建 'train'/'test' 文件
    def test(self):
        self._mode = 'test'
    
    # 根据路径/内容/结尾方式，将内容写入对应的 txt/yaml 文件
    # log：写入内容         file_name：写入文件路径，默认为'train_log.txt'
    # mode：以追加'a'/重写'w' 等方式打开文件       end：结尾方式，默认以回车'\n' 结尾，
    def write(self, log, file_name=None, mode='a', end='\n', is_print=True):
        if file_name is None:
            file_name = self._FILE_NAME[self._mode]
        log_path = os.path.join(self._save_root, file_name)
        # 调用 Log 类，写入文件
        logger = Log(log_path)
        logger.write(log=log, end=end, is_print=is_print, mode=mode)
    
    # 将每个 epoch 的训练结果指标通过 tensor_board 记录
    # tag：指标名称       scalar_value：指标值     global_step：当前 epoch
    def add_scalar(self, tag, scalar_value, global_step):
        # 创建 tensorboard 实例并指定保存路径
        if self._tensor_log is None:
            self._tensor_log = SummaryWriter(self._save_root)
        # tensorboard 记录这个 epoch 的某个指标
        self._tensor_log.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step)

    #  用 torch.save 方法将 checkpoint 保存至指定位置，并写入 'train_log.txt' 文件
    def save_checkpoint(self, checkpoint, name):
        # 存储 checkpoint 的文件夹路径 = /home/xnh/event-stereo/SE-CFF/save/weight
        checkpoint_root = os.path.join(self._save_root, self._DIR_NAME['weight'])
        os.makedirs(checkpoint_root, exist_ok=True)
        # 存储 checkpoint 的文件路径 = /home/xnh/event-stereo/SE-CFF/save/weight/name
        checkpoint_path = os.path.join(checkpoint_root, name)
        torch.save(checkpoint, checkpoint_path)
        # 该步骤写入 'train_log.txt' 文件
        # self.write(log='Checkpoint is saved to %s' % checkpoint_path)
    
    # 根据路径使用 torch.load 读取 checkpoint
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.write(log='Checkpoint is Loaded from %s' % checkpoint_path)
        return checkpoint

    def save_visualize(self, image, visual_type, sequence_name, image_name):
        visualize_root = os.path.join(self._save_root, self._DIR_NAME['visualize'], visual_type, sequence_name)
        os.makedirs(visualize_root, exist_ok=True)
        visualize_path = os.path.join(visualize_root, image_name)
        image.save(visualize_path)
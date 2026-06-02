import os
from PIL import Image
import numpy as np
import torch.utils.data

# 定义视差 gt 的数据集，继承 torch.utils.data.Dataset
# 为每个序列分别定义一个该数据集，初始化时存储所有是插图路径并读取时间戳文件
# 包含用于 dataloader 的索引函数，根据时间戳读取视差图
class DisparityDataset(torch.utils.data.Dataset):
    _PATH_DICT={
        'timestamp': 'timestamps.txt',
        'event': 'event',
    }
    # 视差图 gt 中分为 event/image，分辨率不同，这里使用事件分辨率的 groundtruth
    # 事件分辨率 = 640*480，图像分辨率 = 1440*1080
    _DOMAIN = ['event']
    NO_VALUE = 0.0

    # 输入该视差图序列路径 = '/DSEC/train/interlaken_00_c/disparity'
    # 存储该序列中所有事件分辨率的视差真值的路径和事件戳
    def __init__(self, root):
        self.root = root
        # 读取视差图时间戳 txt 文件，转换为 int64 存为 numpy
        self.timestamps = load_timestamp(os.path.join(root, self._PATH_DICT['timestamp']))
        
        # 存储序列中所有事件分辨率的视差真值图像的路径
        self.disparity_path_list = {}
        self.timestamp_to_disparity_path = {}
        for domain in self._DOMAIN:
            # 存储该序列 '/DSEC/train/interlaken_00_c/disparity/event' 中的所有视差图路径list
            self.disparity_path_list[domain] = get_path_list(os.path.join(root, self._PATH_DICT[domain]))
            # 每张视差图的时间戳和路径，构建字典
            self.timestamp_to_disparity_path[domain] = {
                timestamp: filepath for timestamp, filepath in
                zip(self.timestamps, self.disparity_path_list[domain])
            }
        
        # 每张视差图的时间戳和index构建的字典
        self.timestamp_to_index = {
            timestamp: int(os.path.splitext(os.path.basename(self.timestamp_to_disparity_path['event'][timestamp]))[0])
            for timestamp in self.timestamp_to_disparity_path['event'].keys()
        }
    
    def __len__(self):
        return len(self.timestamps)
    
    # 用于 dataloader 的索引函数
    # 关于视差图的索引，根据时间戳索引对应的视差图路径，从文件中读取视差图
    def __getitem__(self, timestamp):
        return load_disparity(self.timestamp_to_disparity_path['event'][timestamp])
    
    # 多个 batch 视差图合并，default_collate 将多个 batch 张量堆叠：
    # n 个 H*W*C 张量构成的 list 经堆叠后成为 n*H*W*C
    # 如果每个 batch 是多个同的张量，对多个该 batch 使用 default_collate 堆叠，也会将对应的张量堆叠
    @staticmethod
    def collate_fn(batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        return batch

# 读取时间戳 txt 文件，转换为 int64 存为 numpy
def load_timestamp(root):
    return np.loadtxt(root, dtype='int64')

# 将该序列 '/DSEC/train/interlaken_00_c/disparity/event' 中的所有视差图路径打包成 list
def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]

# 根据视差图路径，读取视差图
def load_disparity(root):
    disparity = np.array(Image.open(root)).astype(np.float32)/256
    return disparity
import os

import torch.utils.data
import torch.utils.data._utils

from torch.utils.data.distributed import DistributedSampler
from .constant import DATA_SPLIT
from .sequence import SequenceDataset
from .dataloader import MultiEpochsDataLoader

# train/test 的数据集，torch.utils.data.Dataset 的子类，包含各序列数据集
# 初始化将输入数据集路径，对所有需要使用的序列初始化序列数据集
# 之后用于索引的数据集是 ConcatDataset 数据集类，将多有序列合并为一整个数据集，索引时相当于直接作用在 SequenceDataset 的序列上
# 返回 ['file_index': int(时间戳对应的视差图索引), 'event': 左右: 1*480*640*1/2*10, 'disparity': 480*640]
class DSECDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, sampling_ratio, event_cfg, disparity_cfg, crop_height, crop_width, num_workers=0, 
                 cross_modality=False, image_location='left', **kwargs):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers

        # 根据这次 train/test，提取所用的所有序列名称，构成 list
        assert split in DATA_SPLIT.keys()
        sequence_list = DATA_SPLIT[split]

        # 遍历数据集中所有序列，根据序列名称-路径，分别生成序列数据集，存入 list 
        self.sequence_data_list = []
        for sequence in sequence_list:
            sequence_root = os.path.join(root, sequence)
            self.sequence_data_list.append(SequenceDataset(root=sequence_root,
                                                           split=split,
                                                           sampling_ratio=sampling_ratio,
                                                           event_cfg=event_cfg,
                                                           disparity_cfg=disparity_cfg,
                                                           crop_height=crop_height,
                                                           crop_width=crop_width,
                                                           num_workers=num_workers,
                                                           cross_modality=cross_modality,
                                                           image_location=image_location,
                                                           **kwargs))
            
        # 用于索引的数据集是 ConcatDataset 数据集类，由所有序列数据集拼接形成，
        # 对应所有序列的可用视差图和对应事件的集合，而不是 SequenceDataset 的链表
        if len(self.sequence_data_list) == 0:
            self.dataset = []
        else:
            self.dataset = torch.utils.data.ConcatDataset(self.sequence_data_list)
        #  ConcatDataset 类，和 SequenceDataset 链表一样包含了所有的 SequenceDataset，但是区别包括：
        # 1. 其 len 可以返回所有序列的长度 len 之和
        # 2.  self.dataset[idx] 索引操作首先确定属于列表中的哪一个SequenceDataset，再在其中索引特定的元素
        # 3. 而这个类中所有的序列同样是彼此独立
    
    # 整个数据集的长度 = 各个序列的长度之和 = 所有可用视差图/时间戳数量之和
    def __len__(self):
        length = len(self.dataset)
        return length

    # 从整个 ConcatDataset 类数据集上索引，首先确定属于列表中的哪一个SequenceDataset，再从中索引特定的元素
    # 即，就会调用所索引 SequenceDataset 的 __getitem__ 方法，返回使用数据
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    # 在 dataloader 中调用，融合多个 batch，输入多个 __getitem__ 输出构成的 list 进行合并
    def collate_fn(self, batch):
        # self.dataset.datasets 即所有序列数据集构成的 list，使用序列数据集中的 collate 函数融合 batch
        return self.dataset.datasets[0].collate_fn(batch)


# train 时，对整个数据集生成一个 dataloader
def get_multi_epochs_dataloader(dataset, dataloader_cfg, num_workers, is_distributed, world_size):
    if len(dataset) == 0:
        return torch.utils.data.DataLoader(dataset)
    if is_distributed:
        batch_size = dataloader_cfg.PARAMS.batch_size // world_size
        shuffle = dataloader_cfg.PARAMS.get('shuffle', False) # 每个 epoch 是否乱序
        drop_last = dataloader_cfg.PARAMS.get('drop_last', False) # 当样本数不能被 batchsize 整除时是否 drop
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        multi_epochs_dataloader = MultiEpochsDataLoader(dataset=dataset,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        collate_fn=dataset.collate_fn,
                                                        batch_size=batch_size,
                                                        drop_last=drop_last,
                                                        sampler=sampler)
    else:
        # 本质上和 torch.utils.data.DataLoader 一样
        multi_epochs_dataloader = MultiEpochsDataLoader(dataset=dataset,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        collate_fn=dataset.collate_fn,
                                                        **dataloader_cfg.PARAMS)

    return multi_epochs_dataloader        


# test 时，对每个序列分别生成一个 dataloader，返回 list，后面 test 时也分别对每个 dataloader 处理
def get_sequence_dataloader(dataset, dataloader_cfg, num_workers, is_distributed, world_size):
    if len(dataset) == 0:
        return torch.utils.data.DataLoader(dataset)
    # 分布式训练时，使用 sampler，batch_size 除以 world_size
    if is_distributed:
        batch_size = dataloader_cfg.PARAMS.batch_size // world_size
        shuffle = dataloader_cfg.PARAMS.get('shuffle', False) # 每个 epoch 是否乱序
        drop_last = dataloader_cfg.PARAMS.get('drop_last', False) # 当样本数不能被 batchsize 整除时是否 drop
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        sequence_dataloader = [torch.utils.data.DataLoader(dataset=sequence_dataset,
                                                           num_workers=num_workers, # 是否多线程读取数据，和分布式训练无关
                                                           pin_memory=True,
                                                           collate_fn=dataset.collate_fn,
                                                           batch_size=batch_size,
                                                           drop_last=drop_last,
                                                           sampler=sampler)
                               for sequence_dataset in dataset.sequence_data_list]
    else:
        sequence_dataloader = [torch.utils.data.DataLoader(dataset=sequence_dataset,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           collate_fn=dataset.collate_fn,
                                                           **dataloader_cfg.PARAMS)
                               for sequence_dataset in dataset.sequence_data_list]

    return sequence_dataloader


# 配置数据集和 dataloader
def get_dataloader(args, dataset_cfg, dataloader_cfg, is_distributed=False):
    # 生成数据集
    dataset = DSECDataset(root=args.data_root,
                          num_workers = args.num_workers,
                          cross_modality= args.cross_modality,
                          image_location= args.image_location,
                          **dataset_cfg.PARAMS)
    # 从数据集生成用于 train/test 的 dataloader
    dataloader = globals()[dataloader_cfg.NAME](dataset=dataset,
                                                dataloader_cfg=dataloader_cfg,
                                                num_workers=args.num_workers,
                                                is_distributed=is_distributed,
                                                world_size=args.world_size if is_distributed else None,)

    return dataloader
    
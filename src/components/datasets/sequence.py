import os
import csv
import copy

import numpy as np
import torch.utils.data

from .disparity.dataset import DisparityDataset
from .event.dataset import EventDataset
from .image.dataset import ImageDataset
from .event_voxel.dataset import EventVoxelDataset
from . import transforms

class SequenceDataset(torch.utils.data.Dataset):
    _PATH_DICT = {
        'event': 'events',
        'disparity': 'disparity',
        'image': 'images',
        'calibration': 'calibration',
    }
    _LOCATION = ['left', 'right']
    HEIGHT = 480
    WIDTH = 640

    def __init__(self, root, split, sampling_ratio, event_cfg, disparity_cfg, 
                 crop_height, crop_width, num_workers=0, cross_modality=False, image_location='left', double_img=False):
        self.root = root
        self.split = split
        self.sampling_ratio = sampling_ratio
        self.event_cfg = event_cfg
        self.disparity_cfg = disparity_cfg
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_workers = num_workers
        self.double_img = double_img

        self.cross_modality = cross_modality
        self.image_location = image_location
        
        self.event_location = self._LOCATION.copy()
        self.event_location.remove(image_location)
        self.event_location = self.event_location[0]

        self.sequence_name = root.split('/')[-1]
        
        if split in ['test']:
            self.disparity_dataset = None
            disp_timestamps, _ = read_csv(os.path.join(root, self.sequence_name + '.csv'))
        else:
            disparity_root = os.path.join(root, self._PATH_DICT['disparity'])  # 视差真值的文件夹路径  = '/DSEC/train/interlaken_00_c/disparity'
            self.disparity_dataset = DisparityDataset(root=disparity_root, **disparity_cfg.PARAMS)
            disp_timestamps=self.disparity_dataset.timestamps

        event_root = os.path.join(root, self._PATH_DICT['event'])  # 事件的文件夹路径  = '/DSEC/train/interlaken_00_c/events'
        self.event_dataset = EventDataset(root = event_root, cross_modality=self.cross_modality, event_location=self.event_location,
                                          disp_timestamps=disp_timestamps, **event_cfg.PARAMS)
        
        self.event_voxel_dataset = EventVoxelDataset(root = event_root, cross_modality=self.cross_modality, 
                                           event_location=self.event_location, **event_cfg.PARAMS)

        if cross_modality:
            image_root = os.path.join(root, self._PATH_DICT['image'])  # '/DSEC/train/interlaken_00_c/image'
            cali_root = os.path.join(root, self._PATH_DICT['calibration'])  # '/DSEC/train/interlaken_00_c/calibration'
            self.image_dataset = ImageDataset(root=image_root, location=image_location, disp_timestamps=disp_timestamps, root_calib=cali_root)


        if split in ['train', 'validation', 'trainval']:
            self.timestamp_to_index = copy.copy(self.disparity_dataset.timestamp_to_index)
            self.timestamps = copy.copy(self.disparity_dataset.timestamps)
            if event_cfg.NAME == 'sbn':
                    minimum_timestamp = self.event_dataset.minimum_timestamp
                    maximum_timestamp = self.event_dataset.maximum_timestamp
            elif event_cfg.NAME != 'none':  # = sbn
                minimum_timestamp = max(self.event_dataset.event_slicer['left'].t_offset,
                                        self.event_dataset.event_slicer['right'].t_offset)
                maximum_timestamp = None
                if event_cfg.NAME == 'base':
                    minimum_timestamp += self.event_dataset.time_interval
            self.timestamps = self.timestamps[self.timestamps >= minimum_timestamp]
            if maximum_timestamp is not None:
                self.timestamps = self.timestamps[self.timestamps <= maximum_timestamp]
        elif split in ['test']:
            self.timestamps, self.timestamp_to_index = read_csv(os.path.join(root, self.sequence_name + '.csv'))
        else:
            raise NotImplementedError
        
        self.timestamps = self.timestamps[[idx for idx in range(0, len(self.timestamps), sampling_ratio)]]

        if split in ['train', 'trainval']:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(crop_height=crop_height, crop_width=crop_width),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ])
        elif split in ['validation']:
            self.transforms = transforms.Compose([
                transforms.Padding(img_height=crop_height, img_width=crop_width,
                                   no_event_value=self.event_dataset.NO_VALUE,
                                   no_disparity_value=self.disparity_dataset.NO_VALUE),
                transforms.ToTensor(),
            ])
        elif split in ['test']:
            self.transforms = transforms.Compose([
                transforms.Padding(img_height=crop_height, img_width=crop_width,
                                   no_event_value=self.event_dataset.NO_VALUE,
                                   no_disparity_value=None),
                transforms.ToTensor(),
            ])
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        data = self.load_data(idx)
        data = self.transforms(data)
        return data
    
    def load_data(self, idx):
        timestamp = self.timestamps[idx]
        data = {}
        event_data = self.event_dataset[timestamp]
        event_voxel_data = self.event_voxel_dataset[timestamp]
        if self.disparity_dataset is not None:
            disparity_data = self.disparity_dataset[timestamp]
            data['disparity'] = disparity_data

        data['file_index'] = self.timestamp_to_index[timestamp]
        if event_data is not None:
            data['event'] = event_data
        if event_voxel_data is not None:
            data['event_voxel'] = event_voxel_data
        if self.cross_modality is True:
            data['image'] = self.image_dataset[timestamp]
        
        return data
    

    def collate_fn(self, batch):
        output = {}
        domain = 'event'
        if domain in batch[0].keys():
            output[domain] = self.event_dataset.collate_fn([sample[domain] for sample in batch])
        
        domain = 'event_voxel'
        if domain in batch[0].keys():
            output[domain] = self.event_voxel_dataset.collate_fn([sample[domain] for sample in batch])

        domain = 'disparity'
        if domain in batch[0].keys():
            output[domain] = self.disparity_dataset.collate_fn([sample[domain] for sample in batch])

        domain = 'image'
        if domain in batch[0].keys():
            output[domain] = self.image_dataset.collate_fn([sample[domain] for sample in batch])
        
        domain = 'image2'
        if domain in batch[0].keys():
            output[domain] = self.image_dataset.collate_fn([sample[domain] for sample in batch])

        for key in batch[0].keys():
            if key not in ['event', 'disparity', 'image', 'pre_image', 'image2']:
                output[key] = torch.utils.data._utils.collate.default_collate([sample[key] for sample in batch])

        return output
    

def read_csv(csv_file):
    timestamps = []
    timestamp_to_index = {}
    with open(csv_file) as csvfile:
        data_reader = csv.reader(csvfile)
        for row in data_reader:
            assert row[0] not in timestamps
            if row[0].isnumeric():
                timestamps.append(int(row[0]))
                timestamp_to_index[int(row[0])] = int(row[1])

    return np.asarray(timestamps), timestamp_to_index
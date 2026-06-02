import os
import numpy as np
import torch.utils.data
from .slice import EventSlicer
from . import constant

class EventVoxelDataset(torch.utils.data.Dataset):
    _PATH_DICT={
        'timestamp': 'timestamps.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = None
    
    def __init__(self, root, num_of_event, cross_modality=False, event_location='right', **kwargs):
        self.root = root
        self.num_of_event = 800000
        num_of_event = 800000
        self.num_bins = 5 # 5
        
        if cross_modality is True:
            self.event_location = event_location
            self._LOCATION = [event_location]
        
        self.event_slicer = {}
        self.minimum_timestamp, self.maximum_timestamp = -float('inf'), float('inf')
        for location in self._LOCATION:
            event_path = os.path.join(root, location, 'events.h5')
            rectify_map_path = os.path.join(root, location, 'rectify_map.h5')
            self.event_slicer[location] = EventSlicer(event_path, rectify_map_path, num_of_event)
            self.minimum_timestamp = max(self.event_slicer[location].min_time, self.minimum_timestamp)
            self.maximum_timestamp = min(self.event_slicer[location].max_time, self.maximum_timestamp)
    
    def __len__(self):
        return 0
    

    def __getitem__(self, timestamp):
        event_data = {}
        minimum_time, maximum_time = -float('inf'), float('inf')


        for location in self._LOCATION:
            event_data[location] = self.event_slicer[location][timestamp]
            minimum_time = max(minimum_time, event_data[location]['t'].min())
            maximum_time = min(maximum_time, event_data[location]['t'].max())
        
        event_data_np = {}
        for location in self._LOCATION:
            mask = np.logical_and(minimum_time <= event_data[location]['t'], event_data[location]['t'] <= maximum_time)
            for data_type in ['x', 'y', 't', 'p']:
                event_data[location][data_type] = event_data[location][data_type][mask]
            event_data_np[location] = np.column_stack((event_data[location]['t'], event_data[location]['x'], event_data[location]['y'], event_data[location]['p']))
        
        event_voxel = {}
        for location in self._LOCATION:
            event_voxel[location] = self.events_to_voxel_grid(event_data_np[location], self.num_bins, 
                                                              constant.EVENT_WIDTH, constant.EVENT_HEIGHT, True)

        return event_voxel


    def events_to_voxel_grid(self, events, num_bins, width, height, normalization=False):
        assert(events.shape[1] == 4)
        assert(num_bins > 0)
        assert(width > 0)
        assert(height > 0)
        
        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
        
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].astype(int)
        ys = events[:, 2].astype(int)
        pols = events[:, 3]
        pols[pols == 0] = -1  # polarity should be +1 / -1
        
        tis = ts.astype(int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)         # 向上取整的小数部分，表示其与向下整数的接近程度
        vals_right = pols * dts                   # 向下取整的小数部分，表示其与向上整数的接近程度

        valid_indices = tis < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
                  + tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])
        
        if normalization:
                mean, stddev = voxel_grid[voxel_grid != 0].mean(), voxel_grid[voxel_grid != 0].std()
                voxel_grid[voxel_grid != 0] = (voxel_grid[voxel_grid != 0] - mean) / stddev

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width)).transpose(1, 2, 0)

        return voxel_grid

    
    def collate_fn(self, batch):
        batch = torch.utils.data._utils.collate.default_collate(batch)
        return batch
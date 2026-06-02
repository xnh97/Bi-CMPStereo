import os
import numpy as np
import torch.utils.data
from .slice import EventSlicer
from .stack import MixedDensityEventStacking
from . import constant


class EventDataset(torch.utils.data.Dataset):
    _PATH_DICT={
        'timestamp': 'timestamps.txt',
        'left': 'left',
        'right': 'right'
    }
    _LOCATION = ['left', 'right']
    NO_VALUE = None
    
    
    def __init__(self, root, num_of_event, stack_method, stack_size,
                 num_of_future_event=0, use_preprocessed_image=False, 
                 cross_modality=False, event_location='right', disp_timestamps = None, **kwargs):
        self.root = root
        self.num_of_event = num_of_event
        self.stack_method = stack_method
        self.stack_size = stack_size # 10
        self.num_of_future_event = num_of_future_event
        self.use_preprocessed_image = use_preprocessed_image
        self.stack_function = MixedDensityEventStacking(stack_size, num_of_event, constant.EVENT_HEIGHT, constant.EVENT_WIDTH, **kwargs)
        self.NO_VALUE = self.stack_function.NO_VALUE

        
        if cross_modality is False:
            data_define = 'sbn_%d_%s_%d_%d' % (self.num_of_event, self.stack_method, self.stack_size, self.num_of_future_event)
        else:
            data_define = event_location + '_sbn_%d_%s_%d_%d' % (self.num_of_event, self.stack_method, self.stack_size, self.num_of_future_event)
            self.event_location = event_location
            self._LOCATION = [event_location]
        self.save_root = os.path.join(self.root, data_define)

        if not os.path.isdir(self.save_root):
            assert disp_timestamps is not None
            os.makedirs(self.save_root, exist_ok=True)
            
            self.event_slicer = {}
            self.minimum_timestamp, self.maximum_timestamp = disp_timestamps[0], disp_timestamps[-1]
            for location in self._LOCATION:
                event_path = os.path.join(root, location, 'events.h5')
                rectify_map_path = os.path.join(root, location, 'rectify_map.h5')
                self.event_slicer[location] = EventSlicer(event_path, rectify_map_path, num_of_event, num_of_future_event)
                self.minimum_timestamp = max(self.event_slicer[location].min_time, self.minimum_timestamp)
                self.maximum_timestamp = min(self.event_slicer[location].max_time, self.maximum_timestamp)
            
            disp_timestamps = disp_timestamps[(disp_timestamps >= self.minimum_timestamp) & 
                                              (disp_timestamps <= self.maximum_timestamp)]
            
            for i in disp_timestamps:
                save_path = os.path.join(self.save_root, '%ld.npy' % i)
                event_data = self._pre_load_event_data(timestamp = i)
                torch.save(event_data, save_path)
        else:
            stack_path_list = get_path_list(self.save_root)
            self.minimum_timestamp = int(stack_path_list[0].split('/')[-1].strip('.npy'))
            self.maximum_timestamp = int(stack_path_list[-1].split('/')[-1].strip('.npy'))

    
    def __len__(self):
        return 0
    
    def __getitem__(self, timestamp):
        
        save_path = os.path.join(self.save_root, '%ld.npy' % timestamp)
        event_data = torch.load(save_path)
        event_data = self._post_load_event_data(event_data)
        return event_data
    
    
    def _pre_load_event_data(self, timestamp):
        event_data = {}
        minimum_time, maximum_time = -float('inf'), float('inf')
        for location in self._LOCATION:
            event_data[location] = self.event_slicer[location][timestamp]
            minimum_time = max(minimum_time, event_data[location]['t'].min())
            maximum_time = min(maximum_time, event_data[location]['t'].max())
        
        for location in self._LOCATION:
            mask = np.logical_and(minimum_time <= event_data[location]['t'], event_data[location]['t'] <= maximum_time)
            for data_type in ['x', 'y', 't', 'p']:
                event_data[location][data_type] = event_data[location][data_type][mask]
        
        for location in self._LOCATION:
            event_data[location] = self.stack_function.pre_stack(event_data[location], timestamp)
        
        return event_data
    
    def _post_load_event_data(self, event_data):
        for location in self._LOCATION:
            event_data[location] = self.stack_function.post_stack(event_data[location])
        return event_data

    def collate_fn(self, batch):
        batch = self.stack_function.collate_fn(batch)
        return batch
    
def get_path_list(root):
    return [os.path.join(root, filename) for filename in sorted(os.listdir(root))]
import math
import h5py
import hdf5plugin
from typing import Dict, Tuple
from numba import jit

import numpy as np
import torch.utils.data
from .constant import EVENT_HEIGHT, EVENT_WIDTH


class EventSlicer(torch.utils.data.Dataset):
    
    def __init__(self, event_root, rectify_map_root, num_of_event, num_of_future_event=0):
        self.event_root = event_root
        self.rectify_map_root = rectify_map_root
        self.num_of_event = num_of_event
        self.num_of_future_event = num_of_future_event

        with h5py.File(event_root, 'r') as h5f:
            self.ms_to_idx = np.asarray(h5f['ms_to_idx'], dtype='int64')
            self.t_offset = int(h5f['t_offset'][()]) # [()] 是在获取 HDF5 文件中的数据集的单值数据
            self.t_final = int(h5f['events/t'][-1]) + self.t_offset
            self.min_time = int(h5f['events/t'][num_of_event]) + self.t_offset
            self.max_time = int(h5f['events/t'][-1 - num_of_future_event]) + self.t_offset
            self.total_event = len(h5f['events/t'])

        with h5py.File(rectify_map_root, 'r') as h5_rect:
            self.rectify_map = h5_rect['rectify_map'][()]

    def __len__(self):
        return 0
    
    
    def __getitem__(self, ts_end):
        event_data = self.get_events_base_number(self.num_of_event, ts_end)
        rectified_events = self.rectify_events(event_data)

        return rectified_events

    def get_events_base_number(self, number_of_event:int, t_end_us:int) -> Dict[str, np.ndarray]:
        t_start_us = t_end_us - 1000
        assert t_start_us < t_end_us
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)
        assert t_start_ms_idx is not None or t_end_ms_idx is not None

        with h5py.File(self.event_root, 'r') as h5f:
            time_array_conservative = np.asarray(h5f['events/{}'.format('t')][t_start_ms_idx:t_end_ms_idx])
        _, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_end_us_idx = t_start_ms_idx + idx_end_offset

        t_start_us_idx = max(0, t_end_us_idx - number_of_event)
        t_end_us_idx = min(self.total_event, t_end_us_idx + self.num_of_future_event)

        events = dict()
        with h5py.File(self.event_root, 'r') as h5f:
            events['t'] = np.asarray(h5f['events/{}'.format('t')][t_start_us_idx: t_end_us_idx])+self.t_offset
            for dset_str in ['p', 'x', 'y']:
                events[dset_str] = np.asarray(h5f['events/{}'.format(dset_str)][t_start_us_idx:t_end_us_idx])
                assert events[dset_str].size == events['t'].size
        
        return events
    
    def rectify_events(self, event_data):
        xy_rect = self.rectify_map[event_data['y'], event_data['x']]
        x_rect = xy_rect[:, 0]
        y_rect = xy_rect[:, 1]

        mask = (0 <= x_rect) & (x_rect < EVENT_WIDTH) & (0 <= y_rect) & (y_rect < EVENT_HEIGHT)
        return{
            'x': x_rect[mask],
            'y': y_rect[mask],
            't': event_data['t'][mask],
            'p': event_data['p'][mask],
        }

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us / 1000)
        window_end_ms = math.ceil(ts_end_us / 1000)
        return window_start_ms, window_end_ms
    
    @staticmethod
    @jit(nopython=True)
    def get_time_indices_offsets(time_array: np.ndarray, time_start_us: int, time_end_us:int) -> Tuple[int,int]:
        assert time_array.ndim == 1
        idx_start = -1
        if time_array[-1] < time_start_us:
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >=time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0
        assert time_array[idx_start] >= time_start_us
        idx_end = time_array.size
        for idx_from_end in range(time_array.size-1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break
        
        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms):
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]
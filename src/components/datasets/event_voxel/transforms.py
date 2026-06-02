import torch
import numpy as np

class ToTensor:
    def __call__(self, sample):
        if 'left' in sample.keys():
             sample['left'] = torch.from_numpy(np.transpose(sample['left'], (2, 0, 1)))
        if 'right' in sample.keys():
            sample['right'] = torch.from_numpy(np.transpose(sample['right'], (2, 0, 1)))
        return sample
    
class Padding:
    def __init__(self, img_height, img_width, no_event_value):
        self.img_height = img_height
        self.img_width = img_width
        self.no_event_value = no_event_value

    def __call__(self, sample):
        if 'left' in sample.keys():
            ori_height, ori_width = sample['left'].shape[:2]
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width
            assert top_pad >= 0 and right_pad >= 0
            sample['left'] = np.lib.pad(sample['left'],
                                    ((top_pad, 0), (0, right_pad), (0,0)),
                                    mode='constant',
                                    constant_values=self.no_event_value)
            
        if 'right' in sample.keys():
            ori_height, ori_width = sample['right'].shape[:2]
            top_pad = self.img_height - ori_height
            right_pad = self.img_width - ori_width
            assert top_pad >= 0 and right_pad >= 0
            sample['right'] = np.lib.pad(sample['right'],
                                     ((top_pad, 0), (0, right_pad), (0, 0)),
                                     mode='constant',
                                     constant_values=self.no_event_value)
        return sample

class Crop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample, offset_x, offset_y):
        start_y, end_y = offset_y, offset_y + self.crop_height
        start_x, end_x = offset_x, offset_x + self.crop_width
        
        for location in ['left', 'right']:
            if location in sample.keys():
                sample[location] = sample[location][start_y:end_y, start_x:end_x]
        
        return sample

class VerticalFlip:
    def __call__(self, sample):
        for location in ['left', 'right']:
            if location in sample.keys():
                 sample[location] = np.copy(np.flipud(sample[location]))
        
        return sample
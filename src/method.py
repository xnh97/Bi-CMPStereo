import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from utils import visualizer

@torch.no_grad()
def testall(model, data_loader, cross_modality=False, test_time=False):
    model.eval()
    pred_list = []

    for batch_data in tqdm(data_loader):
        with autocast(enabled=test_time):
            batch_data = batch_to_cuda(batch_data)

            if not cross_modality:
                pred = model(image=batch_data['event']['left'],
                           event=batch_data['event']['right'])
            else:
                pred = model(event=batch_data['event']['right'],
                               image=batch_data['image'],
                               event_voxel=batch_data['event_voxel']['right'])

            pred = pred[:,0,:,:]

            for idx in range(pred.size(0)):
                width = data_loader.dataset.WIDTH
                height = data_loader.dataset.HEIGHT
                cur_pred = pred[idx, :height, :width].cpu()
                cur_pred_dict = {
                    'file_name': str(batch_data['file_index'][idx].item()).zfill(6) + '.png',
                    'pred': visualizer.tensor_to_disparity_image(cur_pred),
                    'pred_magma': visualizer.tensor_to_disparity_magma_image(cur_pred, vmax=100),
                }
                pred_list.append(cur_pred_dict)

    return pred_list


# 将事件左右目 stack 和视差图放到 cuda 上
def batch_to_cuda(batch_data):
    # 定义了一个嵌套的辅助函数，仅在对事件处理时调用，会将字典或 tensor 放到 cuda
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cuda()
        else:
            raise NotImplementedError
        
        return batch_data
    
    for domain in ['event']:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ['left', 'right']:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(batch_data[domain][location])
            else: 
                batch_data[domain][location] = None
    if 'disparity' in batch_data.keys() and batch_data['disparity'] is not None:
        batch_data['disparity'] = batch_data['disparity'].cuda()

    if 'image' in batch_data.keys() and batch_data['image'] is not None:
        batch_data['image'] = batch_data['image'].cuda()
    return batch_data
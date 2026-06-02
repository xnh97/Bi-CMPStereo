import numpy as np
import torch
import torch.nn.functional as F


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

    
def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_gwc_volume_on_right(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, :-i] = groupwise_correlation(refimg_fea[:, :, :, :-i], targetimg_fea[:, :, :, i:],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def disparity_regression(prob):
    assert len(prob.shape) == 4
    maxdisp = prob.size(1)
    disp_values = torch.arange(0, maxdisp, dtype=prob.dtype, device=prob.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(prob * disp_values, 1, keepdim=True)



def coords_grid_v0(batch, ht, wd):
    x_grid, y_grid = np.meshgrid(np.arange(wd), np.arange(ht))
    x_grid = torch.tensor(x_grid, dtype=torch.float32)
    y_grid = torch.tensor(y_grid, dtype=torch.float32)
    coords = torch.stack([x_grid, y_grid], dim=0)
    coords = coords.repeat(batch, *([1]*coords.dim()))
    return coords

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    H, W = img.shape[-2:]
    coords[...,0] = coords[:,:,:,0]/(W-1)*2 - 1
    coords[...,1] = coords[:,:,:,1]/(H-1)*2 - 1
    coords = coords.contiguous()
    if img.dtype is torch.float16:
        coords = coords.half()
    img = F.grid_sample(img, coords, mode=mode, padding_mode="zeros", align_corners=True)
    if mask:
        mask = (
            (coords[:, :, :, 0:1] < 0)
            | (coords[:, :, :, 0:1] > W - 1)
            | (coords[:, :, :, 1:2] < 0)
            | (coords[:, :, :, 1:2] > H - 1)
        )
        mask = torch.logical_not(mask)
        return img, mask.to(torch.float32)
    return img
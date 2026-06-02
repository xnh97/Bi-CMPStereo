import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractorU_recons import feature_encoder, feature_decoder, recons_decoder

from .extractorR import MultiBasicEncoder

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


# 输入左目 disp，对右目特征图 warp，或输入左目 -disp，对左目 warp
def warp_fmap(disp, fmap):
    assert fmap.shape[-1] == disp.shape[-1]
    
    # 如果输入一维，则为回归的左目视差，使用时取反，输入为二维，则为 crestereo 中的左目视差之反，不需取反
    N, C, H, W = disp.shape
    if C == 1:
        flow_y = torch.zeros([N, 1, H, W], dtype=disp.dtype).to(disp.device)
        # C=1 则为 回归 估计的视差，需要取反
        flow_x = -disp
        flow = torch.cat([flow_x, flow_y], dim=1)
    elif C == 2:
        # C=2 则为 crestereo 估计的视差，本身为负，不需要取反
        flow = disp

    coords = coords_grid(fmap.shape[0], fmap.shape[2], fmap.shape[3], fmap.device)
    coords = coords + flow # [b  2  h  w]
    coords = coords.permute(0, 2, 3, 1) # [b  h  w  2]
    warped_fmap = bilinear_sampler(fmap, coords)
    return warped_fmap


class DLNR(nn.Module):
    def __init__(self, max_disp=128):
        super().__init__()
        ###
        
        self.f_net_V = MultiBasicEncoder(output_dim=256, norm_fn="instance", dropout=0)
        self.f_net_E = feature_encoder(input_dim=5)
        self.f_net_F = feature_encoder(input_dim=3)
        self.f_net_D = feature_decoder()
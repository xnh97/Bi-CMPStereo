import torch
import torch.nn as nn
import torch.nn.functional as F
from .extractorU_recons import feature_encoder, feature_decoder
from .extractorR import MultiBasicEncoder
from .geometry import Encoding_Volume
from .bigeometry import Encoding_Volume as VOlume

from .update import BasicUpdateBlock
from .submodule import *

from ..utils import build_gwc_volume, coords_grid, bilinear_sampler

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


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.attention_block = attention_block(channels_3d=in_channels * 4, num_heads=16, block=(4, 4, 4))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.attention_block(conv4)
        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return conv6


class DLNR(nn.Module):
    def __init__(self, max_disp=128):
        super().__init__()
        ###
        self.max_disp = max_disp
        self.corr_radius = 4
        
        self.f_net_V = MultiBasicEncoder(output_dim=256, norm_fn="instance", dropout=0)
        self.update_block_stair1 = BasicUpdateBlock(hidden_dim=128, cor_planes=2*2*8*(2*self.corr_radius+1), mask_size=4)
        self.update_block_stair2 = BasicUpdateBlock(hidden_dim=128, cor_planes=3*8*(2*self.corr_radius+1), mask_size=4)

        self.dres1_att_ = nn.ModuleList()


        self.dres0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))        
        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)     
        
        self.classif = nn.Sequential(convbn_3d(32, 16, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(16, 8, kernel_size=3, padding=1, stride=1, bias=False))

        for _ in range(2):
            self.dres1_att_.append(nn.Sequential(convbn_3d(8, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 3, 1, 1)))
    

    def forward(self, image4, image8, event4, event8, image_V):
        fmap_V, fmap_V8, _ = self.f_net_V(image_V)

        predictions = []

        disp_volume8 = []
        for i in range(2):
            image_feat = image8[i].to(torch.float32)
            event_feat = event8[i].to(torch.float32)
            disp_volume8.append(build_gwc_volume(image_feat, event_feat, self.max_disp//8, 8))

        geo_fn8 = Encoding_Volume(disp_volume8, radius=self.corr_radius, num_levels=2)
        
        fmap_V8 = fmap_V8.to(torch.float32)
        net_dw8, inp_dw8 = torch.split(fmap_V8, [128, 128], dim=1)
        net_dw8 = F.tanh(net_dw8)
        inp_dw8 = F.relu(inp_dw8)

        b, _, h, w = image8[0].shape
        disp8 = torch.zeros([b, 1, h, w], dtype=torch.float32, device=image8[0].device)
            
        for _ in range(6):
            disp8 = disp8.detach()
            init_corr = geo_fn8(disp8)
            net_dw8, mask_feat_4, delta_disp = self.update_block_stair1(net_dw8, inp_dw8, init_corr, disp8)
            disp8 = disp8 + delta_disp

            disp = self.convex_upsample(disp8, mask_feat_4, rate=4)

            disp_up = 2*F.interpolate(disp,
                                       size=(2*disp.shape[2], 2*disp.shape[3]),
                                       mode='bilinear',
                                       align_corners=True,)
            predictions.append(disp_up)

        
        fmap_V = fmap_V.to(torch.float32)
        net, inp = torch.split(fmap_V, [128, 128], dim=1)
        net = F.tanh(net)
        inp = F.relu(inp)

        disp_volume4 = []
        cost_volume4 = []
        for i in range(2):
            image_feat = image4[i].to(torch.float32)
            event_feat = event4[i].to(torch.float32)
            cost_volume = build_gwc_volume(image_feat, event_feat, self.max_disp//4, 8)
            cost_volume = self.dres1_att_[i](cost_volume)
            cost_volume4.append(cost_volume)
        
        cost_volume_fused = torch.cat(cost_volume4, dim=1)
        cost0 = self.dres0(cost_volume_fused)
        cost0 = self.dres1(cost0) + cost0
        out1 = self.dres2(cost0)
        out2 = self.dres3(out1)
        
        att_weights = self.classif(out2)
        disp_volume4.append(att_weights)

        geo_fn4 = VOlume(att_weights, radius=self.corr_radius, num_levels=3)

        scale = event4[0].shape[2]/disp.shape[2]
        disp4 = scale * F.interpolate(disp,
                                    size=(event4[0].shape[2], event4[0].shape[3]),
                                    mode='bilinear',
                                    align_corners=True,)
            
        for _ in range(6):
            disp4 = disp4.detach()
            init_corr = geo_fn4(disp4)
            net, mask_feat_4, delta_disp = self.update_block_stair2(net, inp, init_corr, disp4)
            disp4 = disp4 + delta_disp

            disp = self.convex_upsample(disp4, mask_feat_4, rate=4)
            predictions.append(disp)

        
        return predictions
    
    def zero_init(self, fmap):
        N, _, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat([_x, _y], dim=1).to(fmap.device)
        return zero_flow

    def convex_upsample(self, flow, mask, rate=4):
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 1, rate*H, rate*W)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..othersCFF.feature_extractor import ResNetFeature
from .submodule import *
from .extractor import BasicEncoder
from .corr import AGCL
from .update import BasicUpdateBlock

from .utils import build_gwc_volume, disparity_regression

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


class DLNR(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()
        ###
        self.max_disp = max_disp
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.extractor_event = ResNetFeature(in_channels=1, base_channels=16)
        self.extractor_image = ResNetFeature(in_channels=3, base_channels=16)
        self.deconv16_8 = Conv2x(256, 128, deconv=True, concat=True)
        self.conv8 = BasicConv(128*2, 128*2, kernel_size=3, stride=1, padding=1)
        self.deconv8_4 = Conv2x(128*2, 64, deconv=True, concat=True)
        self.conv4 = BasicConv(64*2, 64*2, kernel_size=3, stride=1, padding=1)
        self.f_net_V = BasicEncoder(output_dim=128, norm_fn="instance", dropout=0, input_dim=3)
        self.update_block = BasicUpdateBlock(hidden_dim=64, cor_planes=4*9, mask_size=4)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def upsample_flow(self, flow, mask):
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)
    

    def forward(self, image, event):
        with autocast(enabled=True):
            ###
            image4, image8, image16 = self.extractor_image(image)
            event4, event8, event16 = self.extractor_event(event)

            # [n, group=8, disp, h, w] 表示 8 组特征下，每个像素在某个视差下和对应位置像素之间的相关性
            disp_volume16 = build_gwc_volume(image16, event16, self.max_disp//16, 8)
            # 3*1*1 卷积将多头 cost 聚合为 1：[b,  disp,  h,  w]，softmax 计算每个像素上各视差的概率：[b,  disp,  h,  w]
            disp_volume16 = self.classifier(disp_volume16)
            prob_volume16 = F.softmax(disp_volume16.squeeze(1), dim=1)
            agg_disp16 = disparity_regression(prob_volume16)

            image8 = self.deconv16_8(image16, image8)
            image8 = self.conv8(image8)
            event8 = self.deconv16_8(event16, event8)
            event8 = self.conv8(event8)
            disp_volume8 = build_gwc_volume(image8, event8, self.max_disp//8, 8)
            disp_volume8 = self.classifier(disp_volume8)
            prob_volume8 = F.softmax(disp_volume8.squeeze(1), dim=1)
            agg_disp8 = disparity_regression(prob_volume8)
            
            image4 = self.deconv8_4(image8, image4)
            image4 = self.conv4(image4)
            event4 = self.deconv8_4(event8, event4)
            event4 = self.conv4(event4)

            disp_volume4 = build_gwc_volume(image4, event4, self.max_disp//4, 8)
            disp_volume4 = self.classifier(disp_volume4)
            prob_volume4 = F.softmax(disp_volume4.squeeze(1), dim=1)
            agg_disp4 = disparity_regression(prob_volume4)

            corr_fn = AGCL(image4, event4)
            fmap_V = self.f_net_V(image)
            net, inp = torch.split(fmap_V, [64, 64], dim=1)
            net = F.tanh(net)
            inp = F.relu(inp)

            predictions = []
            N, _, H, W = agg_disp4.shape
            flow_y = torch.zeros([N, 1, H, W], dtype=agg_disp4.dtype).to(agg_disp4.device)
            flow = torch.cat([agg_disp4, flow_y], dim=1)
            flow_up = []

            scale = image4.shape[2]/flow.shape[2]
            flow = -scale * F.interpolate(flow,
                                    size=(image4.shape[2], image4.shape[3]),
                                   mode='bilinear',
                                   align_corners=True,)
            
            for itr in range(5):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = False
            
                flow = flow.detach()
                out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)
                flow = flow + delta_flow

                flow_up = -self.convex_upsample(flow, up_mask, rate=4)
                predictions.append(flow_up)

        return [agg_disp16, agg_disp8, agg_disp4], predictions


    def convex_upsample(self, flow, mask, rate=4):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        # print(flow.shape, mask.shape, rate)
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate*H, rate*W)
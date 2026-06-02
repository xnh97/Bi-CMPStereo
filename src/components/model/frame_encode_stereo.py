
import torch
import torch.nn as nn

import sys
sys.path.append('/data/hjw/events')  # 添加项目根目录到路径
from Bi_CMPStereo.imgCMPStereo.src.components.model.Rstereo.final.EF_DoubleStairs import DLNR

from .reconstration import ReConstrationNet


class EventFrameEncodeStereoMatchingNetwork(nn.Module):

    def __init__(self, in_channels=10):
        super(EventFrameEncodeStereoMatchingNetwork, self).__init__()
        self.reconstration_net = ReConstrationNet(in_channels=5)
        self.stereo_matching_net = DLNR()
        

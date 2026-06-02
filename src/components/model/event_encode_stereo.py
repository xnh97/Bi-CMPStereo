import os
import torch.nn as nn

import sys
sys.path.append('/data/hjw/events')
from Bi_CMPStereo.evCMPStereo.src.components.model.Rstereo.final.EF_DoubleStairs import DLNR

from .reconstration import ReConstrationNet
from .concentration import ConcentrationNet

class EventFrameEncodeStereoMatchingNetwork(nn.Module):

    def __init__(self, in_channels=3):
        super(EventFrameEncodeStereoMatchingNetwork, self).__init__()
        self.reconstration_net = ReConstrationNet(in_channels=3)
        self.concentration_net = ConcentrationNet(in_channels=10)
        for param in self.concentration_net.parameters():
            param.requires_grad = False
        self.stereo_matching_net = DLNR()

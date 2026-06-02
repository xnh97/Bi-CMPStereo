import os
from einops import rearrange
import torch
import torch.nn as nn
from .frame_encode_stereo import EventFrameEncodeStereoMatchingNetwork
from .event_encode_stereo import EventFrameEncodeStereoMatchingNetwork as EventEncodeStereoMatchingNetwork

from .Rstereo.DualV.EF_DoubleStairsFusion import DLNR


class EventFrameStereoNetwork(nn.Module):

    def __init__(self, concentration_net=None, disparity_estimator=None, image_location='left'):
        super(EventFrameStereoNetwork, self).__init__()

        self.image_modality_feat = EventFrameEncodeStereoMatchingNetwork()
        self.event_modality_feat = EventEncodeStereoMatchingNetwork()

        # frozen
        for param in self.image_modality_feat.parameters():
            param.requires_grad = False
            
        for param in self.event_modality_feat.parameters():
            param.requires_grad = False

        self.stereo_matching_net = DLNR()
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.recons_loss = torch.nn.L1Loss()


    def forward(self, event, image, event_voxel):
        image_tensor = image.clone().float()
        event_voxel_stack = event_voxel.clone()
        event_stack = event.clone()
        event_stack = rearrange(event_stack, 'b c h w t s -> b (c s t) h w').contiguous()

        if True:
            reconstration_event_stack = self.image_modality_feat.reconstration_net(event_voxel_stack)
            image_en_feat = self.image_modality_feat.stereo_matching_net.f_net_F(image_tensor)
            event_en_feat  = self.image_modality_feat.stereo_matching_net.f_net_E(reconstration_event_stack)
            image4, image8 = self.image_modality_feat.stereo_matching_net.f_net_D(image_en_feat[2:])
            event4, event8 = self.image_modality_feat.stereo_matching_net.f_net_D(event_en_feat[2:])
        
        if True:
            concentration_event_stack = self.event_modality_feat.concentration_net(event_stack)
            reconstration_image = self.event_modality_feat.reconstration_net(image_tensor)
            image_en_feat_ = self.event_modality_feat.stereo_matching_net.f_net_F(reconstration_image)
            event_en_feat_  = self.event_modality_feat.stereo_matching_net.f_net_E(concentration_event_stack)
            image4_, image8_ = self.event_modality_feat.stereo_matching_net.f_net_D(image_en_feat_[2:])
            event4_, event8_ = self.event_modality_feat.stereo_matching_net.f_net_D(event_en_feat_[2:])

        fmap_4f = [image4, image4_]
        fmap_4e = [event4, event4_]
        fmap_8f = [image8, image8_]
        fmap_8e = [event8, event8_]

        pred_disparity = self.stereo_matching_net(image4=fmap_4f, image8=fmap_8f, 
                                                   event4=fmap_4e, event8=fmap_8e, 
                                                   image_V=image_tensor)

        
        return pred_disparity[-1]

import os
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Rstereo.crestereo import CREStereoIo:
from .Rstereo.crestereo_seperate_RF import CREStereo
# from .Rstereo.FusionCrestereo import FusionCREStereo
from .concentration import ConcentrationNet, ImageReconstruct
from .othersCFF.stereo_matching import StereoMatchingNetwork
from torchvision.utils import save_image
import numpy as np
import scipy.stats as st

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis = 0)
    return out_filter

class EventFrameEncodeStereoMatchingNetwork(nn.Module):
    _LOCATION = ['left', 'right']

    def __init__(self, concentration_net=None, disparity_estimator=None, image_location='left'):
        super(EventFrameEncodeStereoMatchingNetwork, self).__init__()
        self.concentration_net = ConcentrationNet(**concentration_net.PARAMS)
        self.stereo_matching_net = CREStereo(**disparity_estimator.PARAMS, cross_modality=True)
        # self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.PARAMS)
        self.image_recon_net = ImageReconstruct()

        # SmoothL1Loss 在范围（通常是[-1,1]）内使用平方误差，超出时用绝对误差
        self.criterion = nn.SmoothL1Loss(reduction='none')

        # 图片位置
        self.image_location = image_location
        # 事件位置
        self.event_location = self._LOCATION.copy()
        self.event_location.remove(image_location)
        self.event_location = self.event_location[0]
    
    def blur(self, x, kernel = 21, channels = 3, stride = 1, padding = 1):
        # 返回一个 Numpy 类型的高斯滤波器，然后将其转换为 Tensor
        kernel_var = torch.from_numpy(gauss_kernel(kernel, 3, channels)).to('cuda').float()
        # 执行卷积操作，将创建的高斯滤波器应用于输入图像，实现高斯模糊
        return F.conv2d(x, kernel_var, stride = stride, padding = padding, groups = channels)
    
    def forward(self, event, image, gt_disparity=None):
        event_stack = event.clone()
        event_stack = rearrange(event_stack, 'b c h w t s -> b (c s t) h w')
        # 使用 U-net 输出各 stack 的得分，之后对各 stack 加权融合，重建输出事件帧 [b 1 h w]
        concentration_event_stack = self.concentration_net(event_stack)
        
        image_tensor = image.clone().float()
        recons_image = self.image_recon_net(image_tensor)
    
        pred_disparity = self.stereo_matching_net(
                recons_image,
                concentration_event_stack,
                image_tensor
            )
        
        # # for i in range(concentration_event_stack.shape[0]):
        # #     single_image = concentration_event_stack[i].unsqueeze(0)  # 增加一个维度，使形状变为 [1, 1, h, w]
        # #     save_path = os.path.join('/home/xnh/event-stereo/EF-CMS/save/weights', f'image_{i}.png')  # 定义保存路径
        # #     save_image(single_image, save_path)  # 保存图片
        # # for i in range(recons_image.shape[0]):
        # #     single_image = recons_image[i].unsqueeze(0)  # 增加一个维度，使形状变为 [1, 1, h, w]
        # #     save_path = os.path.join('/home/xnh/event-stereo/EF-CMS/save/weights', f'image_{i}a.png')  # 定义保存路径
        # #     save_image(single_image, save_path)

        loss_disp = None
        gt_disparity = gt_disparity.unsqueeze(dim=1)
        gt_disparity = torch.cat([gt_disparity, gt_disparity*0], dim=1)
        if gt_disparity is not None:
            loss_disp, loss_scale = self.sequence_loss(pred_disparity, gt_disparity, gamma=0.8)

        loss = loss_disp.mean()

        # loss_disp = None
        # if gt_disparity is not None:
        #     loss_disp = self._cal_loss(pred_disparity, gt_disparity)
        
        return pred_disparity[-1], loss, loss_scale
    
    # 由预测的 5 尺度 (1/12  1/6  1/3  1  1) 视差计算 loss
    # 输入视差 list = [b*h/12*w/12     b*h/6*w/6     b*h/3*w/3     b*h*w     b*h*w]
    # 将各尺度视差对齐到原尺寸，再分别计算各尺度 loss，再加权相加得到最终 loss = [b*h*w]
    def _cal_loss(self, pred_disparity_pyramid, gt_disparity):
        # 金字塔视差不同分辨率计算 loss 的权重不同（共 5 层视差，分辨率为 1/12  1/6  1/3  1  1）
        pyramid_weight = [1/3, 2/3, 1.0, 1.0, 1.0]
        loss = 0.0
        mask = gt_disparity > 0

        # 分别对各层预测视差，对齐到原尺寸后加权计算 loss
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            # 金字塔视差中和原始尺寸不同的预测视差，通过双线性插值恢复到原始视差，视差值也相应缩放
            if pred_disp.size(-1) != gt_disparity.size(-1):
                # 预测的视差扩维 [b h w]——>[b 1 h w]，以适配 interpolate 函数的输入要求
                pred_disp = pred_disp.unsqueeze(1)
                # 视差恢复到真值尺寸，视差值也相应缩放
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                              gt_disparity.size(-1)/pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)    # [b h w]

            # 计算各尺度视差图的损失（仅视差真值存在的像素），加权求和得最终损失
            # reduction=none 返回和输入同 shape 的 loss，pred_disp[mask] 将 [b h w] 变为维度为 1 的向量，因此输出也是向量
            cur_loss = self.criterion(pred_disp[mask], gt_disparity[mask])
            loss += weight * cur_loss
            
        return loss
    
    # 由每次迭代输出的视差计算 loss，每次输出的视差已对齐到原始尺度，输入视差 list [b*h*w]
    # 计算每次迭代的平均像素视差 loss，再加权相加得到总 loss，loss 权重随迭代次数增大
    def sequence_loss(self, flow_preds, flow_gt, gamma=0.8):
        # 迭代估计得到的视差数，即迭代总次数
        n_predictions = len(flow_preds)
        flow_loss = 0.0
        mask = flow_gt > 0
        loss_scale = []
        # 对每个迭代视差估计，计算像素视差平均 loss，再加权相加，得到总 loss
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions-i-1)
            i_loss = torch.abs(flow_preds[i] - flow_gt)
            loss_scale.append(i_loss[mask].mean())
            flow_loss += i_weight *  i_loss[mask].mean()
            # 该方法求的 mean 对整个平面，不是仅对 mask=True 的像素，导致 loss 过小
            # flow_loss += i_weight * (mask.unsqueeze(dim=1) * i_loss).mean()
        return flow_loss, loss_scale


    # 网络模型初始化后，为不同参数配置不同的学习率，具体地：
    # 网络中的 ['offset_conv.weight', 'offset_conv.bias'] 特殊参数学习率设置为输入的 0.1，其他的参数正常，
    # 返回 list=[{'params': 特殊参数, 'lr': 0.1 * 学习率}, {'params': 其他参数, 'lr': 学习率}]
    def get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False
        
        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True
        
        # named_parameters()返回模型中所有参数，每个=(name参数名称, paramete参数对象)的元组
        # filter() 接收 1 个函数和 1 个元组，返回 1 个新元组，包含原元组中使函数返回 True 的元素
        # 即 specific 为网络中所有 offset_conv 的参数，base 为所有其他参数
        specific_params = list(filter(filter_specific_params, self.named_parameters()))
        base_params = list(filter(filter_base_params, self.named_parameters()))
        
        # 提取具体参数
        specific_params = [kv[1] for kv in specific_params]
        base_params = [kv[1] for kv in base_params]

        # 对 specific 参数配置 0.1 的学习率，base 则使用输入的学习率
        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]

        return params_group

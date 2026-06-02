import torch
import torch.nn as nn
import torch.nn.functional as F

from .extractor import BasicEncoder
from .corr import AGCL
from .update import BasicUpdateBlock
from .attention.position_encoding import PositionEncodingSine
from .attention.transformer import LocalFeatureTransformer
from .fusion.RFNet import RFNet
from .fusion.RFGru import RF

class CREStereo(nn.Module):
    def __init__(self, base_channels=256, max_disp=192, mixed_precision=False, test_mode=False, iters=10, cross_modality=False):
        super(CREStereo, self).__init__()
        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode
        self.iters = iters

        self.base_channels = base_channels
        self.hidden_dim = base_channels//2   # 128
        self.context_dim = base_channels//2   # 128
        self.drop_out = 0
        self.search_num = 9

        # 双目特征提取网络
        self.f_net_V = BasicEncoder(output_dim=base_channels, norm_fn="instance", dropout=self.drop_out, input_dim=3)
        self.f_net = BasicEncoder(output_dim=base_channels, norm_fn="instance", dropout=self.drop_out, input_dim=1)
        # self.f_net2 = BasicEncoder(output_dim=base_channels, norm_fn="instance", dropout=self.drop_out, input_dim=1)
        # GRU 循环更新网络，各尺度使用相同的循环网络进行状态更新
        self.update_block = BasicUpdateBlock(hidden_dim=self.hidden_dim, cor_planes=4*9, mask_size=4)
        self.update_block1 = BasicUpdateBlock(hidden_dim=self.hidden_dim, cor_planes=4*9, mask_size=4)
        self.recurrent_fusion = RFNet(in_channels=base_channels, out_channels=base_channels, flow_channels=2, 
                                      base_1_channels=256, base_2_channels=64, stage=3)
        self.recurrent_fusion2 = RFNet(in_channels=base_channels, out_channels=base_channels, flow_channels=2, 
                                      base_1_channels=256, base_2_channels=64, stage=3)
        # self.RF = RF(in_channels=base_channels, base_1_channels=256, base_2_channels=64, stage=3)
        self.norm1 = torch.nn.InstanceNorm2d(base_channels)
        self.norm2 = torch.nn.InstanceNorm2d(base_channels)
        self.conv_i = nn.Conv2d(base_channels, base_channels, kernel_size=1, stride=1, padding=0)
        self.conv_e = nn.Conv2d(base_channels, base_channels, kernel_size=1, stride=1, padding=0)

        # 对左目 1/8 和 1/16 特征图《3*3 卷积》提取额外偏置
        self.conv_offset_8 = nn.Conv2d(base_channels, self.search_num*2, kernel_size=3, stride=1, padding=1)
        self.conv_offset_16 = nn.Conv2d(base_channels, self.search_num*2, kernel_size=3, stride=1, padding=1)

        # self/cross 注意力机制
        self.self_att_fn = LocalFeatureTransformer(
            d_model=base_channels, nhead=8, layer_name="self", attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=base_channels, nhead=8, layer_name="cross", attention="linear"
        )

    def forward(self, image1, image2, image_V):
        # ResNet 提取 1/4 双目特征：[b  3  h  w]——>[b  256  h/4  w/4]
        # 特征图用于双目相关性计算，以及直接作为 GRU 初始隐藏状态和正则化输入
        fmap1, fmap2 = self.f_net([image1, image2])  # 输入 list，返回 tuple
        fmapi = self.conv_i(fmap1).to(torch.float32)
        fmape = self.conv_i(fmap2).to(torch.float32)
        fmapi_dw8 = F.avg_pool2d(fmapi, (2,2))
        fmape_dw8 = F.avg_pool2d(fmape, (2,2))
        fmap_V = self.f_net_V(image_V)

        fmap1 = fmap1.to(torch.float32)
        fmap2 = fmap2.to(torch.float32)

        fmap_V = fmap_V.to(torch.float32)
        assert self.hidden_dim+self.context_dim == fmap1.size(1)

        # 将左目 1/4 特征图按通道分为 net (1-128) 和 inp (129-256)
        net, inp = torch.split(fmap_V, [self.hidden_dim, self.context_dim], dim=1)
        # net 使用 tanh 输出 1-128 通道，作为 1/4 GRU 隐藏状态的初值，各尺度下随迭代更新
        net = F.tanh(net)
        # inp 使用 relu 输出 129-256 通道，作为 GRU 的正则化输入，迭代过程中保持不变，辅助视差/双目相关性更新
        inp = F.relu(inp)
        
        # 1/4 特征图 1/2 平均池化下采样，得到 1/8 特征图=[b  256  h/8  w/8]
        fmap1_dw8 = F.avg_pool2d(fmap1, (2,2))
        fmap2_dw8 = F.avg_pool2d(fmap2, (2,2))
        # 1/8 尺度下的 GRU 正则化输入和隐藏状态初值，也由 1/4 的下采样生成
        net_dw8 = F.avg_pool2d(net, (2,2))
        net8 = F.avg_pool2d(net, (2,2))
        inp_dw8 = F.avg_pool2d(inp, (2,2))
        #《3*3 卷积 + sigmoid》处理左目 1/8 特征图 = [b  18  h/8  w/8]，将 sigmoid 的 [0  1] 转换为 [-1  1]
        # 表示 1/8 匹配过程中，每个像素的 9 个额外搜索偏置 (x,y)，对 3*3 和 1*9 搜索模式，搜索偏置一样
        offset_dw8 = self.conv_offset_8(fmap1_dw8)
        offset_dw8 = (F.sigmoid(offset_dw8) - 0.5)*2.0

        # 1/4 特征图 1/4 下采样，得到 1/16 特征图=[b  256  h/16  w/16]
        fmap1_dw16 = F.avg_pool2d(fmap1, (4,4))
        fmap2_dw16 = F.avg_pool2d(fmap2, (4,4))
        # 1/16 尺度下的 GRU 正则化输入和隐藏状态初值，也由 1/4 的下采样生成
        net_dw16 = F.avg_pool2d(net, (4 ,4))
        net16 = F.avg_pool2d(net, (4 ,4))
        inp_dw16 = F.avg_pool2d(inp, (4,4))
        #《3*3 卷积 + sigmoid》处理左目 1/16 特征图 = [b  18  h/16  w/16]，得到各像素的搜索额外偏置
        # 1/4 尺度下不需要额外搜索偏置
        offset_dw16 = self.conv_offset_16(fmap1_dw16)
        offset_dw16 = (F.sigmoid(offset_dw16)-0.5)*2.0

        # 生成 1/16 特征图的位置编码函数
        pos_encoding_fn_small = PositionEncodingSine(
            d_model=self.base_channels, max_shape=(image1.shape[2]//16, image2.shape[3]//16)
        )
        # 左目 1/16 特征图添加位置编码
        x_tmp = pos_encoding_fn_small(fmap1_dw16)
        # 2 维特征图平铺为 1 维 [n, c, h, w]——>[n, h*w, c]
        fmap1_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])

        # 右目 1/16 特征图添加位置编码，平铺 [n, c, h, w]——>[n, h*w, c]
        x_tmp = pos_encoding_fn_small(fmap2_dw16)
        fmap2_dw16 = x_tmp.permute(0, 2, 3, 1).reshape(x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1])

        # self 注意力机制分别处理双目 1/16 特征图
        fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
        # 修改多头输出，恢复 shape
        fmap1_dw16, fmap2_dw16 = [
                        x.reshape(x.shape[0], image1.shape[2] // 16, -1, x.shape[2]).permute(0, 3, 1, 2)
                        for x in [fmap1_dw16, fmap2_dw16]
                    ]

        # 3 个尺度下的双目相关计算迭代函数，传入对应尺度的双目图进行初始化
        corr_fn = AGCL(fmap1, fmap2)
        corr_fn_dw8 = AGCL(fmap1_dw8, fmap2_dw8)
        corr_fn_dw16 = AGCL(fmap1_dw16, fmap2_dw16, att=self.cross_att_fn)

        # 迭代视差估计，1/16：5次，1/8：5次，1/4：10 次
        predictions = []
        flow = []
        flow_up = []
        # 1/16 尺度下 5 次视差迭代，每次迭代的结果都存储计算 loss
        # 1/16 尺度的初值光流 [b  2  h  w]=0，最后一次迭代结果上采样作为 1/8 尺度视差初值
        flow_dw16 = self.zero_init(fmap1_dw16)

        for itr in range(self.iters//4):  # 循环 10//2=5 次
            # itr = 0/2/4 时搜索区域沿极线方向 x=-4~4 y=0，itr =1/3 时沿极线+竖直方向 x=y=-1~1
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True
            
            # 每次迭代，将之前的模型固定，因此在训练时仅对这次迭代中的新网络进行优化，
            # 即一次迭代对应的训练，期望在该尺度下，基于某初始视差和之前的隐藏状态，
            # 可以通过《一次 GRU 循环网络 + convex 上采样网络》恢复视差，
            # 此外，用于生成特征图//inp//net 的特征提取网络和生成额外搜索偏置的网络也会进行一次训练
            flow_dw16 = flow_dw16.detach()

            # 在初始视差和额外搜索偏置下，计算多头双目相关性=[b, 9*4, h, w]，搜索模式为 3*3 或 1*9
            # 表示右目每个像素的 9 个搜索像素，和左目对应像素，每组特征的双目相关性
            out_corrs = corr_fn_dw16(flow=flow_dw16, 
                                                          extra_offset=offset_dw16,
                                                          small_patch=small_patch)
            
            #《初始视差 + 该视差的 9 个搜索偏置下的双目相关性 + 正则化项》作为输入，
            # 结合 GRU 对隐藏状态更新，用于下一次迭代，并从隐藏状态生成视差更新量和上采样 mask
            net_dw16, up_mask, delta_flow = self.update_block(net_dw16, inp_dw16, out_corrs, flow_dw16)      
            
            # 更新视差，用于下次迭代
            flow_dw16 = flow_dw16 + delta_flow
            
            # 1/16 视差 4 倍上采样，用于存储计算 loss (最后一次的作为 1/8 特征图的初始视差)
            flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
            # 再 4 倍上采样得到原尺度视差，存储计算 loss，不进入迭代
            flow_up = -4*F.interpolate(flow,
                                       size=(4*flow.shape[2], 4*flow.shape[3]),
                                       mode="bilinear",
                                       align_corners=True,)

            predictions.append(flow_up)
        
        for itr in range(self.iters//4):  # 循环 10//2=5 次
            # itr = 0/2/4 时搜索区域沿极线方向 x=-4~4 y=0，itr =1/3 时沿极线+竖直方向 x=y=-1~1
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True
            
            # 每次迭代，将之前的模型固定，因此在训练时仅对这次迭代中的新网络进行优化，
            # 即一次迭代对应的训练，期望在该尺度下，基于某初始视差和之前的隐藏状态，
            # 可以通过《一次 GRU 循环网络 + convex 上采样网络》恢复视差，
            # 此外，用于生成特征图//inp//net 的特征提取网络和生成额外搜索偏置的网络也会进行一次训练
            flow_dw16 = flow_dw16.detach()
            
            if itr == 0:
                x, _ = self.recurrent_fusion(last_state=None, up_left=fmapi_dw8, up_right=fmape_dw8, 
                                                           flow=flow_dw16, init=True, ini_state=fmap1_dw16)
                fmap1_dw16_ = fmap1_dw16 + self.norm1(x)

                y, _ = self.recurrent_fusion(last_state=None, up_left=fmapi_dw8, up_right=fmape_dw8, 
                                                       flow=flow_dw16, init=True, ini_state=fmap2_dw16)
                fmap2_dw16_ = fmap2_dw16 + self.norm1(y)

                corr_fn_dw16_ = AGCL(fmap1_dw16_, fmap2_dw16_, att=self.cross_att_fn)
                

            # 在初始视差和额外搜索偏置下，计算多头双目相关性=[b, 9*4, h, w]，搜索模式为 3*3 或 1*9
            # 表示右目每个像素的 9 个搜索像素，和左目对应像素，每组特征的双目相关性
            out_corrs = corr_fn_dw16_(flow=flow_dw16, 
                                                          extra_offset=offset_dw16,
                                                          small_patch=small_patch)
            
            #《初始视差 + 该视差的 9 个搜索偏置下的双目相关性 + 正则化项》作为输入，
            # 结合 GRU 对隐藏状态更新，用于下一次迭代，并从隐藏状态生成视差更新量和上采样 mask
            net16, up_mask, delta_flow = self.update_block1(net16, inp_dw16, out_corrs, flow_dw16)
            
            # 更新视差，用于下次迭代
            flow_dw16 = flow_dw16 + delta_flow
            
            # 1/16 视差 4 倍上采样，用于存储计算 loss (最后一次的作为 1/8 特征图的初始视差)
            flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
            # 再 4 倍上采样得到原尺度视差，存储计算 loss，不进入迭代
            flow_up = -4*F.interpolate(flow,
                                       size=(4*flow.shape[2], 4*flow.shape[3]),
                                       mode="bilinear",
                                       align_corners=True,)
            
            predictions.append(flow_up)

        # 1/8 迭代估计的初始视差，由 1/16 的视差 4 倍 convex 上采样，再 1/2 下采样获得
        scale = fmap1_dw8.shape[2]/flow.shape[2]
        flow_dw8 = scale*F.interpolate(flow,
                                   size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                                   mode="bilinear",
                                   align_corners=True,)
                                   
        # 1/8 尺度下 5 次视差迭代，每次迭代的结果都存储计算 loss
        for itr in range(self.iters//4):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True
            
            flow_dw8 = flow_dw8.detach()

            out_corrs = corr_fn_dw8(flow=flow_dw8, 
                                                          extra_offset=offset_dw8, 
                                                          small_patch=small_patch)
            
            net_dw8, up_mask, delta_flow = self.update_block(net_dw8, inp_dw8, out_corrs, flow_dw8)        
                    
            flow_dw8 = flow_dw8 + delta_flow

            flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
            flow_up = -2*F.interpolate(flow,
                                       size=(2*flow.shape[2], 2*flow.shape[3]),
                                       mode='bilinear',
                                       align_corners=True,)

            predictions.append(flow_up)
            
        for itr in range(self.iters//4):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True
            
            flow_dw8 = flow_dw8.detach()
            
            if itr == 0:
                x, _ = self.recurrent_fusion(last_state=None, up_left=fmapi, up_right=fmape,
                                                           flow=flow_dw8.detach(), init=True, ini_state=fmap1_dw8)
                fmap1_dw8_ = fmap1_dw8 + self.norm1(x)

                y, _ = self.recurrent_fusion(last_state=None, up_left=fmapi, up_right=fmape,
                                                           flow=flow_dw8.detach(), init=True, ini_state=fmap2_dw8)
                fmap2_dw8_ = fmap2_dw8 + self.norm1(y)

                corr_fn_dw8_ = AGCL(fmap1_dw8_, fmap2_dw8_)


            out_corrs = corr_fn_dw8_(flow=flow_dw8, 
                                                          extra_offset=offset_dw8, 
                                                          small_patch=small_patch)
            
            net8, up_mask, delta_flow = self.update_block1(net8, inp_dw8, out_corrs, flow_dw8)        
                    
            flow_dw8 = flow_dw8 + delta_flow

            flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
            flow_up = -2*F.interpolate(flow,
                                       size=(2*flow.shape[2], 2*flow.shape[3]),
                                       mode='bilinear',
                                       align_corners=True,)

            predictions.append(flow_up)

        # 1/4 迭代估计的初始视差，由 1/8 的视差 4 倍 convex 上采样，再 1/2 下采样获得
        scale = fmap1.shape[2]/flow.shape[2]
        flow = scale * F.interpolate(flow,
                                    size=(fmap1.shape[2], fmap1.shape[3]),
                                   mode='bilinear',
                                   align_corners=True,)
        # 1/4 尺度下 10 次视差迭代，每次迭代的结果都存储计算 loss
        for itr in range(0):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True
            
            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)
            net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)
            flow = flow + delta_flow

            flow_up = -self.convex_upsample(flow, up_mask, rate=4)
            predictions.append(flow_up)
        
        # 测试时仅返回最后的估计结果
        if self.test_mode:
            return flow_up
        
        # 训练则将每次迭代估计的视差传递回去计算 loss
        return predictions

    # 4 倍视差上采样，用于 1/16  1/8 尺度的视差估计后，上采样进行之后的尺度的视差估计
    # 从隐藏状态生成 mask=[b, 4**2*9, h, w]，表示每个像素相邻 9 像素的 4*4 组权重，
    # 分别对每个像素的相邻 9 像素加权，得到每个像素 4*4 个扩充像素的视差，实现 4 倍上采样
    def convex_upsample(self, flow, mask, rate=4):
        N, C, H, W = flow.shape
        # 从 RNN 隐藏状态生成的视差 4 倍上采样 mask，分别对应每个像素相邻 9 个像素的 4*4 组权值，
        # 1 组权值对相邻 9 个像素加权融合输出 1 个像素值，共 16 个对应 4 倍上采样的 16 个输出像素
        mask = mask.reshape(N, 1, 9, rate, rate, H, W)  # [b, 4^2*9, h, w]——>[b, 1, 9, 4, 4, h, w]
        mask = F.softmax(mask, dim=2)

        # 3*3 滑窗提取视差图中每个像素的相邻 9 个像素，通过对其加权融合，得到每个像素的扩充像素
        up_flow = rate*flow  # 4 倍上采样中，光流大小也要扩充 4 倍
        # 在外围填充，用于对边界像素提取滑窗
        x = F.pad(up_flow, (1, 1, 1, 1), mode='constant', value=0)
        # 3*3 滑窗在提取每个像素的 9 个相邻像素
        x = x.unfold(dimension=2, size=3, step=1)
        x = x.unfold(dimension=3, size=3, step=1)
        x = x.reshape(N, C, -1, 3*3)                 # [b, c, h*w, 3*3]
        x = x.permute(0, 1, 3, 2)                       # [b, c, 3*3, h*w]
        up_flow = x.reshape(N, C, 3*3, 1, 1, H, W)  # [b, c, 3*3, 1, 1, h, w]

        # 用 mask 的 4*4 组权重对每个像素的相邻 9 像素加权，得到 4*4 个扩充像素的视差，实现 4 倍上采样
        up_flow = torch.sum(up_flow*mask, dim=2)  # [b, 2, 4, 4, h, w]
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [b, 2, h, 4, w, 4]
        up_flow = up_flow.reshape(N, 2, rate*H, rate*W)    # [b, 2, 4*h, 4*w]
        return up_flow
    
    # 生成和 fmap=[b  c  h  w] 同 size 的 0 初值光流=[b  2  h  w]
    def zero_init(self, fmap):
        N, _, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W], dtype=torch.float32)
        _y = torch.zeros([N, 1, H, W], dtype=torch.float32)
        zero_flow = torch.cat([_x, _y], dim=1).to(fmap.device)
        return zero_flow

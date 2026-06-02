import torch
import torch.nn as nn
import torch.nn.functional as F

# 卷积块 =《3*3卷积核 + BatchNorm2d归一化层 + LeakyReLU非线性激活函数》，bias 表示是否添加偏置
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(*[
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1), # LeakyReLU 将小于 0 的数*0.1            
        ])

    def forward(self, x):
        return self.block(x)

# 上采样解码 =《2 倍插值上采样 + 3*卷积块将通道数/2》
# 第一层卷积将输入 2 倍上采样后，与对应尺度 (上一尺度) 的编码结果 (shape 相同) 在通道上拼接
# 再经过一层卷积将通道 /2，最后经过一个不改变 size 的卷积块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(UpBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = ConvBlock(in_channels=2 * out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = ConvBlock(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding)

    def forward(self, x, others):
        # 插值上采样，使尺寸扩大两倍
        x = F.interpolate(x, scale_factor=(2,2))
        x = self.conv1(x)
        # x 和 others 在通道上拼接
        x = self.conv2(torch.cat([x, others], 1))
        x = self.conv3(x)

        return x

# 下采样编码 =《1/2 平均池化 + 3*卷积块将通道数*2》，实现 size/2 && channel*2
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DownBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.conv2 = ConvBlock(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        self.conv3 = ConvBlock(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        
    def forward(self, x):
        # 平均池化降采样，尺寸为原先一半
        x = F.avg_pool2d(x, (2,2))
        # 3 层相同卷积核大小的卷积，第一层控制通道数，后面保持
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x

# 从目标时刻前(后)固定数量事件生成的10(20)个 stack，重建用于双目深度估计的清晰事件表征，每个 stack 为不同数量的新增事件
# 使用 U-net 输出各 stack 的得分，之后对各 stack 加权融合，重建输出事件帧
class ConcentrationNet(nn.Module):
    def __init__(self, in_channels, base_channels=32, attention_method='soft'):
        super(ConcentrationNet, self).__init__()
        self.attention_method = attention_method
        # 两个卷积块，输出都是 32 通道，不改变size，预处理输入 stack
        self.conv1 = ConvBlock(in_channels=in_channels,
                              out_channels=base_channels,
                              kernel_size=(3,3),
                              padding=(1,1))
        self.conv2 = ConvBlock(in_channels=base_channels,
                              out_channels=base_channels,
                              kernel_size=(3,3),
                              padding=(1,1))
        # 三层编码层下采样，尺寸变为 1/2  1/4  1/8，通道数变为 32*2  32*4  32*8
        self.down1 = DownBlock(base_channels, base_channels*2, (3,3), (1,1))
        self.down2 = DownBlock(base_channels*2, base_channels*4, (3,3), (1,1))
        self.down3 = DownBlock(base_channels*4, base_channels*8, (3,3), (1,1))
        # 三层解码层上采样，尺寸变为 2  4  8，通道数变为 32*4  32*2  32
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, (3, 3), (1, 1))
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, (3, 3), (1, 1))
        self.up3 = UpBlock(base_channels * 2, base_channels, (3, 3), (1, 1))
        # 将 U-net 输出恢复到原通道数
        self.last_conv = nn.Conv2d(in_channels=base_channels,
                                        out_channels=in_channels,
                                        kernel_size=(3,3),
                                        padding=(1,1))
        
        # 参数初始化
        # self.modules() 遍历该网络中所有 nn.Module 的子类 (nn.Conv2d, nn.BatchNorm nn.LeakyRelu 等)，遍历包括模型自身/子模块等
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 输入单目事件 stack=[b  c=10/20  h  w]，返回重建 stack=[b  1  h  2]
    def forward(self, x):
        # 2 层卷积处理输入，改变通道数 [b  c  h  w]=[b  32  h  w]，送入 U-net
        s1 = self.conv2(self.conv1(x)) 
        
        # U-net 处理 10 通道事件 stack，返回同 shape 的张量，表示同像素上各通道 stack 的得分，即每个 stack 对应的新增事件的得分
        # U-net 编码层 = 3 层下采样编码
        s2 = self.down1(s1)     # [b  32  h  w]=[b  64  h/2  w/2]
        s3 = self.down2(s2)     # [b  64  h/2  w/2]=[b  128  h/4  w/4]
        out = self.down3(s3)  # [b  128  h/4  w/4]=[b  256  h/8  w/8]
        # U-net 解码层 = 3 层上采样解码，每层上采样后和对应尺度拼接，再恢复尺度，最后恢复到原尺度
        out = self.up1(out, s3)  # [b  256  h/8  w/8]= [b  128  h/4  w/4]
        out = self.up2(out, s2)  # [b  128  h/4  w/4]=[b  64  h/2  w/2]
        out = self.up3(out, s1)  # [b  64  h/2  w/2]=[b  32  h  w]
        # 将 U-net 输出恢复到原通道数 [b  32  h  w]=[b  10  h  w]
        out = self.last_conv(out)
        
        # 基于输出的各 stack 得分，使用 hard//soft 注意力机制融合所有 stack 得到重建事件输出
        # soft 注意力，通过各 stack 得分对相应的 stack 加权，像素值=各 stack 的加权融合
        if self.attention_method == 'soft':
            # 使用 softmax 计算每个像素在各通道上的权重，和输入特征融合
            soft_attention = F.softmax(out, dim=1)
            # 对各 stack 加权
            new_x = x * soft_attention
            # 每个像素上的值 = 该像素上各 stack 值的加权融合=[b  1  h  w]
            new_x = new_x.sum(dim=1, keepdim=True).contiguous()
        elif self.attention_method == 'hard': # hard 注意力，每个像素上取得分最高的 stack 值（报错）
            # max(dim=1) 取各像素沿通道方向的最大值，[1] 返回对应的通道索引=[b  h  w]，[0] 返回值
            hard_attention = out.max(dim=1)[1]
            # 输入的 stack，每个像素上，各通道都取该像素所有通道上得分最高的值=[b c h w]
            new_x = x[
                torch.arange(x.size(0), device='cuda').view(x.size(0), 1, 1, 1),
                torch.stack([hard_attention]*x.size(1), dim=1),
                torch.arange(x.size(2), device='cuda').view(1, 1, x.size(2), 1),
                torch.arange(x.size(3), device='cuda').view(1, 1, 1, x.size(3)),
            ]
            # error
            new_x = new_x.squeeze(dim=4).contiguous()
        else:
            raise NotImplementedError
        
        return new_x


class ImageReconstruct(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super(ImageReconstruct, self).__init__()
        # 两个卷积块，输出都是 32 通道，不改变size，预处理输入 stack
        self.conv1 = ConvBlock(in_channels=in_channels,
                              out_channels=base_channels,
                              kernel_size=(3,3),
                              padding=(1,1))
        self.conv2 = ConvBlock(in_channels=base_channels,
                              out_channels=base_channels,
                              kernel_size=(3,3),
                              padding=(1,1))
        # 三层编码层下采样，尺寸变为 1/2  1/4  1/8，通道数变为 32*2  32*4  32*8
        self.down1 = DownBlock(base_channels, base_channels*2, (3,3), (1,1))
        self.down2 = DownBlock(base_channels*2, base_channels*4, (3,3), (1,1))
        self.down3 = DownBlock(base_channels*4, base_channels*8, (3,3), (1,1))
        # 三层解码层上采样，尺寸变为 2  4  8，通道数变为 32*4  32*2  32
        self.up1 = UpBlock(base_channels * 8, base_channels * 4, (3, 3), (1, 1))
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, (3, 3), (1, 1))
        self.up3 = UpBlock(base_channels * 2, base_channels, (3, 3), (1, 1))
        # 将 U-net 输出恢复到原通道数
        self.last_conv = nn.Conv2d(in_channels=base_channels,
                                        out_channels=out_channels,
                                        kernel_size=(3,3),
                                        padding=(1,1))
        
        # 参数初始化
        # self.modules() 遍历该网络中所有 nn.Module 的子类 (nn.Conv2d, nn.BatchNorm nn.LeakyRelu 等)，遍历包括模型自身/子模块等
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 输入单目事件 stack=[b  c=10/20  h  w]，返回重建 stack=[b  1  h  2]
    def forward(self, x):
        # 2 层卷积处理输入，改变通道数 [b  c  h  w]=[b  32  h  w]，送入 U-net
        s1 = self.conv2(self.conv1(x)) 
        
        # U-net 处理 10 通道事件 stack，返回同 shape 的张量，表示同像素上各通道 stack 的得分，即每个 stack 对应的新增事件的得分
        # U-net 编码层 = 3 层下采样编码
        s2 = self.down1(s1)     # [b  32  h  w]=[b  64  h/2  w/2]
        s3 = self.down2(s2)     # [b  64  h/2  w/2]=[b  128  h/4  w/4]
        out = self.down3(s3)  # [b  128  h/4  w/4]=[b  256  h/8  w/8]
        # U-net 解码层 = 3 层上采样解码，每层上采样后和对应尺度拼接，再恢复尺度，最后恢复到原尺度
        out = self.up1(out, s3)  # [b  256  h/8  w/8]= [b  128  h/4  w/4]
        out = self.up2(out, s2)  # [b  128  h/4  w/4]=[b  64  h/2  w/2]
        out = self.up3(out, s1)  # [b  64  h/2  w/2]=[b  32  h  w]
        # 将 U-net 输出恢复到原通道数 [b  32  h  w]=[b  10  h  w]
        # out = self.last_conv(out)
        out = F.tanh(self.last_conv(out))
        
        return out
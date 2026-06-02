import torch
import torch.nn as nn
import torch.nn.functional as F


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.InstanceNorm2d(out_channels))


def conv_no_bn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))


# 基本残插层，《3*3 + batch + mish + 3*3 + batch》，输出后再和输入 or 输入的 downsample 相加
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   Mish())

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
    

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")

    def forward(self, x):
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x *( torch.tanh(F.softplus(x)))



class feature_encoder(nn.Module):
    def __init__(self, input_dim):
        super(feature_encoder, self).__init__()

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(input_dim, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())

        self.layer2 = self._make_layer(BasicBlock, 64, 1, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
    
    # blocks 都是 1，即生成的都是 1 层残插网络
    #《3*3 + batch + mish + 3*3 + batch》，输出后再和输入 or 输入的 downsample 相加
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        # block.expansion 是 1，即输出通道由 planes 直接控制
        # 残插层中的 downsample =《1*1 + batch》
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion), )

        layers = []
        # 基本残插层，《3*3 + in + mish + 3*3 + in》，输出后再和输入 or 输入的 downsample 相加
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 3 *《3*3 + in + mish》输出 32 通道特征 = [b,  c=32,  h/2,  w/2]
        x = self.firstconv(x)
        # 对特征层层残差编码，得到多尺度特征
        # 1 层残插层 = [b,  c=64,  h/2,  w/2]
        l2 = self.layer2(x)     #1/2
        # 1 层残插层 = [b,  c=128,  h/4,  w/4]
        l3 = self.layer3(l2)    #1/4
        # 1 层残插层 = [b,  c=256,  h/8,  w/8]
        l4 = self.layer4(l3)    #1/8
        # 1 层残插层 = [b,  c=512,  h/16,  w/16]
        l5 = self.layer5(l4) 
        
        return [x, l2, l3, l4, l5]


class feature_decoder(nn.Module):
    def __init__(self):
        super(feature_decoder, self).__init__()
        
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv_no_bn(512, 256, 3, 1, 1, 1),
                                     Mish())
        self.iconv4 = nn.Sequential(conv_no_bn(512, 256, 3, 1, 1, 1),
                                    Mish())
        
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv_no_bn(256, 128, 3, 1, 1, 1),
                                     Mish())
        self.iconv3 = nn.Sequential(conv_no_bn(256, 128, 3, 1, 1, 1),
                                    Mish())
        
        self.gw3 = nn.Sequential(conv_no_bn(128, 128, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(128, 128, kernel_size=1, padding=0, stride=1,
                                           bias=False))
        
        self.gw4 = nn.Sequential(conv_no_bn(256, 256, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(256, 128, kernel_size=1, padding=0, stride=1,
                                           bias=False))
    
    # blocks 都是 1，即生成的都是 1 层残插网络
    #《3*3 + batch + mish + 3*3 + batch》，输出后再和输入 or 输入的 downsample 相加
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        # block.expansion 是 1，即输出通道由 planes 直接控制
        # 残插层中的 downsample =《1*1 + batch》
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(planes * block.expansion), )

        layers = []
        # 基本残插层，《3*3 + in + mish + 3*3 + in》，输出后再和输入 or 输入的 downsample 相加
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)
        

    def forward(self, en_feat):
        l3, l4, l5 = en_feat
        # 层层解码
        # 1/16 解码特征《 2 倍上采样 + batch》后和 1/8 concat +《3*3 + batch + mish》=  [b,  192,  h/8,  w/8]
        concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)
        # 1/8 解码特征《 2 倍上采样 + batch》后和 1/4 concat +《3*3 + batch + mish》=  [b,  128,  h/4,  w/4]
        concat3 = torch.cat((l3, self.upconv4(decov_4)), dim=1)
        decov_3 = self.iconv3(concat3)

        gw3 = self.gw3(decov_3)
        gw4 = self.gw4(decov_4)
        # gw5 = self.gw5(self.layer5(l5))
        
        return gw3, gw4


class recons_decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(recons_decoder, self).__init__()

        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv_no_bn(512, 256, 3, 1, 1, 1),
                                     Mish())
        self.iconv5 = nn.Sequential(conv_no_bn(512, 256, 3, 1, 1, 1),
                                    Mish())
        
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv_no_bn(256, 128, 3, 1, 1, 1),
                                     Mish())
        self.iconv4 = nn.Sequential(conv_no_bn(256, 128, 3, 1, 1, 1),
                                    Mish())
        
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv_no_bn(128, 64, 3, 1, 1, 1),
                                     Mish())
        self.iconv3 = nn.Sequential(conv_no_bn(128, 64, 3, 1, 1, 1),
                                    Mish())
        
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     conv_no_bn(64, 32, 3, 1, 1, 1),
                                     Mish())
        self.iconv2 = nn.Sequential(conv_no_bn(64, 32, 3, 1, 1, 1),
                                    Mish())

        self.pred = nn.Sequential(
                                 nn.Conv2d(32, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
                                 nn.Sigmoid())

    def forward(self, en_feat):
        x, l2, l3, l4, l5 = en_feat
        # 层层解码
        # 1/16 特征《 2 倍上采样 + batch》后和 1/8 concat +《3*3 + batch + mish》=  [b,  256,  h/16,  w/16]
        concat5 = torch.cat((l4, self.upconv6(l5)), dim=1)
        decov_5 = self.iconv5(concat5)
        # 1/8 解码特征《 2 倍上采样 + batch》后和 1/4 concat +《3*3 + batch + mish》=  [b,  128,  h/4,  w/4]
        concat4 = torch.cat((l3, self.upconv5(decov_5)), dim=1)
        decov_4 = self.iconv4(concat4)
        # 1/4 解码特征《 2 倍上采样 + batch》后和 1/2 concat +《3*3 + batch + mish》=  [b,  64,  h/4,  w/4]
        concat3 = torch.cat((l2, self.upconv4(decov_4)), dim=1)
        decov_3 = self.iconv3(concat3)
        # 1/2 解码特征《 2 倍上采样 + batch》后和 1 concat +《3*3 + batch + mish》=  [b,  64,  h/2,  w/2]
        concat2 = torch.cat((x, self.upconv3(decov_3)), dim=1)
        decov_2 = self.iconv2(concat2)

        pred = self.pred(decov_2)

        return pred


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)

def skip_sum(x1, x2):
    return x1 + x2

class recons_decoder_1layer(nn.Module):
    def __init__(self, use_upsample_conv=True, output_channels=3, skip_type='sum', activation='sigmoid', norm=None):
        super(recons_decoder_1layer, self).__init__()
        self.norm = norm
        self.output_channels = output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation

        if use_upsample_conv:
            self.UpsampleLayer = UpsampleConvLayer
        else:
            self.UpsampleLayer = TransposedConvLayer

        decoder_input_sizes = [512, 256]
        decoder_output_sizes = [256, 128]

        self.decoders = nn.ModuleList()
        for i, input_size in enumerate(decoder_input_sizes):
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    decoder_output_sizes[i], kernel_size=5, padding=2, norm=self.norm))

        self.pred = ConvLayer(decoder_output_sizes[-1] if self.skip_type == 'sum' else 2 * decoder_output_sizes[-1],
                              self.output_channels, 1, activation=None, norm=self.norm)
    
        self.activation = getattr(torch, self.activation, 'sigmoid')


    def forward(self, x):
        l0, l1, l2 = x

        deconv = self.decoders[0](l2)  ## 1/16 --- 1/8
        deconv = self.decoders[1](self.apply_skip_connection(deconv, l1))    ## 1/8 --- 1/4      
        img = self.activation(self.pred(self.apply_skip_connection(deconv, l0)))    ## 1/8 --- 1/4

        return img
    
class decoder_layer(nn.Module):
    def __init__(self, input_dim, up_dim, middle_dim, output_dim):
        super(decoder_layer, self).__init__()
        self.upconv = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(input_dim, up_dim, 3, 1, 1, 1),
                                     Mish())
        self.iconv = nn.Sequential(convbn(up_dim*2, up_dim, 3, 1, 1, 1),
                                    Mish())
        
        self.gw = None
        if output_dim is not None:
            self.gw = nn.Sequential(convbn(up_dim, middle_dim, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(middle_dim, output_dim, kernel_size=1, padding=0, stride=1,
                                           bias=False))

    def forward(self, input, up):
        # 1/32 特征《 2 倍上采样 + in》后和 1/16 concat +《3*3 + in + mish》=  [b,  256,  h/16,  w/16]
        concat = torch.cat((up, self.upconv(input)), dim=1)
        decov = self.iconv(concat)
        if self.gw is not None:
            # 输出层 =《3*3 + in + mish + 1*1》
            output = self.gw(decov)       
            return decov, output
        else:
            return decov
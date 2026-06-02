import numpy as np
import torch
import torch.nn  as  nn
import torch.nn.functional as F
from .utils import coords_grid, bilinear_sampler

class AGCL:
    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.att = att
        # meshgrid 得到特征图的坐标系，并且重复 batch 次=[b  2  h  w]，对应像素坐标 (x, y)
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3]).to(fmap1.device)

    def resetLeft(self, fmap1):
        assert self.fmap1.shape == fmap1.shape
        self.fmap1 = fmap1

    def resetRight(self, fmap2):
        assert self.fmap2.shape == fmap2.shape
        self.fmap2 = fmap2
    
    # 初始化时已经传入的特征图，之后每次调用函数计算双目相关性，传入新的初始视差和搜索模式
    # 在给定初始视差下，计算特征图的多头双目相关性=[b, 9*4, h, w]，搜索模式为 3*3 或 1*9 搜索
    def __call__(self, flow, extra_offset, small_patch=False, iter_mode=False, cross_att=None):
        if iter_mode:
            corr = self.corr_iter(self.fmap1, self.fmap2, flow, small_patch)
        else:
            corr = self.corr_att_offset(self.fmap1, self.fmap2, flow, extra_offset, small_patch, cross_att=cross_att)
        return corr


    # 在给定初始视差下，计算特征图的多头双目相关性=[b, 9*4, h, w]，搜索模式为 3*3 或 1*9 搜索
    # 由于没有使用额外搜索偏置，因此使用 grid_sample 对右目每个像素视差对齐后，
    # 直接用滑窗整体平移右目特征图，平移 9 次，每次计算相关性
    def corr_iter(self, left_feature, right_feature, flow, small_patch):
        # 基于初始视差，将右目特征图映射到新的坐标
        # 2. 这里的初始视差，也是加到坐标系上进行右目特征提取，因此该初始视差 = -真实视差，
        # 即每次迭代估计的左目视差，是负的真实视差，所以每次迭代估计的视差和真实视差对比时取负；
        coords = self.coords + flow # [b  2  h  w]
        coords = coords.permute(0, 2, 3, 1) # [b  h  w  2]
        right_feature = bilinear_sampler(right_feature, coords)
        
        N, C, H, W = left_feature.shape        
        lefts = torch.split(left_feature, C//4, dim=1)
        rights = torch.split(right_feature, C//4, dim=1)
        
        # 对每个像素，在 y/x 方向搜索的像素数。每个尺度进行 5 次迭代视差估计。
        if small_patch:
            psize = (3,3)   # 1/3 次估计时，搜索周围 9 个像素
        else:
            psize = (1,9)   # 0/2/4 次估计时，搜索 x 方向左右 9 个像素
        psizey, psizex = psize[0], psize[1]
        pady, padx = psizey//2, psizex//2
        # 对右目特征图进行复制填充，填充上下左右各一个像素，或左右各 4 个像素
        padding = (padx, padx, pady, pady)  # 按照左/右/上/下的顺序填充

        corrs = []
        for i in range(len(lefts)):
            # 填充这一组右目特征图
            right_pad = F.pad(rights[i], padding, mode="replicate")
            # 滑动窗口从填充特征图中提取 9 个 [h w] 的特征图，对应 9 个搜索偏置下的右目特征图
            right_slide = right_pad.unfold(dimension=2, size=H, step=1)
            right_slide = right_slide.unfold(dimension=3, size=W, step=1)
            # 修改张量形状
            right_slide = right_slide.contiguous().view(N, C//4, -1, H, W)  # [b  c=64  9  H  W]
            left_feature = lefts[i].unsqueeze(dim=2)   # [b  c=64  1  H  W]
            # 计算双目相关性
            corr = torch.mean(left_feature*right_slide, dim=1)  # [b  9  H  W]
            corrs.append(corr)

        # 将 4 组特征在 9 个搜索偏置下的双目相关性 concat=[b, 9*4, h, w]
        final_corr = torch.cat(corrs, dim=1)
        return final_corr


    # 在给定初始视差下，计算特征图的多头双目相关性，视差模式设置为 3*3 搜索 或 1*9 搜索
    # 基于每个像素的初始视差和额外搜索偏置，对右目特征图每个像素平移周围 9 个像素，
    # 并将 256 通道特征分为 4 份，计算双目相关性=[b, 9*4, h, w]，表示每组特征在每个搜索偏置下的双目相关性
    # 由于使用了额外搜索偏置，右目每个像素的 9 个搜索像素都有一个不同的偏置，因此使用 grid_sample
    def corr_att_offset(self, left_feature, right_feature, flow, extra_offset, small_patch, cross_att=None):
        N, C, H, W = left_feature.shape
        
        # 对双目输入特征进行 cross 注意力处理，仅 1/16 特征图
        if self.att is not None:
            # 双目特征平铺 [b, c, h, w]——>[b, h*w, c]
            left_feature, right_feature = [
                        x.permute(0, 2, 3, 1).reshape(N, H*W, C)  for x in [left_feature, right_feature]
                    ]
            # cross 注意力处理双目特征
            left_feature, right_feature = self.att(left_feature, right_feature)
            # 多头输出恢复 shape：[b, h*w, 8, c//8]——>[b, c, h, w]
            left_feature, right_feature = [
                        x.reshape(N, H, W, C).permute(0, 3, 1, 2)  for x in [left_feature, right_feature]
                    ]
            
        if cross_att is not None:
            coords_att = self.coords + flow # [b  2  h  w]
            coords_att = coords_att.permute(0, 2, 3, 1) # [b  h  w  2]
            right_feature_att = bilinear_sampler(right_feature, coords_att)
            left_feature = cross_att(left_feature, right_feature_att)
            self.resetLeft(left_feature)

            coords_att = self.coords - flow # [b  2  h  w]
            coords_att = coords_att.permute(0, 2, 3, 1) # [b  h  w  2]
            left_feature_att = bilinear_sampler(left_feature, coords_att)
            right_feature = cross_att(right_feature, left_feature_att)
            self.resetRight(right_feature)
        
        # 沿通道将左右目特征图分为 4 份进行多头相关性计算，每份特征数=256/4=64，即分别计算不同组特征的相关性
        C = C//4
        lefts = torch.split(left_feature, C, dim=1)
        rights = torch.split(right_feature, C, dim=1)
        
        search_num = 9
        # 每个像素的 9 个搜索像素上的额外偏置
        # [b  18  h  w]——>[b  9  2  h  w]——>[b  9  h  w  2]
        extra_offset =  (torch.reshape(extra_offset, (N, search_num, 2, H, W))).permute(0, 1, 3, 4, 2)
        
        # 对每个像素，在 y/x 方向搜索的像素数，每个尺度进行 5 次迭代视差估计
        if small_patch:
            psize = (3,3)   # 1/3 次估计时，搜索周围 9 个像素
        else:
            psize = (1,9)   # 0/2/4 次估计时，搜索 x 方向左右 9 个像素
        
        psizey, psizex = psize[0], psize[1]
        ry, rx = psizey//2, psizex//2
        # 生成每个像素固定的 3*3 或 1*9 搜索偏置，x=[[-rx, -rx+1 ... rx] ...],    y=[[-ry ...], [-ry+1 ...] ... [ry ...]]
        x_grid, y_grid = np.meshgrid(np.arange(-rx, rx+1), np.arange(-ry, ry+1))
        x_grid = torch.tensor(x_grid, device=self.fmap1.device)
        y_grid = torch.tensor(y_grid, device=self.fmap1.device)
        offsets = torch.stack([x_grid, y_grid], dim=0)     # 2*3*3 或 2*1*9
        offsets = torch.reshape(offsets, (2, -1))                  # 2*9
        offsets = offsets.permute(1, 0)                                   # 9*2 对应像素的 9 个固定搜索像素的偏置 (x,y)
        for dim in [0,2,3]:                                                             # 先 x 后 y，配合 grid_sample
            offsets = offsets.unsqueeze(dim=dim)              # [1  9  1  1  2]
            
        # 每个像素上的 9 个搜索偏置  =  固定偏置 + 额外搜索偏置，得到 [b  9  h  w  2]
        offsets = offsets + extra_offset # [b  9  h  w  2]
        # 在上一次估计的视差的基础上进行；这一步决定了 flow 的 x y 顺序和 coords [b  2  h  w] 保持一致，即先 x 后 y
        # coords 用于在 bilinear_sampler 中提取像素值，由于 grid_sample 函数按照 x y 的顺序提取，因此 coords 也先 x 后 y
        coords = self.coords + flow  # [b  2  h  w]
        coords = coords.permute(0, 2, 3, 1)  # [b  h  w  2]
        coords = coords.unsqueeze(dim=1)  # [b  1  h  w  2]
        # 添加偏置
        coords = coords + offsets  # [b  9  h  w  2]
        coords = torch.reshape(coords, (N, -1, W, 2))  # [b  9*h  w  2]
        
        corrs = []
        # 对 4 组特征，分别添加搜索偏置，计算双目相关性
        for i in range(len(lefts)):
            left_feature, right_feature = lefts[i], rights[i]  # [b,  64,  h,  w]            
            # 根据添加了视差搜索偏置的坐标系，从右目特征图上提取特征，对整张右目特征图平移 9 次
            # Note: 
            # 1. 右目 + offset 的像素提取 = 右目每个像素取右边的像素值 = 右目向左平移，
            # 而左目视差 d 对应右目向右平移 d，即右目每个像素取左边的像素值；
            # 因此，offset 对应的左目视差是 -offset（该代码原先用于光流估计，前后时刻的 offset>0）；
            # 而这一步只是获取 9 个 offset 下的双目相关性(不输出 offset)，具体视差由下一步 RNN 从该相关性中计算；
            # 因此不需要 offset 和视差之间完全对应，代码可以将 offset 对应的双目相关性直接理解为视差=-offset 的双目相关性；
            # 即，offset 是加 or 减到坐标系上没有任何影响；
            # 2. 这里输入的初始视差，也是加到坐标系上进行右目特征提取，因此该初始视差 = -真实视差，
            # 即每次迭代估计的左目视差，是负的真实视差，所以每次迭代估计的视差和真实视差对比时取负；
            # 3. 对右目 (x,y) 像素取 (x-d, y) 的值，再和左目 (x,y) 计算相关性，
            # 即对左目每个像素，取其左边 d 的右目像素计算相关性，对应的 d 是左目 (x,y) 处的视差
            right_feature = bilinear_sampler(right_feature, coords)          # [b  c=64  9*h  w]
            right_feature = torch.reshape(right_feature, (N, C, -1, H, W))  # [b  c=64  9  h  w]
            
            left_feature = left_feature.unsqueeze(dim=2)  # [b  c=64  1  h  w]
            # 左右目像素间相关性 = 特征相乘/特征总数=[b, 9, h, w]
            corr = torch.mean(left_feature*right_feature, dim=1)
            corrs.append(corr)
        
        # 将 4 组特征在 9 个搜索偏置下的双目相关性 concat=[b, 9*4, h, w]
        final_corr = torch.cat(corrs, dim=1)
        return final_corr
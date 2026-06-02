import copy
import torch
import torch.distributed as dist

# 总 error 记录器，
# 一个 epoch 总 loss 记录器，为每个 batch 的 loss 之和
# 定义更新方法，用作记录的某具体误差的属性
class SummationMeter:
    # string_format 表示 __str__ 中的输出格式
    def __init__(self, string_format):
        self.sum = 0
        self.string_format = string_format
    
    # 更新输入 1 组 batch_size 中各 batch 的均值 loss，总 loss 更新为 += b*loss
    def update(self, val, n=1):
        self.sum += val*n
    
    # 定义 .value 属性，返回总 loss
    @property
    def value(self):
        return copy.copy(self.sum)
    
    # 定义 str() 函数，返回输出总 loss
    def __str__(self):
        return self.string_format % self.value
    
    def reset(self):
        self.__init__(string_format=self.string_format)


# 一个 epoch 平均 loss 记录器，为每个 batch 的 loss 之和/总数据量
# 定义更新方法，用作记录的某具体误差的属性
class AverageMeter(SummationMeter):
    def __init__(self, string_format=None):
        super().__init__(string_format=string_format)
        self.count = 0
        self.avg = 0

    # 更新输入各 batch 的均值 loss，平均 loss 更新为添加 b*loss 后除以总数据量
    def update(self, val, n=1):
        super().update(val=val, n=n)
        self.count += n
        self.avg = self.sum/self.count

    # .value 返回属性为均值 loss
    @property
    def value(self):
        return copy.copy(self.avg)
    

# 定义特定的视差 error，每次输入 1 个 batch_size 的预测视差和真值，更新该 epoch 的平均 error
# 最终得到该 epoch 所有 batch 的平均对应误差
class Metric:
    def __init__(self, average_by='image', string_format=None):
        assert average_by in ['pixel', 'image']
        self.average_meter = AverageMeter(string_format=string_format)
        self.average_by = average_by  # 'image'
    
    # 输入 1 个 batch_size 的预测视差和真值，更新该 epoch 的平均 error
    @torch.no_grad()
    def update(self, pred, ground_truth, mask):
        data_count = 0
        error = 0.0
        # 每次输入 1 个 batch_size 的预测和真值，仅循环 1 次
        for p, gt, m in zip(pred, ground_truth, mask):
            if not m.any():
                continue
            error += self.calculate_error(p, gt, m).to(torch.float).item()
            if self.average_by == 'pixel':
                data_count += m.sum().item()
            elif self.average_by == 'image':
                data_count += 1
            else:
                raise NotImplementedError
            
        error /= data_count
        self.average_meter.update(val=error, n=data_count)
    
    # 返回平均 error
    @property
    def value(self):
        return self.average_meter.value
    
    # 定义 str() 函数，返回 error 均值
    def __str__(self):
        return str(self.average_meter)

    def reset(self):
        self.__init__(average_by=self.average_by, string_format=self.average_meter.string_format)
    
    def calculate_error(self, pred, ground_truth, mask):
        raise NotImplementedError
    

# 计算所有像素的平均视差 error，每次输入 1 个 batch_size 的估计结果，更新这个 epoch 的估计结果，
# 最后更新得到该 epoch 的所有 batch 的像素平均视差 error
class EndPointError(Metric):
    def calculate_error(self, pred, ground_truth, mask):
        pred, ground_truth = pred[mask], ground_truth[mask]
        # 计算每个像素的视差 error
        error = torch.abs(pred - ground_truth)
        # 使用 'image'，计算 batch_size 个图像所有像素的视差 error 的均值
        if self.average_by == 'pixel':
            final_error = error.sum()
        elif self.average_by == 'image':
            final_error = error.mean()
        else:
            raise NotImplementedError
        
        return final_error


# 计算视差 error > n 的像素占比，最后更新得到该 epoch 的所有 batch 的像素占比
class NPixelError(Metric):
    def __init__(self, n=1, average_by='image', string_format=None):
        super().__init__(average_by=average_by, string_format=string_format)
        self.n = n
    
    def calculate_error(self, pred, ground_truth, mask):
        pred, ground_truth = pred[mask], ground_truth[mask]
        # 计算每个像素的视差 error
        error = torch.abs(pred - ground_truth)
        # 取 error 大于 n 的像素
        error_mask = error > self.n
        # bool 转换为 float 用于计算
        error_mask = error_mask.to(torch.float)

        # 使用 'image'，计算 batch_size 个图像像素视差 error > n 数量的占比
        if self.average_by == 'pixel':
            final_error = error_mask.sum() # 误差大于 n 的像素总数
        elif self.average_by == 'image':
            final_error = error_mask.mean()
        else:
            raise NotImplementedError
        
        return final_error * 100.0


# 各像素视差 error 的均方根误差
class RootMeanSquareError(Metric):
    def calculate_error(self, pred, ground_truth, mask):
        assert self.average_by == 'image'
        pred, ground_truth = pred[mask], ground_truth[mask]
        error = ((pred - ground_truth) ** 2).mean().sqrt()
        return error
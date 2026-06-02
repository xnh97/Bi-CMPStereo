import torch
import torch.distributed as dist

from tqdm import tqdm
from collections import OrderedDict
from utils.metric import AverageMeter, EndPointError, NPixelError, RootMeanSquareError
from torch.cuda.amp import autocast, GradScaler
from utils.summary import ModelWarper, summary, summary_with_loss

def check_gradients_by_module(model):
    problem_modules = []
    
    for module_name, module in model.named_modules():
        if len(list(module.parameters())) > 0:  # 只检查有参数的模块
            has_problem = False
            for param_name, param in module.named_parameters(recurse=False):  # 不递归
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_problem = True
                        break
            
            if has_problem:
                problem_modules.append(module_name)
                print(f"Module {module_name} has gradient problems")
    
    return problem_modules

def analyze_gradient_norms(model):
    grad_norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            
            # 异常大的梯度范数通常指示问题
            if grad_norm > 1000:  # 可调整阈值
                print(f"Large gradient norm in {name}: {grad_norm}")
    
    # 找出梯度范数最大的几个参数
    sorted_norms = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 largest gradient norms:")
    for name, norm in sorted_norms[:5]:
        print(f"  {name}: {norm}")
    
    return grad_norms

# 1 个 epoch 训练，对所有数据进行 1 次训练，记录并返回这次 epoch 的整体平均误差指标
def train(model, data_loader, optimizer, is_distributer=False, world_size=1, cross_modality=False):
    # 将模型设置为训练模式
    model.train()
    # '%6.3lf' 字符串：%=字符串的开始，6=输出 6 个字符，不足 6 位则在前面填充空格，
    # 超过 6 位则完整输出而不截断，.3=小数点后保留 3 位数字，lf=长双精度浮点数 (long double)
    # 对该 epoch 记录视差估计 error，共 5 个，每个都是 1 个数，每次对 batch_size 个数据估计后，更新该结果
    log_dict = OrderedDict([
        ('Loss',  AverageMeter(string_format='%6.3lf')),
        ('EPE', EndPointError(average_by='image', string_format='%6.3lf')),
        ('1PE', NPixelError(n=1, average_by='image', string_format='%6.3lf')),
        ('2PE', NPixelError(n=2, average_by='image', string_format='%6.3lf')),
        ('RMSE', RootMeanSquareError(average_by='image', string_format='%6.3lf')),
        ('l1-1',  AverageMeter(string_format='%6.3lf')),
        ('l1-2',  AverageMeter(string_format='%6.3lf')),
        ('l1-3',  AverageMeter(string_format='%6.3lf')),
        #('l1-4',  AverageMeter(string_format='%6.3lf')),
        ('l2-1',  AverageMeter(string_format='%6.3lf')),
        ('l2-2',  AverageMeter(string_format='%6.3lf')),
        ('l2-3',  AverageMeter(string_format='%6.3lf')),
        #('l2-4',  AverageMeter(string_format='%6.3lf')),
    ])
    
    scaler = GradScaler()
    # 每个纪元，遍历数据集每个序列，tqdm 为可视化进度条
    for batch_data in tqdm(data_loader):
        with autocast():
            # 数据都放到 cuda 上
            batch_data = batch_to_cuda(batch_data)
            # 取可用视差区域
            mask = batch_data['disparity'] > 0
            # mask 数组中所有元素都为 False 时（视差图无真值）continsummary_with_lossue
            if not mask.any(): 
                continue

            # summary(model.module.concentration_net, (1, 10, 432 , 576))
            # summary(ModelWarper(model.module.stereo_matching_net), (2, 1, 1, 432 , 576))

            # 调用模型的前向传播函数 forwad 预测视差图并计算 loss
            if not cross_modality:
                pred, loss = model(left_event=batch_data['event']['left'],
                           right_event=batch_data['event']['right'],
                           gt_disparity=batch_data['disparity'])
            else:
                pred, loss, loss_scale = model(event=batch_data['event']['right'],
                           image=batch_data['image'],
                           gt_disparity=batch_data['disparity'])
            # 优化器梯度清零，
            optimizer.zero_grad()
            # loss 为所有像素的各尺度加权 loss 的均值，即多个 batch 的 loss 为各自 loss 的均值，是 1 个数
            # 算法公式为 loss = (l1 + l2 + ...)/b，因此在反向传播梯度计算时会对每个 batch 分别计算，
            # 再将同样参数的梯度相加，则将多个 batch 的 loss 平均为 1 个数和分别计算 loss 反向传播等价
            loss = loss.mean()
            
            # 反向传播梯度计算
            # loss.backward()
            scaler.scale(loss).backward()
            
            grad_stats = {
                'total_params': 0,
                'params_with_grad': 0,
                'nan_grads': 0,
                'inf_grads': 0
                }
            
            for name, param in model.module.stereo_matching_net.named_parameters():
                grad_stats['total_params'] += 1
                if param.grad is not None:
                    grad_stats['params_with_grad'] += 1
                    if torch.isnan(param.grad).any():
                        grad_stats['nan_grads'] += 1
                        # print(f"{name} has nan gradients")
                    if torch.isinf(param.grad).any():
                        grad_stats['inf_grads'] += 1
                        print(f"{name} has inf gradients")
                        
            if grad_stats['inf_grads'] > 0 or grad_stats['nan_grads'] > 0:
                print(f"Gradient stats: {grad_stats}")
                print("=== Detailed Gradient Analysis ===")
                check_gradients_by_module(model.module.stereo_matching_net)
                analyze_gradient_norms(model.module.stereo_matching_net)

            # 优化器更新模型参数，每个 epoch 使用相同的学习率，scheduler 对不同 epoch 调整学习率
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            pred = pred[:,0,:,:]

            # 1 组 batch_size 估计后，更新该 epoch 的视差估计度量 error
            # 更新平均 loss，区分 'EPE'，loss 是每个像素上各尺度视差 error 的加权均值
            log_dict['Loss'].update(loss.item(), pred.size(0))
            # 平均像素视差估计 error，仅由原始尺度预测的视差获取
            log_dict['EPE'].update(pred, batch_data['disparity'], mask)
            # 视差估计 > 1 的像素数量占比，仅由原始尺度预测的视差获取
            log_dict['1PE'].update(pred, batch_data['disparity'], mask)
            # 视差估计 > 2 的像素数量占比，仅由原始尺度预测的视差获取
            log_dict['2PE'].update(pred, batch_data['disparity'], mask)
            # 像素视差估计均方根 error，仅由原始尺度预测的视差获取
            log_dict['RMSE'].update(pred, batch_data['disparity'], mask)

            log_dict['l1-1'].update(loss_scale[0].mean().item(), pred.size(0))
            log_dict['l1-2'].update(loss_scale[1].mean().item(), pred.size(0))
            log_dict['l1-3'].update(loss_scale[2].mean().item(), pred.size(0))
            #log_dict['l1-4'].update(loss_scale[3].mean().item(), pred.size(0))
            log_dict['l2-1'].update(loss_scale[3].mean().item(), pred.size(0))
            log_dict['l2-2'].update(loss_scale[4].mean().item(), pred.size(0))
            log_dict['l2-3'].update(loss_scale[5].mean().item(), pred.size(0))
            #log_dict['l2-4'].update(loss_scale[7].mean().item(), pred.size(0))

    # 返回该 epoch 训练误差指标
    return log_dict


@torch.no_grad()
def test(model, data_loader, cross_modality=False):
    model.eval()
    
    log_dict = OrderedDict([
        ('Loss',  AverageMeter(string_format='%6.3lf')),
        ('EPE', EndPointError(average_by='image', string_format='%6.3lf')),
        ('1PE', NPixelError(n=1, average_by='image', string_format='%6.3lf')),
        ('2PE', NPixelError(n=2, average_by='image', string_format='%6.3lf')),
        ('l1-1',  AverageMeter(string_format='%6.3lf')),
        ('l1-2',  AverageMeter(string_format='%6.3lf')),
        ('l1-3',  AverageMeter(string_format='%6.3lf')),
        #('l1-4',  AverageMeter(string_format='%6.3lf')),
        ('l2-1',  AverageMeter(string_format='%6.3lf')),
        ('l2-2',  AverageMeter(string_format='%6.3lf')),
        ('l2-3',  AverageMeter(string_format='%6.3lf')),
        #('l2-4',  AverageMeter(string_format='%6.3lf')),
    ])

    # 对于单个序列，遍历所有数据：视差图+对应事件representation
    for batch_data in tqdm(data_loader):
        batch_data = batch_to_cuda(batch_data)
        
        mask = batch_data['disparity'] > 0
        if not mask.any(): 
            continue

        if not cross_modality:
            pred, _ = model(left_event=batch_data['event']['left'],
                           right_event=batch_data['event']['right'],
                           gt_disparity=batch_data['disparity'])
        else:
            pred, loss, loss_scale = model(event=batch_data['event']['right'],
                           image=batch_data['image'],
                           gt_disparity=batch_data['disparity'])

        loss = loss.mean()
        pred = pred[:,0,:,:]        

        # 1 组 batch_size 估计后，更新该 epoch 的视差估计度量 error
        log_dict['Loss'].update(loss.item(), pred.size(0))
        # 平均像素视差估计 error，仅由原始尺度预测的视差获取
        log_dict['EPE'].update(pred, batch_data['disparity'], mask)
        # 视差估计 > 1 的像素数量占比，仅由原始尺度预测的视差获取
        log_dict['1PE'].update(pred, batch_data['disparity'], mask)
        # 视差估计 > 2 的像素数量占比，仅由原始尺度预测的视差获取
        log_dict['2PE'].update(pred, batch_data['disparity'], mask)
        
        log_dict['l1-1'].update(loss_scale[0].mean().item(), pred.size(0))
        log_dict['l1-2'].update(loss_scale[1].mean().item(), pred.size(0))
        log_dict['l1-3'].update(loss_scale[2].mean().item(), pred.size(0))
        #log_dict['l1-4'].update(loss_scale[3].mean().item(), pred.size(0))
        log_dict['l2-1'].update(loss_scale[3].mean().item(), pred.size(0))
        log_dict['l2-2'].update(loss_scale[4].mean().item(), pred.size(0))
        log_dict['l2-3'].update(loss_scale[5].mean().item(), pred.size(0))
        #log_dict['l2-4'].update(loss_scale[7].mean().item(), pred.size(0))

    return log_dict


# 将事件左右目 stack 和视差图放到 cuda 上
def batch_to_cuda(batch_data):
    # 定义了一个嵌套的辅助函数，仅在对事件处理时调用，会将字典或 tensor 放到 cuda
    def _batch_to_cuda(batch_data):
        if isinstance(batch_data, dict):
            for key in batch_data.keys():
                batch_data[key] = _batch_to_cuda(batch_data[key])
        elif isinstance(batch_data, torch.Tensor):
            batch_data = batch_data.cuda()
        else:
            raise NotImplementedError
        
        return batch_data
    
    # 将事件左右目 stack 放到 cuda
    for domain in ['event']:
        if domain not in batch_data.keys():
            batch_data[domain] = {}
        for location in ['left', 'right']:
            if location in batch_data[domain].keys():
                batch_data[domain][location] = _batch_to_cuda(batch_data[domain][location])
            else: 
                batch_data[domain][location] = None
    # 将视差图放到 cuda
    if 'disparity' in batch_data.keys() and batch_data['disparity'] is not None:
        batch_data['disparity'] = batch_data['disparity'].cuda()
    if 'image' in batch_data.keys() and batch_data['image'] is not None:
        batch_data['image'] = batch_data['image'].cuda()

    return batch_data
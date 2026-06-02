import os

import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from utils.logger import ExpLogger
from utils.metric import SummationMeter, Metric
from components.datasets import dsec
from components.model import event_stereo, event_frame_stereo, event_encode_frame_stereo
import method

class DLManager:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg =cfg        
        # 训练记录器，输入路径初始化，新建 save 文件夹
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None
        # 配置初始化
        if self.cfg is not None:
            self._init_from_cfg(cfg)
        # 当前训练纪元
        self.current_epoch = 0
        if self.args.is_continue_train:
            self.load_concentration(name=self.args.continue_path)

    # 配置初始化，生成数据集//模型//优化器//调度器
    def _init_from_cfg(self, cfg):
        assert cfg is not None
        # cfg['DATALOADER']['TRAIN']['PARAMS']['batch_size']=1
        self.cfg = cfg

        # 数据集 dataloader
        self.get_train_loader = dsec.get_dataloader
        self.get_test_loader = dsec.get_dataloader
        # 训练方法
        self.method = method
        # 初始化模型
        self.model = _prepare_model(self.cfg.MODEL,
                                    is_distributed=self.args.is_distributed,
                                    local_rank=self.args.local_rank if self.args.is_distributed else None,
                                    cross_modality=self.args.cross_modality)
        # 网络模型初始化后为不同参数配置学习率，['offset_conv.weight', 'offset_conv.bias'] 为 0.1*学习率，其他的正常
        self.optimizer = _prepare_optimizer(self.cfg.OPTIMIZER, self.model)
        # 生成优化器后，配置学习率调度器
        self.scheduler = _prepare_scheduler(self.cfg.SCHEDULER, self.optimizer)


    def train(self):
        # 设置记录器的训练 mode，用于创建 'train'/'test' 文件
        if self.args.is_master:
            self.logger.train()
                    
        # 生成数据集和 dataloader
        train_loader = self.get_train_loader(args=self.args,
                                             dataset_cfg=self.cfg.DATASET.TRAIN,
                                             dataloader_cfg=self.cfg.DATALOADER.TRAIN,
                                             is_distributed=self.args.is_distributed)
        
        # 从当前 epoch (一般从 1 开始) 遍历到总 epoch
        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            # 分布式训练
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch) # 仅对分布式训练的 DistributedSampler

            # 1 个 epoch 训练，对所有数据进行 1 次训练，记录并返回这次 epoch 的整体平均误差指标
            train_log_dict = self.method.train(model=self.model,
                                               data_loader=train_loader,
                                               optimizer=self.optimizer,
                                               is_distributer=self.args.is_distributed,
                                               world_size=self.args.world_size,
                                               cross_modality=self.args.cross_modality)
            
            # 调度器将学习率随训练周期调整
            self.scheduler.step()
            # 当前 current_epoch++，用于记录 checkpoint
            self.current_epoch += 1
            # 一次 epoch 训练后，记录该 epoch 的训练结果指标，更新 checkpoint 并存储
            if self.args.is_master:
                self._log_after_epoch(epoch+1, train_log_dict, 'train') # epoch + 1 因为从 epoch=0 开始，但记录为 epoch=1
    

    def train_test(self):
        # 设置记录器的训练 mode，用于创建 'train'/'test' 文件
        if self.args.is_master:
            self.logger.train()
                    
        # 生成数据集和 dataloader
        train_loader = self.get_train_loader(args=self.args,
                                             dataset_cfg=self.cfg.DATASET.TRAIN,
                                             dataloader_cfg=self.cfg.DATALOADER.TRAIN,
                                             is_distributed=self.args.is_distributed)
        
        test_loader = self.get_test_loader(args=self.args,
                                               dataset_cfg=self.cfg.DATASET.TEST,
                                               dataloader_cfg=self.cfg.DATALOADER.TEST)
        
        # 从当前 epoch (一般从 1 开始) 遍历到总 epoch
        for epoch in range(self.current_epoch, self.cfg.TOTAL_EPOCH):
            # 分布式训练
            if self.args.is_distributed:
                dist.barrier()
                train_loader.sampler.set_epoch(epoch) # 仅对分布式训练的 DistributedSampler

            # 1 个 epoch 训练，对所有数据进行 1 次训练，记录并返回这次 epoch 的整体平均误差指标
            train_log_dict = self.method.train(model=self.model,
                                               data_loader=train_loader,
                                               optimizer=self.optimizer,
                                               is_distributer=self.args.is_distributed,
                                               world_size=self.args.world_size,
                                               cross_modality=self.args.cross_modality)
            
            # 调度器将学习率随训练周期调整
            self.scheduler.step()
            # 当前 current_epoch++，用于记录 checkpoint
            self.current_epoch += 1
            # 一次 epoch 训练后，记录该 epoch 的训练结果指标，更新 checkpoint 并存储
            if self.args.is_master:
                self._log_after_epoch(epoch+1, train_log_dict, 'train') # epoch + 1 因为从 epoch=0 开始，但记录为 epoch=1

            if  self.current_epoch%2 == 0:
                test_log_dict = self.method.test(model=self.model,
                                                    data_loader=test_loader,
                                                    cross_modality=self.args.cross_modality)
                
                if self.args.is_master:
                    self._log_after_epoch(epoch+1, test_log_dict, 'validation', False)


    # 将一次 epoch 训练后的 模型/优化器/参数 存入字典，生成 checkpoint
    def _make_checkpoint(self):
        checkpoint = {
            'epoch': self.current_epoch,
            'args': self.args,
            'cfg': self.cfg,
            'model': self.model.module.state_dict(),    # 区分 self.model.module 和 ..modules
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        return checkpoint

    # 一次 epoch 训练后，记录该 epoch 的训练结果指标，更新 checkpoint 并存储
    # log_dict：存储该 epoch 平均 error 的字典，part：'train'//'test'
    def _log_after_epoch(self, epoch, log_dict, part, save_checkpoint=True):
        log = 'Epoch: %d | ' % epoch
        # %5s 为长度 >= 5 的字符串
        log += '%5s' % part
        for key in log_dict.keys():
            # log='train | key: log_dict[key]'
            log += ' | %s: %s' % (key, str(log_dict[key]))
            # 将每个 epoch 的训练结果指标通过 tensor_board 记录
            if isinstance(log_dict[key], SummationMeter) or isinstance(log_dict[key], Metric):
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key].value, epoch)
            else:
                self.logger.add_scalar('%s/%s' % (part, key), log_dict[key], epoch)
            # 将每个 epoch 的训练结果指标写入 'train_log.txt'
        self.logger.write(log=log)

        if save_checkpoint:
            # 将训练的 模型/优化器/参数 存入字典，生成 checkpoint
            checkpoint = self._make_checkpoint()
            #  利用 torch.save 方法，将每个 epoch 后的 checkpoint 保存至指定位置，实现实时更新
            self.logger.save_checkpoint(checkpoint=checkpoint, name='final.pth')
            # 每 25 epoch 后，存储一次，和刚刚的路径不同
            if epoch % self.args.save_term == 0:
                self.logger.save_checkpoint(checkpoint, '%d.pth' % epoch)
    
    # 指定路径读取 checkpoint，配置模型
    def load(self, name):
        checkpoint = self.logger.load_checkpoint(name)
        # self._init_from_cfg(checkpoint['cfg'])
        self.model.module.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        # # 由于存储的是训练后 +1 的 epoch，因此直接读取即可，不需要额外 +1 读取
        self.current_epoch = checkpoint['epoch']

    # 指定路径读取 checkpoint，配置模型
    def load_concentration(self, name):
        checkpoint = self.logger.load_checkpoint(name)
        module_a_state_dict = {k.replace('concentration_net.', ''): v for k, v in checkpoint['model'].items() if k.startswith('concentration_net.')}
        self.model.module.concentration_net.load_state_dict(module_a_state_dict)
        for param in self.model.module.concentration_net.parameters():
            param.requires_grad = False
        

# 生成优化器后，配置调度器
def _prepare_scheduler(scheduler_cfg, optimizer):
    name = scheduler_cfg.NAME  # CosineAnnealingWarmupRestarts
    parameters = scheduler_cfg.PARAMS

    if name == 'CosineAnnealingWarmupRestarts':
        from utils.scheduler import CosineAnnealingWarmupRestarts
        scheduler = CosineAnnealingWarmupRestarts(optimizer, **parameters)
    else:
        scheduler = getattr(optim.lr_scheduler, name)(optimizer, **parameters)

    return scheduler


# 网络模型初始化后配置优化器，主要为不同参数配置不同学习率,
# 网络中特殊参数 ['offset_conv.weight', 'offset_conv.bias'] 学习率为 0.1 倍输入学习率，其他的参数正常
def _prepare_optimizer(optimizer_cfg, model):
    parameters = optimizer_cfg.PARAMS
    learning_rate = parameters.lr  # 0.0005
    
    # 网络模型初始化后为不同参数配置学习率，list=[{'params': 特殊参数, 'lr': 0.1 * 学习率}, {'params': 其他参数, 'lr': 学习率}]
    # 特殊参数为 ['offset_conv.weight', 'offset_conv.bias']，其他的参数正常
    params_group = model.module.get_params_group(learning_rate)
    # 从 pytorch 中生成 Adam 优化器
    optimizer = optim.Adam(params_group, **parameters)
    
    return optimizer


# 初始化模型
def _prepare_model(model_cfg, is_distributed=False, local_rank=None, cross_modality=False):
    # 生成模型
    if not cross_modality:
        model = event_stereo.EventStereoMatchingNetwork(**model_cfg.PARAMS)
    else:
        model = event_encode_frame_stereo.EventFrameEncodeStereoMatchingNetwork(**model_cfg.PARAMS)

    if is_distributed:
        # 分布式训练
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        # 不是分布式训练，则使用数据并行，将模型放到GPU上
        model = nn.DataParallel(model).cuda()  
    
    return model
import os

import torch.nn as nn
from utils.logger import ExpLogger
from components.datasets import dsec
from components.model import event_frame_stereo_houglass
import method

class DLManager:
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg =cfg
        self.logger = ExpLogger(save_root=args.save_root) if args.is_master else None
        if self.cfg is not None:
            self._init_from_cfg(cfg)
        self.current_epoch = 0
        if self.args.is_continue_train:
            self.load(name=self.args.continue_path)

    def _init_from_cfg(self, cfg):
        assert cfg is not None
        self.cfg = cfg

        self.get_test_loader = dsec.get_dataloader
        self.method = method

        self.model = _prepare_model(self.cfg.MODEL)
        self.model = self.model.to('cuda:0')
        self.model = nn.DataParallel(self.model, device_ids=[0, 1])

    
    def test_all(self, test_time=False):
        if self.args.is_master:
            self.logger.train()
                    
        test_loader = self.get_test_loader(args=self.args,
                                               dataset_cfg=self.cfg.DATASET.TESTALL,
                                               dataloader_cfg=self.cfg.DATALOADER.TESTALL)

        for sequence_dataloader in test_loader:
                sequence_name = sequence_dataloader.dataset.sequence_name
                sequence_pred_list = self.method.testall(model=self.model,
                                                      test_time=test_time,
                                                      data_loader=sequence_dataloader,
                                                      cross_modality=self.args.cross_modality)

                for cur_pred_dict in sequence_pred_list:
                    file_name = cur_pred_dict.pop('file_name')
                    for key in cur_pred_dict:
                        self.logger.save_visualize(image=cur_pred_dict[key],
                                                   visual_type=key,
                                                   sequence_name=os.path.join('test', sequence_name),
                                                   image_name=file_name)


    
    def load(self, name):
        checkpoint = self.logger.load_checkpoint(name)
        self.model.module.load_state_dict(checkpoint['model'])
        checkpoint = {
            'model': self.model.module.state_dict()
        }
        self.logger.save_checkpoint(checkpoint=checkpoint, name='final.pth')
        


def _prepare_model(model_cfg):
    model = event_frame_stereo_houglass.EventFrameStereoNetwork(**model_cfg.PARAMS)
    return model

from yacs.config import CfgNode as CN

# 读取配置文件
def get_cfg(cfg_path):
    cfg = CN(new_allowed=True) # new_allowed=True表示可以添加新的键
    cfg.merge_from_file(cfg_path) # 读取文件cfg_path里的配置，并将其合并到cfg对象中
    cfg.freeze() # 冻结配置，之后不能再修改

    return cfg
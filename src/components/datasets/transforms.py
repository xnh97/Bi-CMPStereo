import numpy  as  np
from .event import transforms as event_transform
from .disparity import transforms as disparity_transform
from .image import transforms as image_transform
from .event_voxel import transforms as event_voxel_transform

# 对于传入的 tranforms 进行级联
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    # 相当于 () 操作，依次执行传入的预处理操作链 transforms 中的每一个操作
    # 通过 t = Compose(~) 定义后，使用 t(input) 调用 __call__
    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

# 将视差图和事件 stack 从 numpy 数组转为 pytorch 的张量，输入 sample {'disparity': np, 'event':'left':np,'right':np}
# 其中事件维度顺序改变：H*W*1/2*10*1 ——> 1*H*W*1/2*10
class ToTensor:
    def __init__(self):
        self.event_transform = event_transform.ToTensor()
        self.disparity_transform = disparity_transform.ToTensor()
        self.image_transform = image_transform.ToTensor()
        self.event_voxel_transform = event_voxel_transform.ToTensor()
        
    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])
        if 'disparity' in sample.keys():
            sample['disparity'] = self.disparity_transform(sample['disparity'])
        if 'image' in sample.keys():
            sample['image'] = self.image_transform(sample['image'])
        if 'event_voxel' in sample.keys():
            sample['event_voxel'] = self.event_voxel_transform(sample['event_voxel'])
        return sample
    
# train 时以随即位置为起点，进行事件 stack 和视差图的剪裁
# 性能提升：通过对训练图像进行随机剪裁，可以训练模型去学习识别图片不同区域的特征，而不仅仅是中间或者特定部分的特征
# 防止过拟合：随机剪裁可以被看做一种正则化技术，有助于防止模型过度依赖训练集的特定特征，防止过拟合现象的发生
# 数据增强：通过随机剪裁，我们可以从每个图像中获得多个不同的样本，有助于扩大训练数据集，进一步提高模型的性能
class RandomCrop:
    def __init__(self, crop_height, crop_width):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.event_transform = event_transform.Crop(crop_height, crop_width)
        self.disparity_transform = disparity_transform.Crop(crop_height, crop_width)
        self.image_transform = image_transform.Crop(crop_height, crop_width)
        self.event_voxel_transform = event_voxel_transform.Crop(crop_height, crop_width)

    def __call__(self, sample):
        # 高/宽
        if 'event' in sample.keys():
            if 'left' in sample['event'].keys():
                ori_height, ori_width = sample['event']['left'].shape[:2]
            else:
                ori_height, ori_width = sample['event']['right'].shape[:2]
        else:
            raise NotImplementedError
        assert self.crop_height <= ori_height and self.crop_width <= ori_width

        # 生成一个介于0和ori_width - self.crop_width（包含）之间的随机整数
        offset_x = np.random.randint(ori_width - self.crop_width +1)
        offset_y = np.random.randint(ori_height - self.crop_height +1)

        # 从随即位置进行事件和视差图的剪裁
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'], offset_x, offset_y)
        if 'disparity' in sample.keys():
            sample['disparity'] = self.disparity_transform(sample['disparity'], offset_x, offset_y)
        if 'image' in sample.keys():
            sample['image'] = self.image_transform(sample['image'], offset_x, offset_y)
        if 'event_voxel' in sample.keys():
            sample['event_voxel'] = self.event_voxel_transform(sample['event_voxel'], offset_x, offset_y)
        return sample
    
# test 填充视差图和事件 stack，控制输入尺寸（也是 480*640）
class Padding:
    def __init__(self, img_height, img_width, no_event_value=0, no_disparity_value=0, no_image_value=0):
        self.img_height = img_height
        self.img_width = img_width
        self.event_transform = event_transform.Padding(img_height, img_width, no_event_value)
        self.disparity_transform = disparity_transform.Padding(img_height, img_width, no_disparity_value)
        self.image_transform = image_transform.Padding(img_height, img_width, no_image_value)
        self.event_voxel_transform = event_voxel_transform.Padding(img_height, img_width, no_event_value)

    def __call__(self, sample):
        if 'event' in sample.keys():
            sample['event'] = self.event_transform(sample['event'])
        if 'disparity' in sample.keys():
            sample['disparity'] = self.disparity_transform(sample['disparity'])
        if 'image' in sample.keys():
            sample['image'] = self.image_transform(sample['image'])
        if 'event_voxel' in sample.keys():
            sample['event_voxel'] = self.event_voxel_transform(sample['event_voxel'])
        return sample
    
    
# train 时进行随机会翻转（50%概率）
# 提高模型鲁棒性：如果在图像中的实际应用场景中，图像可能会出现多种方向，那么这种翻转可以增加模型对于镜像变换的鲁棒性
class RandomVerticalFlip:
    def __init__(self):
        self.event_transform = event_transform.VerticalFlip()
        self.disparity_transform = disparity_transform.VerticalFlip()
        self.image_transform = image_transform.VerticalFlip()
        self.event_voxel_transform = event_voxel_transform.VerticalFlip()

    def __call__(self, sample):
        if np.random.random() < 0.5:
            if 'event' in sample.keys():
                sample['event'] = self.event_transform(sample['event'])
            if 'disparity' in sample.keys():
                sample['disparity'] = self.disparity_transform(sample['disparity'])
            if 'image' in sample.keys():
                sample['image'] = self.image_transform(sample['image'])
            if 'event_voxel' in sample.keys():
                sample['event_voxel'] = self.event_voxel_transform(sample['event_voxel'])
        return sample
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Normalize, Resize
from .randaug import RandAugment
from .utils import MinMaxResize, MaxWHResize

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESIZE = {'normal': Resize, 'minmax': MinMaxResize, 'maxwh': MaxWHResize}


def denormalize():
    dmean = [-m / s for m, s in zip(MEAN, STD)]
    dstd = [1 / s for s in STD]
    return Compose([Normalize(mean=dmean, std=dstd), transforms.ToPILImage()])


def normalize():
    return transforms.Normalize(mean=MEAN, std=STD)


def get_transform(cfg):
    resize = RESIZE[cfg.resize_name](cfg.size)  # [384, 640]
    if cfg.randaug:
        return {
            'train': Compose([resize, RandAugment(), ToTensor(), normalize()]),
            # resize: 调整图像大小。
            # RandAugment(): 应用随机增强，这是一种数据增强技术，通过随机选择和应用一系列增强操作来增加图像的多样性。
            # ToTensor(): 将图像转换为 PyTorch 张量。
            # normalize: 归一化图像，通常是将图像像素值缩放到 [0, 1] 或者进行标准化。
            'valid': Compose([resize, ToTensor(), normalize()]),
        }
    else:
        return {
            'train': Compose([resize, ToTensor(), normalize()]),
            'valid': Compose([resize, ToTensor(), normalize()]),
        }

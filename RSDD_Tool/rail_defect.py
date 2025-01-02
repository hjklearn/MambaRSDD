import torch as t
import os                                                           #系统库（获取图片地址）
from torch.utils.data import Dataset, DataLoader                    #工具箱里关于数据的工具
import cv2
from PIL import Image                                                #读取图片
import matplotlib                                                    #绘图工具
import numpy as np                                                   #图像处理
import random
import torchvision                                                   #图形库
from glob import glob                                                #从glob模块导入glob函数
import matplotlib.pyplot as plt

image_h = 320
image_w = 320



class RDDataset(Dataset):
    def __init__(self, root_path, method='train'):
        self.rootpath = root_path
        self.method = method
        self.data_path = os.path.join(self.rootpath, method)

        self.rgb_img_path = os.path.join(self.data_path, 'rgb')
        self.rgb_img_item = sorted(glob(os.path.join(self.rgb_img_path, '*.bmp')))
        self.depth_img_path = os.path.join(self.data_path, 'd')                                  #获取地址
        self.depth_img_item = sorted(glob(os.path.join(self.depth_img_path, '*.tiff')))
        self.gt_img_path = os.path.join(self.data_path, 'gt')
        self.gt_img_item = sorted(glob(os.path.join(self.gt_img_path, '*.png')))
        if method not in ['train', 'value', 'test']:
            raise Exception("not implement")
        if method == 'train':
            self.transform = torchvision.transforms.Compose([                                #串联多个图片变换。遍历
                scaleNorm(),
                RandomHSV((0.9, 1.1), (0.9, 1.1), (25, 25)),
                RandomFlip(),
                ToTensor(),
                Normalize()
            ])
        if method == 'value':
            self.transform = torchvision.transforms.Compose([
                scaleNorm(),
                ToTensor(),
                Normalize(),
            ])
        if method == 'test':
            self.transform = torchvision.transforms.Compose([
                scaleNorm(),
                ToTensor(),
                Normalize(),
            ])

    def __getitem__(self, index):                                                    #获取每一个图片
        rgb_item = self.rgb_img_item[index]                                          #从rgb_img_item这里读取一个图片
        depth_item = self.depth_img_item[index]                                      #
        gt_item = self.gt_img_item[index]                                            #
        rgb = Image.open(rgb_item)                                                   #读取图片
        rgb = np.asarray(rgb)                                                        #将结构数据转化为n维数组对象（矩阵）
        depth = Image.open(depth_item)
        depth = np.asarray(depth)
        gt = Image.open(gt_item)
        gt = np.asarray(gt).astype(float)
        if gt.max() == 255.:
            gt = gt / 255.
        if self.method == 'train' or 'value':
            sample = {'RGB': rgb, 'depth': depth, 'label': gt}
        if self.method == 'test':
            name = rgb_item.split('/')[-1].split('.')[-2]                           #
            sample = {'RGB': rgb, 'depth': depth, 'label': gt, 'name': name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):                                                                #数据集长度返回
        return len(self.rgb_img_item)


class RDDatasetwithbound(Dataset):
    def __init__(self, root_path, method='train'):
        self.rootpath = root_path
        self.method = method
        self.data_path = os.path.join(self.rootpath, method)
        self.rgb_img_path = os.path.join(self.data_path, 'rgb')
        self.rgb_img_item = sorted(glob(os.path.join(self.rgb_img_path, '*.bmp')))
        self.depth_img_path = os.path.join(self.data_path, 'depth_anything_v2')
        self.depth_img_item = sorted(glob(os.path.join(self.depth_img_path, '*.png')))
        self.gt_img_path = os.path.join(self.data_path, 'gt')
        self.gt_img_item = sorted(glob(os.path.join(self.gt_img_path, '*.png')))
        if method not in ['train', 'value', 'Test_model']:
            raise Exception("not implement")
        if method == 'train':
            self.transform = torchvision.transforms.Compose([
                scaleNorm(),
                RandomHSV((0.9, 1.1),(0.9, 1.1),(25, 25)),
                RandomFlip(),
                ToTensor(),
                Normalize()
            ])
        if method == 'value':
            self.transform = torchvision.transforms.Compose([
                scaleNorm(),
                ToTensor(),
                Normalize(),
            ])
        if method == 'Test_model':
            self.transform = torchvision.transforms.Compose([
                scaleNorm(),
                ToTensor(),
                Normalize(),
            ])

    def __getitem__(self, index):
        rgb_item = self.rgb_img_item[index]
        print(rgb_item)
        depth_item = self.depth_img_item[index]
        gt_item = self.gt_img_item[index]
        rgb = Image.open(rgb_item)
        rgb = np.asarray(rgb)
        depth = Image.open(depth_item)
        depth = np.asarray(depth)
        gt = Image.open(gt_item)
        gt = np.asarray(gt).astype(float)
        if gt.max() == 255.:
            gt = gt / 255.
        if self.method == 'train' or 'value':
            sample = {'RGB': rgb, 'depth': depth, 'label': gt}
        if self.method == 'Test_model':
            name = rgb_item.split('/')[-1].split('.')[-2]
            sample = {'RGB': rgb, 'depth': depth, 'label': gt, 'name': name}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.rgb_img_item)

class scaleNorm(object):
    def __init__(self, image_h=320, image_w=320):
        self.image_h = image_h
        self.image_w = image_w

    def __call__(self, sample):
        if len(sample) == 3:
            image, depth, label = sample['RGB'], sample['depth'], sample['label']                   #缩放，差值重新计算图像
            image = cv2.resize(image, (self.image_h, self.image_w), interpolation=cv2.INTER_LINEAR)   # 双线性插值/上采样
            depth = cv2.resize(depth, (self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)  #最邻近插值
            label = cv2.resize(label, (self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)  # Nearest-neighbor
            return {'RGB': image, 'depth': depth, 'label': label}
        if len(sample) == 4:                             #此算法减少了由于将图像调整大小为非整数缩放因子而导致的某些视觉失真
            image, depth, label, name = sample['RGB'], sample['depth'], sample['label'], sample['name']
            image = cv2.resize(image, (self.image_h, self.image_w), interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, (self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (self.image_h, self.image_w), interpolation=cv2.INTER_NEAREST)
            name = name
            return {'RGB': image, 'depth': depth, 'label': label, 'name': name}
class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range

    def __call__(self, sample):
        img = sample['RGB']
        img_hsv = matplotlib.colors.rgb_to_hsv(img)                                      #rgb变为hsv
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))              #随机采样
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)                                       #截取矩阵中范围0-1
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)                     #堆叠：给数组升维 第二维开始
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
        return {'RGB': img_new, 'depth': sample['depth'], 'label': sample['label']}
class RandomFlip(object):                                                                     #左右翻转
    def __call__(self, sample):
        image, depth, label = sample['RGB'], sample['depth'], sample['label']
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            depth = np.fliplr(depth).copy()
            label = np.fliplr(label).copy()
        return {'RGB': image, 'depth': depth, 'label': label}

class Normalize(object):                                                      #归一化，使数据服从正态分布
    def __call__(self, sample):
        if len(sample) == 3:
            image, depth, label = sample['RGB'], sample['depth'], sample['label']
            image = image / 255.0
            image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],   #用均值和标准差对张量图像进行归一化
                                                     std=[0.229, 0.224, 0.225])(image)
            # if depth.max() > 256.0:
            #     depth = depth / 31197.0
            # else:
            depth = depth / 255.0
            # depth = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                          std=[0.229, 0.224, 0.225])(depth)
            label = label
            sample['RGB'] = image
            sample['depth'] = depth
            sample['label'] = label
            return sample
        if len(sample) == 4:                                               #
            image, depth, label, name = sample['RGB'], sample['depth'], sample['label'], sample['name']
            image = image / 255.0
            image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])(image)
            # if depth.max() > 256.0:
            #     depth = depth / 31197.0
            # else:
            depth = depth / 255.0
            # depth = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                          std=[0.229, 0.224, 0.225])(depth)
            label = label
            name = name
            sample['RGB'] = image
            sample['depth'] = depth
            sample['label'] = label
            sample['name'] = name
            return sample
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if len(sample) == 3:
            image, depth, label = sample['RGB'], sample['depth'], sample['label']
            # swap color axis because
            # numpy RGB: H x W x C
            # torch RGB: C X H X W
            image = image.transpose((2, 0, 1))                                      #图像格式转换

            depth = np.array([depth])
            depth = depth / 1.0
            # depth = depth.transpose((2, 0, 1))
            # label = np.expand_dims(label, 0).astype(np.float)
            label = np.expand_dims(label, 0).astype(float)                         #扩展数组的形状。

            return {'RGB': t.from_numpy(image).float(),
                    'depth': t.from_numpy(depth).float(),
                    'label': t.from_numpy(label).float(), }
        if len(sample) == 4:
            image, depth, label, name = sample['RGB'], sample['depth'], sample['label'], sample['name']
            # swap color axis because
            # numpy RGB: H x W x C
            # torch RGB: C X H X W
            image = image.transpose((2, 0, 1))
            # depth = depth.transpose((2, 0, 1))
            depth = np.array([depth])
            depth = depth / 1.0
            label = np.expand_dims(label, 0).astype(float)
            return {'RGB': t.from_numpy(image).float(),
                    'depth': t.from_numpy(depth).float(),
                    'label': t.from_numpy(label).float(),
                    'name': name
                    }


if __name__ == '__main__':
    trainDatasets = RDDataset(rootpath, 'train')
    valDatasets = RDDataset(rootpath, 'val')
    testDatasets = RDDataset(rootpath, 'Test_model')
    sample = trainDatasets[100]
    # sample = valDatasets[100]
    # sample = testDatasets[100]
    l1 = sample['label']
    img = sample['RGB']
    depth = sample['depth']
    # name = sample['name']
    print('label', l1.shape)
    # print('name', name)
    # img1 = torchvision.transforms.ToPILImage()(img)
    img2 = torchvision.transforms.ToPILImage()(depth)
    # plt.imshow(img1)
    plt.imshow(img2)
    plt.waitforbuttonpress()




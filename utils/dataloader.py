# -*- coding:utf-8 -*-
# @Time: 2023-9-7 14:02
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: dataloader.py
# @ProjectName: PolypNet
import os

from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import numpy as np


# ---- Several data augmentation ----
def cv_random_clip(img, gt):
    # left and right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
    return img, gt


def randomCrop(img, gt):
    border = 30
    img_width = img.size[0]
    img_height = img.size[1]
    cropped_width = np.random.randint(img_width - border, img_width)
    cropped_height = np.random.randint(img_height - border, img_height)
    cropped_region = (
        (img_width - cropped_width) >> 1, (img_height - cropped_height) >> 1,
        (img_width + cropped_width) >> 1, (img_height + cropped_height) >> 1
    )
    return img.crop(cropped_region), gt.crop(cropped_region)


def randomRotation(img, gt):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        img = img.rotate(random_angle, mode)
        gt = gt.rotate(random_angle, mode)
    return img, gt


def colorEnhance(img):
    bright_intensity = random.randint(5, 15) / 10
    img = ImageEnhance.Brightness(img).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10
    img = ImageEnhance.Contrast(img).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10
    img = ImageEnhance.Color(img).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10
    img = ImageEnhance.Sharpness(img).enhance(sharp_intensity)
    return img


def randomGaussian(img, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(img)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    # print('P2: ', img.dtype())
    img = np.array(img)
    # print('P3: ', type(img))
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


def get_loader(img_root,
               gt_root,
               batch_size,
               train_size,
               shuffle=True,
               num_workers=12,
               pin_memory=True,
               prefetch_factor=4):
    dataset = CamObjDataset(img_root, gt_root, train_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  prefetch_factor=prefetch_factor)
    return data_loader


# ---- Train Dataset ----
class CamObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, train_size):
        self.train_size = train_size

        # get fileanme
        self.image_list = [image_root + f for f in os.listdir(image_root) if f.endswith('jpg') or f.endswith('png')]
        self.gt_list = [gt_root + f for f in os.listdir(gt_root) if f.endswith('jpg') or f.endswith('png')]

        # sort file
        self.image_list = sorted(self.image_list)
        self.gt_list = sorted(self.gt_list)

        # # choose sample
        # self.image_list = self.image_list[:50]  # TODO
        # self.gt_list = self.gt_list[:50]

        # filter files
        self.filter_files()

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.train_size, self.train_size)),
            transforms.ToTensor()  # 将数值压缩至[0, 1]
        ])

        # get len of dataset
        self.dataset_size = len(self.image_list)
        print('>>> Training/validing with {} samples'.format(self.dataset_size))

    def __getitem__(self, index):
        image = self.rgb_loader(self.image_list[index])
        gt = self.binary_loader(self.gt_list[index])

        # data augmentation
        image, gt = cv_random_clip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)

        image = colorEnhance(image)
        # print('P1: ', image)
        gt = randomPeper(gt)

        image = self.img_transforms(image)
        gt = self.gt_transforms(gt)

        return image, gt

    def __len__(self):
        return self.dataset_size

    def rgb_loader(self, path):
        # print(path)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            gt = Image.open(f)
            return gt.convert('L')

    def filter_files(self):
        assert len(self.image_list) == len(self.gt_list)
        images = []
        gts = []
        for img_path, gt_path in zip(self.image_list, self.gt_list):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)  # 这边相当于把图像打开再检查一遍，没问题就把path存进去，不是存的PIL！！！
                gts.append(gt_path)
            else:
                print('Size miss match!', img_path)
            self.image_list = images
            self.gt_list = gts
        # print('self.image_list[0] in filter_files:', self.image_list[0])

    def resize(self, img, gt):
        assert img.size == gt.size
        W, H = img.size
        if H < self.train_size or W < self.train_size:
            H = max(H, self.train_size)
            W = max(W, self.train_size)
            return img.resize((W, H), Image.BILINEAR), gt.resize((W, H), Image.BILINEAR)
        else:
            return img, gt


# === Test Dataset
class test_dataset:
    def __init__(self, image_root, gt_root, test_size):
        self.test_size = test_size
        self.image_list = [image_root + f for f in os.listdir(image_root) if f.endswith('jpg') or f.endswith('png')]
        self.gt_list = [gt_root + f for f in os.listdir(gt_root) if f.endswith('jpg') or f.endswith('png')]
        self.image_list = sorted(self.image_list)
        self.gt_list = sorted(self.gt_list)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),  # Normalize需要加在ToTensor之后
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()  # 测试阶段不能改变label
        self.size = len(self.image_list)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.image_list[self.index])
        image = self.image_transform(image).unsqueeze(0)  # 这边image做了resize变换，尺寸压缩，变成和训练一样的格式

        gt = self.binary_loader(self.gt_list[self.index])  # 这边没有做transform，只通过PIL打开

        name = self.image_list[self.index].split('/')[-1]

        image_for_post = self.rgb_loader(self.image_list[self.index])
        image_for_post = image_for_post.resize(gt.size)  # 不做变换，只通过PIL打开

        # 这边没看懂，为啥要从jpg转到png
        # 解释：这边读到的name后面用于保存predict map，所以需要把.jpg转换成.png
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

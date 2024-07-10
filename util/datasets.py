# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import os, random

# import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

random.seed(42)

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, "train" if is_train else "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


## pretrain
class SeismicSet(data.Dataset):
    def __init__(self, path, input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.file_list = [os.path.join(path, f) for f in os.listdir(path)]
        random.shuffle(self.file_list)
        print("self.file_list", len(self.file_list))

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        d = self.load_data(file_path)
        return d, torch.tensor([1])

    def load_data(self, file_path):
        with open(file_path, "rb") as f:
            d = np.fromfile(f, dtype=np.float32)
            d = d.reshape(1, self.input_size, self.input_size)
            d = (d - d.mean()) / (d.std() + 1e-6)
        return d

class SeismicSet_singleFile(data.Dataset):
    def __init__(self, path, input_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.dat_path = path
        self.read_len = input_size * input_size * np.single().itemsize 
        with open(self.dat_path, 'rb') as binary_file:
            binary_file.seek(0, 2)
            file_length = binary_file.tell()
            assert file_length % self.read_len == 0, f"The file length is not divisible by read_len"
            self.img_count = int(file_length / self.read_len)
        self.shuffle_idx = np.arange(self.img_count)
        random.shuffle(self.shuffle_idx)
        self.fopen = open(self.dat_path, 'rb')
        print("len(self.file_list)", self.img_count)

    def __len__(self) -> int:
        return self.img_count

    def __getitem__(self, index):
        self.fopen.seek(self.read_len * self.shuffle_idx[index])
        chunk = self.fopen.read(self.read_len)  
        d = np.frombuffer(chunk, dtype=np.single()).reshape(1, self.input_size, self.input_size)
        d = (d - d.mean()) / (d.std() + 1e-6)
        return d, torch.tensor([1])
    


def to_transforms(d, input_size):
    t = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                input_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    return t(d)


### fintune
class FacesSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self, folder, shape=[768, 768], is_train=True) -> None:
        super().__init__()
        self.shape = shape

        # self.data_list = sorted(glob.glob(folder + 'seismic/*.dat'))
        self.data_list = [folder + "seismic/" + str(f) + ".dat" for f in range(117)]

        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:100]
        elif not is_train:
            self.data_list = self.data_list[100:]
        self.label_list = [f.replace("/seismic/", "/label/") for f in self.data_list]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape) - 1
        l = l.astype(int)
        return torch.tensor(d), torch.tensor(l)

    def __len__(self):
        return len(self.data_list)


class SaltSet(data.Dataset):

    def __init__(self, folder, shape=[224, 224], is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.data_list = [folder + "seismic/" + str(f) + ".dat" for f in range(4000)]
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list[:3500]
        elif not is_train:
            self.data_list = self.data_list[3500:]
        self.label_list = [f.replace("/seismic/", "/label/") for f in self.data_list]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32).reshape(self.shape)
        l = l.astype(int)
        return torch.tensor(d), torch.tensor(l)

    def __len__(self):
        return len(self.data_list)


class InterpolationSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self, folder, shape=[224, 224], is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.data_list = [folder + str(f) + ".dat" for f in range(6000)]
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
        elif not is_train:
            self.data_list = [
                folder + "U" + +str(f) + ".dat" for f in range(2000, 4000)
            ]
        self.label_list = self.data_list

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        return torch.tensor(d), torch.tensor(d)

    def __len__(self):
        return len(self.data_list)
        # return 10000


class DenoiseSet(data.Dataset):
    def __init__(self, folder, shape=[224, 224], is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.data_list = [folder + "seismic/" + str(f) + ".dat" for f in range(2000)]
        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
            self.label_list = [
                f.replace("/seismic/", "/label/") for f in self.data_list
            ]
        elif not is_train:
            self.data_list = [folder + "field/" + str(f) + ".dat" for f in range(4000)]
            self.label_list = self.data_list

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d.reshape([1] + self.shape)
        # d = (d - d.mean())/d.std()
        l = np.fromfile(self.label_list[index], np.float32)
        l = l.reshape([1] + self.shape)
        # l = (l - d.mean())/l.std()
        return torch.tensor(d), torch.tensor(l)

    def __len__(self):
        return len(self.data_list)


class ReflectSet(data.Dataset):
    # folder/train/data/**.dat, folder/train/label/**.dat
    # folder/test/data/**.dat, folder/test/label/**.dat
    def __init__(self, folder, shape=[224, 224], is_train=True) -> None:
        super().__init__()
        self.shape = shape
        self.data_list = [folder + "seismic/" + str(f) + ".dat" for f in range(2200)]

        n = len(self.data_list)
        if is_train:
            self.data_list = self.data_list
            self.label_list = [
                f.replace("/seismic/", "/label/") for f in self.data_list
            ]
        elif not is_train:
            self.data_list = [
                folder + "SEAMseismic/" + str(f) + ".dat" for f in range(4000)
            ]
            self.label_list = [
                f.replace("/SEAMseismic/", "/SEAMreflect/") for f in self.data_list
            ]

    def __getitem__(self, index):
        d = np.fromfile(self.data_list[index], np.float32)
        d = d - d.mean()
        d = d / (d.std() + 1e-6)
        d = d.reshape([1] + self.shape)
        l = np.fromfile(self.label_list[index], np.float32)
        l = l - l.mean()
        l = l / (l.std() + 1e-6)
        l = l.reshape([1] + self.shape)
        return torch.tensor(d), torch.tensor(l)

    def __len__(self):
        return len(self.data_list)

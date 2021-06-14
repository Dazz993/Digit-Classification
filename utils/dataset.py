import os
import numpy as np
import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataset(path='../data/', preprocess=None):
    # define normalization and transforms
    normalize = transforms.Normalize(mean=[0.449, 0.450, 0.450], std=[0.200, 0.199, 0.199])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if preprocess is None:
        train_dataset = datasets.SVHN(path, split='train', transform=train_transform, download=False)
        test_dataset = datasets.SVHN(path, split='test', transform=test_transform, download=False)
    else:
        raise NotImplementedError

    return train_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset(path='../data/')
import os
import cv2
import random
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Data Preparation')
parser.add_argument('--data-dir', default='', type=str, metavar='PATH',
                    help='path to the image folders')

def main():
    args = parser.parse_args()
    means, stds = compute_image_parameters(args.data_dir)

    print(means, stds)

def compute_image_parameters(path, sample_rate=1.0):
    '''
    Compute means and standard deviations of images

    Args:
        path (str): path to the image folder
        resize_to (tuple, optional): the size we want to resize to. Defaults to (224, 224).
        sample_rate (float, optional): sample rate to compute. Defaults to 0.1.

    Returns:
        means (list of floats): means value in (0, 1)
        stds (list of floats): stds value in (0, 1)
    '''
    imgs = sio.loadmat(path)['X']
    print(f"==> Successfully load images. file path: {path}, num of images: {imgs.shape[-1]}")

    means, stds = [], []

    imgs = imgs.transpose(3, 2, 0, 1)
    imgs = np.random.permutation(imgs)[:int(len(imgs) * sample_rate)]
    imgs = imgs.astype(np.float32) / 255

    print(f"==> Calculating means and standard deviations")
    for i in tqdm(range(3)):
        means.append(np.mean(imgs[:, :, :, i]))
        stds.append(np.std(imgs[:, :, :, i]))

    return means, stds

if __name__ == '__main__':
    main()
import os
import cv2
import time
import yaml
import time
import argparse
import torch
import torch.utils.data
import numpy as np
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from tqdm import tqdm
from time import perf_counter as t
from models import LinearModel, SimpleConv, vgg, resnet

def cam(model, input_tensor, ori_image, target_layer, target_category):
    cam = EigenCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(ori_image, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    ori_image = cv2.cvtColor(np.uint8(ori_image * 255), cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'docs/figs/ori.png', ori_image)
    cv2.imwrite(f'docs/figs/cam_{time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}.png', cam_image)

def guided_propagation(model, input_tensor, target_category):
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=torch.cuda.is_available())
    gb = gb_model(input_tensor, target_category=target_category)
    gb = _deprocess_image(gb)

    cv2.imwrite(f'docs/figs/gb_{time.strftime("_%Y_%m_%d_%H_%M_%S", time.localtime())}.png', gb)

def _deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

if __name__ == '__main__':
    # define some parameters
    device = 'cuda'

    # load train images
    def get_images_and_labels(path):
        import scipy.io as sio
        data = sio.loadmat(path)
        images = data['X'].astype(float)
        images = images.transpose(3, 0, 1, 2)
        labels = data['y']
        np.place(labels, labels == 10, 0)
        return images, labels

    train_images, train_labels = get_images_and_labels('data/train_32x32.mat')
    test_images, test_labels = get_images_and_labels('data/test_32x32.mat')

    # define LogsiticRegreesion model
    print(f"==> Input size: {test_images[0].shape}")

    # Simple Conv
    # model = SimpleConv(num_classes=10).to(device)
    # state_dicts = torch.load('states/SimpleConv_0.001__2021_06_17_22_42_13/checkpoint_SimpleConv_best.tar')
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.features[-1]

    # VGG Net
    # model = vgg(layers=16, num_classes=10).to(device)
    # state_dicts = torch.load('states/VGG16_0.0001__2021_06_18_01_23_48/checkpoint_VGG_best.tar')
    # model.load_state_dict(state_dicts['state_dict'])
    # target_layer = model.features[-1]

    # ResNet
    model = resnet(layers=18, num_classes=10).to(device)
    state_dicts = torch.load('states/ResNet18_0.01__2021_06_18_09_19_52/checkpoint_ResNet_best.tar')
    model.load_state_dict(state_dicts['state_dict'])
    target_layer = model.layer4[-1]

    model.eval()

    img_idx = 2
    images, labels = train_images, train_labels
    # images, labels = test_images, test_labels
    ori_image = np.float32(images[img_idx]) / 255
    label = labels[img_idx]
    input_tensor = preprocess_image(ori_image, mean=[0.449, 0.450, 0.450], std=[0.200, 0.199, 0.199]).to(device)

    print(f"==> Input image id: {img_idx}, true label: {label}")

    cam(model, input_tensor=input_tensor, ori_image=ori_image, target_layer=target_layer, target_category=None)

    guided_propagation(model=model, input_tensor=input_tensor, target_category=None)

    print(f"==> Output label: {model(input_tensor).argmax(1).item()}")
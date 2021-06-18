'''
VGG implementation with PyTorch
Refer to
(1) https://arxiv.org/abs/1409.1556
(2) https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# Refer to https://arxiv.org/abs/1409.1556 Table 1 to see VGG configurations
vgg_architecture_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_type, num_classes, init_weights=True):
        super(VGG, self).__init__()
        assert vgg_type in vgg_architecture_cfg.keys()
        self.features = self._make_layers(vgg_architecture_cfg[vgg_type])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg):
        layers = []
        inplanes = 3
        for flag in cfg:
            if flag == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(flag, int):
                layers.append(nn.Conv2d(inplanes, flag, kernel_size=3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(flag))
                layers.append(nn.ReLU(inplace=True))
                inplanes = flag
            else:
                raise ValueError

        return nn.Sequential(*layers)

    # refer to https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.classifier(out)

        return out

def vgg11(num_classes=10):
    return VGG('VGG11', num_classes=num_classes)

def vgg13(num_classes=10):
    return VGG('VGG13', num_classes=num_classes)

def vgg16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes)

def vgg19(num_classes=10):
    return VGG('VGG19', num_classes=num_classes)

def vgg(layers, num_classes=10):
    if layers == 11:
        return vgg11(num_classes=num_classes)
    elif layers == 13:
        return vgg13(num_classes=num_classes)
    elif layers == 16:
        return vgg16(num_classes=num_classes)
    elif layers == 19:
        return vgg19(num_classes=num_classes)
    else:
        raise NotImplementedError
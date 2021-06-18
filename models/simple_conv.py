'''
Simple Conv implementation with PyTorch
Consists of 4 convolution layers and one linear layer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def ConvBatchRelu(inplanes, outplanes, kernel_size, stride, padding):
    conv_block = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    return conv_block


class SimpleConv(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConv, self).__init__()
        self.features = nn.Sequential(
            ConvBatchRelu(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            ConvBatchRelu(64, 128, kernel_size=(3, 3), stride=2, padding=1),
            ConvBatchRelu(128, 256, kernel_size=(3, 3), stride=2, padding=1),
            ConvBatchRelu(256, 512, kernel_size=(3, 3), stride=2, padding=1),
        )

        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(512 * 4 * 4, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)

        return out
import torch
import torch.nn as nn
from functools import reduce

class LinearModel(nn.Module):
    def __init__(self, input_size=(3, 32, 32), num_classes=10):
        super(LinearModel, self).__init__()
        n_elements = reduce(lambda x, y: x * y, input_size)
        self.regression = nn.Linear(n_elements, num_classes)

    def forward(self, x):
        out = x.flatten(1)
        out = self.regression(out)
        return out
import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, stride=1, padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.stride = stride
        self.padding = padding

        self.mask = torch.ones(kernel_size ** 2)
        self.mask[math.floor((kernel_size ** 2) / 2) if mask_type == "A" else
                  math.ceil((kernel_size ** 2) / 2):] = 0
        self.mask = nn.Parameter(self.mask.reshape(kernel_size, kernel_size), requires_grad=False)

        self.weights = nn.Parameter(torch.empty(size=(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True))
        nn.init.kaiming_uniform_(self.weights)

    def forward(self, x):
        return F.conv2d(x, self.weights * self.mask, padding=self.padding, stride=self.stride)

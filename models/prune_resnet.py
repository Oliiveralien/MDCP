# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal

import numpy as np

norm_mean, norm_var = 0.0, 1.0

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


  # """Return new masks that involve pruning the smallest of the final weights.
  #
  # Args:
  #   percents: A dictionary determining the percent by which to prune each layer.
  #     Keys are layer names and values are floats between 0 and 1 (inclusive).
  #   masks: A dictionary containing the current masks. Keys are strings and
  #     values are numpy arrays with values in {0, 1}.
  #   final_weights: The weights at the end of the last training run. A
  #     dictionary whose keys are strings and whose values are numpy arrays.
  #
  # Returns:
  #   A dictionary containing the newly-pruned masks.
  # """

def prune_by_percent_once(percent, mask, final_weight):
    # Put the weights that aren't masked out in sorted order.
    mask.cuda()
    final_weight.cuda()


    selected_weight=torch.masked_select(final_weight, mask, out=None)
    selected_weight.cuda()

    # Determine the cutoff for weights to be pruned.
    cutoff_index = int(np.ceil(percent * torch.numel(selected_weight)))


    cut_value,num = torch.kthvalue(torch.abs(selected_weight),cutoff_index,dim=0)
    cut_value.cuda()



    m = torch.zeros_like(mask, dtype=torch.uint8, layout=mask.layout, device=mask.device, requires_grad=False)
    return torch.where(torch.abs(final_weight) <= torch.abs(cut_value), m, mask)

class Mask(nn.Module):
    def __init__(self, init_value=[1], planes=None):
        super(Mask, self).__init__()
        self.planes = planes
        self.weight = Parameter(torch.Tensor(init_value))

    def forward(self, input):
        weight = self.weight

        if self.planes is not None:
            weight = self.weight[None, :, None, None]

        return input * weight


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class SparseResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(SparseResBasicBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))


        m = Normal(torch.tensor([norm_mean]), torch.tensor([norm_var])).sample()
        self.mask = Mask(m)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.mask(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, num_layers, num_classes=10, has_mask=None):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'  # 检查条件，不符合就终止程序
        n = (num_layers - 2) // 6

        if has_mask is None: has_mask = [1] * 3 * n  # ['1','1',...,'1']
        # self.count = 0
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)  # 输入通道，输出通道，卷积核，步长
        self.bn1 = nn.BatchNorm2d(self.inplanes)  # 正则化
        self.relu = nn.ReLU(inplace=True)  # inplace为True，将会改变输入的数据 (变成输出)，否则不会改变原输入，只会产生新的输出。

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1, has_mask=has_mask[0:n])
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2, has_mask=has_mask[n:2 * n])
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2, has_mask=has_mask[2 * n:3 * n])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight)
                nn.init.normal_(m.bias)
                # nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, has_mask):
        layers = []
        if has_mask[0] == 0 and (stride != 1 or self.inplanes != planes):
            layers.append(
                LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)))
        if not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        # print(x.size())128*64*1*1
        x = x.view(x.size(0), -1)
        # print(x.size())128*64
        x = self.fc(x)

        # print(x.size())
        # self.count = self.count + 1
        # print(self.count)

        return x


def resnet_56(**kwargs):
    return ResNet(ResBasicBlock, 56, **kwargs)


def resnet_56_sparse(**kwargs):
    return ResNet(SparseResBasicBlock, 56, **kwargs)


def resnet_110(**kwargs):
    return ResNet(ResBasicBlock, 110, **kwargs)


def resnet_110_sparse(**kwargs):
    return ResNet(SparseResBasicBlock, 110, **kwargs)

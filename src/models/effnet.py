"""EffNet 1D-CNN backbone used by SleepFM's per-modality encoders.

Vendored (not pip-installable) from the official SleepFM codebase
(https://github.com/rthapa84/sleepfm-codebase, MIT License, Copyright 2024
the SleepFM authors) so we don't depend on a `sys.path` hack into
third_party/. Only trimmed of unused commented-out code; the forward pass
and layer definitions are unchanged so that pretrained checkpoints load
without remapping.
"""
from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, expansion, activation, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_channel, in_channel * expansion, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channel * expansion, in_channel * expansion, kernel_size=3,
            groups=in_channel * expansion, padding=padding, stride=stride,
        )
        self.conv3 = nn.Conv1d(in_channel * expansion, out_channel, kernel_size=1, stride=1)
        self.b0 = nn.BatchNorm1d(in_channel * expansion)
        self.b1 = nn.BatchNorm1d(in_channel * expansion)
        self.d = nn.Dropout()
        self.act = activation()

    def forward(self, x):
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            return x + y
        y = self.act(self.b0(self.conv1(x)))
        y = self.act(self.b1(self.conv2(y)))
        return self.conv3(y)


class MBConv(nn.Module):
    def __init__(self, in_channel, out_channels, expansion, layers, activation=nn.ReLU6, stride=2):
        super().__init__()
        stack = OrderedDict()
        for i in range(layers - 1):
            stack[f"s{i}"] = Bottleneck(in_channel, in_channel, expansion, activation)
        stack[f"s{layers + 1}"] = Bottleneck(in_channel, out_channels, expansion, activation, stride=stride)
        self.stack = nn.Sequential(stack)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.stack(x))


class EffNet(nn.Module):
    """1D EfficientNet-style encoder. `fc` is replaced with a 512-dim linear
    layer by the caller to turn this into an embedding extractor (see
    src/models/embeddings.py), matching the official generate-embeddings script.
    """

    def __init__(
        self,
        in_channel,
        num_additional_features=0,
        depth=(1, 2, 2, 3, 3, 3, 3),
        channels=(32, 16, 24, 40, 80, 112, 192, 320, 1280),
        dilation=1,
        stride=2,
        expansion=6,
    ):
        super().__init__()
        self.stage1 = nn.Conv1d(in_channel, channels[0], kernel_size=3, stride=stride, padding=1, dilation=dilation)
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, 1)

    def forward(self, x):
        if self.num_additional_features > 0:
            x, additional = x
        x = self.b0(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.Pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.Pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.AAP(x)[:, :, 0])
        x = self.drop(x)
        if self.num_additional_features > 0:
            x = torch.cat((x, additional), 1)
        return self.fc(x)

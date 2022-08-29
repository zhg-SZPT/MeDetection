import numpy as np
import torch.nn as nn
import torch
from    torch.nn import functional as F
from util.spp_layer import  spatial_pyramid_pool
from util.coordatt import CoordAtt as ca
class Ours(nn.Module):
    def __init__(self):
        super().__init__()
        self.S_net = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.Conv2d(64, 64, 3, padding=(1, 1)),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 256 * 28 * 28
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 128, 3, padding=(1, 1)),
            nn.MaxPool2d((2, 2), padding=(1, 1)),  # pooling 256 * 28 * 28
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 256, 3, padding=(1, 1)),
            nn.Conv2d(256, 256, 3, padding=(1, 1)),
            nn.MaxPool2d((2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
        )
        self.C_net1 = nn.Sequential(
            nn.Conv2d(256, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), padding=(0, 0)),
        )
        self.C_net2 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), padding=(0, 0)),
        )
        self.C_net3 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), padding=(0, 0)),
        )
        self.C_net4 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), padding=(0, 0)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(672, 2)
        )
        self.ca = ca(32, 32)
        self.adjust = nn.Conv2d(3, 256, 3)
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)
    def forward(self, input1, input2):
        self.gradients = []
        output1 = self.S_net(input1)
        output2 = self.S_net(input2)
        output = torch.abs(output1 - output2)
        output=F.interpolate(output, size=[256, 256], mode="nearest")
        feature = self.C_net1(output)
        feature=self.ca(feature)
        feature = self.C_net2(feature)
        feature = self.C_net3(feature)
        feature = self.C_net4(feature)
        x = spatial_pyramid_pool(feature, feature.size(0), [feature.size(2), feature.size(3)], [4, 2, 1])
        result = self.classifier(x)
        return result
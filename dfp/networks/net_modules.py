from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math


class ImageModule(nn.Module):
    def __init__(self, img_shape):
        """
        Image Module for DFP
        """
        assert img_shape[0] > 0, "Channels must be greater than 0"
        super(ImageModule, self).__init__()
        # Image operation
        img_dims = reduce(lambda a, b: a * b, img_shape, 1)  # Find full dims of the image

        # The implementation uses 'SAME' padding which gives the weird padding numbers shown below
        self.image_base = nn.Sequential(
            nn.Conv2d(img_shape[0], 32,
                      kernel_size=8, stride=4, padding=2, bias=True),  # This might affect RGB performance
            nn.LeakyReLU(0.2)
        )

        self.image_base2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2)
        )

        output_depth = list(self.image_base.parameters())[-1].size()[0]

        # Linear layer after the convs
        self.image_fc = nn.Sequential(
            # Need to change this if the convs change
            nn.Linear(11 * 11 * 64, 512, bias=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, img):
        x = self.image_base(img)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.image_base2(x)
        x = x.view(x.size()[0], -1)  # flatten
        return self.image_fc(x)


class MapModule(nn.Module):
    def __init__(self, map_shape):
        """
        Image Module for DFP
        """
        assert map_shape[0] > 0, "Map channels must be greater than 0."
        super(MapModule, self).__init__()
        # Image operation
        map_dims = reduce(lambda a, b: a * b, map_shape, 1)  # Find full dims of the image

        self.init_convs = nn.ModuleList()
        for res in range(map_shape[0]):
            module = nn.Sequential(
                nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=True),
                nn.ReLU()
            )

            self.init_convs.append(module)

        # Linear layer after the convs
        dim = np.prod(map_shape[1:])
        self.map_fc = nn.Sequential(
            # Concatenate the original image, and whatever the convolutions spit out
            nn.Linear(dim * map_shape[0], 512, bias=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        vect = []
        for i in range(len(self.init_convs)):
            out = self.init_convs[i](x[:, i:i + 1])
            vect.append(out.view(x.size()[0], -1))

        # concat and flatten
        x = torch.cat(vect, 1).view(x.size()[0], -1)
        return self.map_fc(x)


class MeasModule(nn.Module):
    """
    Measurement module for DFP
    """

    def __init__(self, meas_dim):
        super(MeasModule, self).__init__()
        self.meas_base = nn.Sequential(
            nn.Linear(meas_dim, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128, bias=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, measurement):
        return self.meas_base(measurement)


class GoalModule(nn.Module):
    def __init__(self, goal_dim):
        super(GoalModule, self).__init__()
        pass  # Looks like it wasn't implemented in DFP

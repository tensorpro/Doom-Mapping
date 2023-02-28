## Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from torchvision.models import ResNet, resnet18

import time


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        return x


class Resize(nn.Module):

    def __init__(self, size):
        super(Resize, self).__init__()
        self.size = size

    def forward(self, x):
        output = F.upsample_bilinear(x, size=self.size)
        return output


def lenet_deconv(N=1):
    model = nn.Sequential(Resize((84, 84)),
                          LeNet(),
                          nn.ConvTranspose2d(128, 1, 3, 2),
                          nn.ConvTranspose2d(1, N, 3, 2),
                          Resize((84, 84))
                          )
    return model


def lenet_load(mode, cuda):
    if mode == 'depth':
        net = lenet_deconv(1)
    elif mode == 'label':
        net = lenet_deconv(7)  # change me back?
    else:
        raise ValueError('mode must be `depth` or `label`')

    if cuda:
        net = net.cuda()
        if mode != 'label':  # remove me later
            weights = torch.load('dense_models/lenet_deconv_{}.pth'.format(mode))
            net.load_state_dict(weights)
    else:
        if mode != 'label':  # remove me later
            weights = torch.load('dense_models/lenet_deconv_{}.pth'.format(mode),
                                 map_location=lambda x, y: x)
            net.load_state_dict(weights)

    return net


def lenet_labels(cuda):
    return lenet_load('label', cuda)


def lenet_depth(cuda):
    return lenet_load('depth', cuda)


def resnet_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x


ResNet.forward = resnet_forward


def custom_resnet18():
    net = resnet18()
    del net.fc
    del net.avgpool
    return net


class ConvTBNRelu(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, bias=True):
        super(ConvTBNRelu, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channel, out_channel, k, s, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel=1, k=3, s=2, bias=True):
        super(UpSample, self).__init__()
        # Note always returns in_channel / 16
        self.up_projections = nn.Sequential(
            nn.Dropout(p=.5),
            ConvTBNRelu(in_channel, in_channel / 2, k, s, bias=bias),
            #nn.Dropout(p=.5),
            ConvTBNRelu(in_channel / 2, in_channel / 4, k, s, bias=bias),
            #nn.Dropout(p=.5),
            ConvTBNRelu(in_channel / 4, in_channel / 8, k, s, bias=bias),
            #nn.Dropout(p=.5),
            ConvTBNRelu(in_channel / 8, out_channel, k, s, bias=bias)
        )

    def forward(self, input):
        return self.up_projections(input)


def resnet_upsample(N=1, insize=84):
    return nn.Sequential(Resize((insize, insize)),
                         custom_resnet18(),
                         UpSample(512, 512 / 32),
                         nn.Dropout(p=.5),
                         nn.Conv2d(512 / 32, N, 3),
                         Resize((84, 84)))


def resnet_load(mode, cuda):
    if mode == 'depth':
        net = resnet_upsample(1, insize=240)
    elif mode == 'label':
        net = resnet_upsample(6, insize=240)
    else:
        raise ValueError('mode must be `depth` or `label`')

    weights = torch.load('dense_models/resnet_upsample_{}.pth'.format(mode),
                         map_location=lambda x, y: x)

    net.load_state_dict(weights)
    if cuda:
        net = net.cuda()
    net.eval()
    return net

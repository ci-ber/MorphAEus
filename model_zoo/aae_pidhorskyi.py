# Copyright 2018-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
#
# S. Pidhorskyi, R. Almohsen, and G. Doretto. Generative probabilistic novelty detection with adversarial autoencoders.
# Advances in Neural Information Processing Systems, volume 31.

import torch
from torch import nn
from torch.nn import functional as F
from net_utils.initialize import normal_init


class AAE(nn.Module):
    """
    code from https://github.com/podgorskiy/GPND

    License:
        Apache License 2.0
    """
    def __init__(self, z_dim, cross_batch=False, batchSize=8, d=64, channels=1, extra_layers=0):
        super(AAE, self).__init__()
        self.z_dim = z_dim
        self.cross_batch = cross_batch
        self.batchSize = batchSize
        self.d = d
        self.channels = channels
        self.E = Encoder(z_dim, d, channels, extra_layers)
        self.G = Generator(z_dim, d, channels, extra_layers)
        if cross_batch:
            self.ZD = ZDiscriminator_mergebatch(z_dim, batchSize, d*2, extra_layers)
        else:
            self.ZD = ZDiscriminator(z_dim, batchSize, d*2, extra_layers)
        self.ZD.weight_init(mean=0, std=0.02)
        self.D = Discriminator(d, channels, extra_layers)

        self.weight_init(mean=0, std=0.02)

    def forward(self, x):
        z = self.E(x)
        x_ = self.G(z)
        return x_, {'z': z}

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class Generator(nn.Module):
    def __init__(self, z_size, d=64, channels=1, extra_layers=0):
        super(Generator, self).__init__()
        self.extra_layers = extra_layers
        self.deconv1_1 = nn.ConvTranspose2d(z_size, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)

        # Extra Layers for higher-res images (64x64, or 128x128)
        self.deconv3_1 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv3_bn1 = nn.BatchNorm2d(d*2)
        self.deconv3_2 = nn.ConvTranspose2d(d*2, d*2, 4, 2, 1)
        self.deconv3_bn2 = nn.BatchNorm2d(d*2)

        self.deconv4 = nn.ConvTranspose2d(d*2, channels, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        if self.extra_layers > 0:  # 64x64
            x = F.relu(self.deconv3_bn1(self.deconv3_1(x)))
        if self.extra_layers > 1:  # 128x128
            x = F.relu(self.deconv3_bn2(self.deconv3_2(x)))
        x = torch.tanh(self.deconv4(x)) * 0.5 + 0.5
        return x


class Discriminator(nn.Module):
    def __init__(self, d=64, channels=1, extra_layers=0):
        super(Discriminator, self).__init__()
        self.extra_layers = extra_layers
        self.conv1_1 = nn.Conv2d(channels, d, 4, 2, 1)
        # Extra layer 1
        self.conv1_2 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv1_bn2 = nn.BatchNorm2d(d)
        # Extra layer 2
        self.conv1_3 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv1_bn3 = nn.BatchNorm2d(d)
        # As Original
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        if self.extra_layers > 0:
            x = F.leaky_relu(self.conv1_bn2(self.conv1_2(x)), 0.2)
        if self.extra_layers > 1:
            x = F.leaky_relu(self.conv1_bn3(self.conv1_3(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


class Encoder(nn.Module):
    def __init__(self, z_size, d=64, channels=1, extra_layers=0):
        super(Encoder, self).__init__()
        self.extra_layers = extra_layers
        self.conv1_1 = nn.Conv2d(channels, d, 4, 2, 1)
        # Extra layer 1
        self.conv1_2 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv1_bn2 = nn.BatchNorm2d(d)
        # Extra layer 2
        self.conv1_3 = nn.Conv2d(d, d, 4, 2, 1)
        self.conv1_bn3 = nn.BatchNorm2d(d)
        # As Original
        self.conv2 = nn.Conv2d(d, d*4, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*4)
        self.conv3 = nn.Conv2d(d*4, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, z_size, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        if self.extra_layers > 0:
            x = F.leaky_relu(self.conv1_bn2(self.conv1_2(x)), 0.2)
        if self.extra_layers > 1:
            x = F.leaky_relu(self.conv1_bn3(self.conv1_3(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = self.conv4(x)
        return x


class ZDiscriminator(nn.Module):
    def __init__(self, z_size, batchSize, d=128, extra_layers=0):
        self.extra_layers = extra_layers
        super(ZDiscriminator, self).__init__()
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x


class ZDiscriminator_mergebatch(nn.Module):
    def __init__(self, z_size, batchSize, d=128, extra_layers=0):
        super(ZDiscriminator_mergebatch, self).__init__()
        self.extra_layers = extra_layers
        self.linear1 = nn.Linear(z_size, d)
        self.linear2 = nn.Linear(d * batchSize, d)
        self.linear3 = nn.Linear(d, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2).view(1, -1)  # after the second layer all samples are concatenated
        x = F.leaky_relu((self.linear2(x)), 0.2)
        x = torch.sigmoid(self.linear3(x))
        return x
# -*- coding: utf-8 -*-
# @Time   : 19-4-25 上午10:59
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from dataloader import dataloader
from readDataToGAN import *

class generator(nn.Module):
    """
       convtranspose2d
        #   1.  s =1
        #     o' = i' - 2p + k - 1
        #   2.  s >1
        # o = (i-1)*s +k-2p+n
        # n =  output_padding,p=padding,i=input dims,s=stride,k=kernel
    """
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * (8 * 4)),
            # nn.BatchNorm1d(64 * (8 * 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (4,2), 2, 1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (4,3), 2, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.output_dim, (4,3), 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        # x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = x.view(-1, 64, 8,4)
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 16, (4,2), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (4, 2), 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (4,2), 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            # nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.Linear(64 * ( 8* 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        # x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = x.view(-1, 64 * (8 * 4))
        x = self.fc(x)

        return x


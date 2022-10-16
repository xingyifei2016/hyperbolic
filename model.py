import torch
import numpy as np
import torch.nn as nn
from layer import *

class TwoConvTwoMaxpoolThreeChannelLinear(nn.Module):
  # Conv + relu + Maxpool + Conv + relu + Maxpool + Linear 
  def __init__(self):
        super(TwoConvTwoMaxpoolThreeChannelLinear, self).__init__()
        self.conv1 = nn.Conv2d(3,32,5)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,5)
        self.mp2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64,128,5)
        self.mp3 = nn.MaxPool2d(4)
        self.flatten = Flatten()
        #   self.fc = nn.Linear(18,2)
        self.fc = nn.Linear(2048,11)
        self.relu = nn.ReLU()

  def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.mp3(x)
        x = self.flatten(x)
        print(x.shape)
        x = self.fc(x)
        return x
    
class small_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, *args, **kwargs):
        super(small_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(3, 30, (5, 5), groups=3)
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3), groups=5)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2), groups=5)
        self.bn_3 = nn.GroupNorm(14, 70)
        self.linear_2 = nn.Linear(70, 30)
        self.linear_4 = nn.Linear(30, 10)
        self.res1 = nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
    
    
class small_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, *args, **kwargs):
        super(small_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(3, 30, (5, 5), groups=3)
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3), groups=5)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2), groups=5)
        self.bn_3 = nn.GroupNorm(14, 70)
        self.linear_2 = nn.Linear(70, 30)
        self.linear_4 = nn.Linear(30, 10)
        self.res1 = nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x
    
class small_cnn_li(nn.Module):
    # Backbone cnn layer
    def __init__(self, *args, **kwargs):
        super(small_cnn_li, self).__init__()
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(3, 30, (5, 5), groups=3)
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3), groups=5)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2), groups=5)
        self.bn_3 = nn.GroupNorm(14, 70)
        self.bn = nn.BatchNorm1d(10)
        self.linear_2 = nn.Linear(70, 30)
        self.linear_4 = nn.Linear(30, 10)
        self.res1 = nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))
        self.li = logInverse(1.2)

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)

        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))

        x = self.linear_2(x)
        x = self.relu(x)
        
        x = self.linear_4(x)
        x = self.li(x)
        x = self.bn(x)
        return x
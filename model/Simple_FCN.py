from __future__ import print_function, division
import torch.nn as nn
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(FCN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[2])
        self.Conv2 = conv_block(filters[2], filters[2])
        self.Conv3 = conv_block(filters[2], filters[2])
        # self.Conv4 = conv_block(filters[2], filters[2])
        # self.Conv5 = conv_block(filters[2], filters[2])

        self.Conv6 = nn.Conv1d(filters[2], out_ch, kernel_size=1, stride=1, padding='same')

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        # e4 = self.Conv4(e3)
        # e5 = self.Conv5(e4)
        e6 = self.Conv6(e3)
        return e6

def F_CN():
    return FCN(1, 1)


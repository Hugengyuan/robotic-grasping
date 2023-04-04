import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from .irnn import irnn

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=1.0):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))

    def forward(self, input):
        return irnn.apply(input.contiguous(), self.up_weight.weight.contiguous(), self.right_weight.weight.contiguous(), self.down_weight.weight.contiguous(),
                          self.left_weight.weight.contiguous(), self.up_weight.bias.contiguous(), self.right_weight.bias.contiguous(), self.down_weight.bias.contiguous(),
                          self.left_weight.bias.contiguous())


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return torch.mul(x, out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return torch.mul(x, out)


class CBAMC3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAMC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.channel_attention = ChannelAttention(c2, 16)
        self.spatial_attention = SpatialAttention(7)

    def forward(self, x):
   		# 将最后的标准卷积模块改为了注意力机制提取特征
        return self.spatial_attention(
            self.channel_attention(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))))

    
class DSC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1, alpha=1.0):
        super(DSC_Module, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels, alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels, alpha)
        self.conv_in = conv1x1(in_channels, in_channels)
        self.conv2 = conv1x1(in_channels * 4, in_channels)
        self.conv3 = conv1x1(in_channels * 4, in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = CBAMC3(in_channels, in_channels)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)

        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)

        return out


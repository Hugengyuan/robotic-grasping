import torch.nn as nn
import torch.nn.functional as F
import torch
from inference.models.grasp_model import GraspModel, ResidualBlock
from .DSC import DSC_Module

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv1_dsc = DSC_Module(channel_size, channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.conv2_dsc = DSC_Module(channel_size * 2, channel_size * 2)
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)
        self.conv3_dsc = DSC_Module(channel_size * 4, channel_size * 4)
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 8, channel_size * 4, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 4)

        self.conv5 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size * 2)

        self.conv6 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x1 = F.relu(self.bn1(self.conv1(x_in)))

        x1_dsc = self.conv1_dsc(x1)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2_dsc = self.conv2_dsc(x2)
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3_dsc = self.conv3_dsc(x3)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        context_1 = torch.cat((x3_dsc, x8), 1)
        x9 = F.relu(self.bn4(self.conv4(context_1)))
        print(x9.size())
        print(x2_dsc.size())
        context_2 = torch.cat((x2_dsc, x9), 1)
        x10 = F.relu(self.bn5(self.conv5(context_2)))
        context_3 = torch.cat((x1_dsc, x10), 1)
        x11 = self.conv6(context_3)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x11))
            cos_output = self.cos_output(self.dropout_cos(x11))
            sin_output = self.sin_output(self.dropout_sin(x11))
            width_output = self.width_output(self.dropout_wid(x11))
        else:
            pos_output = self.pos_output(x11)
            cos_output = self.cos_output(x11)
            sin_output = self.sin_output(x11)
            width_output = self.width_output(x11)

        return pos_output, cos_output, sin_output, width_output

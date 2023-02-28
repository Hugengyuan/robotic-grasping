import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=3, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        
        self.rgb_conv1 = nn.Conv2d(input_channels, channel_size * 2, kernel_size=9, stride=1, padding=4)
        self.rgb_bn1 = nn.BatchNorm2d(channel_size * 2)

        self.rgb_conv2 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.rgb_bn2 = nn.BatchNorm2d(channel_size * 4)

        self.rgb_conv3 = nn.Conv2d(channel_size * 4, channel_size * 8, kernel_size=4, stride=2, padding=1)
        self.rgb_bn3 = nn.BatchNorm2d(channel_size * 8)
        
        self.depth_conv1 = nn.Conv2d(input_channels, channel_size * 2, kernel_size=9, stride=1, padding=4)
        self.depth_bn1 = nn.BatchNorm2d(channel_size * 2)

        self.depth_conv2 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.depth_bn2 = nn.BatchNorm2d(channel_size * 4)

        self.depth_conv3 = nn.Conv2d(channel_size * 4, channel_size * 8, kernel_size=4, stride=2, padding=1)
        self.depth_bn3 = nn.BatchNorm2d(channel_size * 8)
        
        # 多层感知机的定义
        # self.fc1 = nn.Linear(in_features=3*224*224, out_features=4096)
        # self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        # self.fc3 = nn.Linear(in_features=4096, out_features=1024)
        

        self.res1 = ResidualBlock(channel_size * 16, channel_size * 16)
        self.res2 = ResidualBlock(channel_size * 16, channel_size * 16)
        self.res3 = ResidualBlock(channel_size * 16, channel_size * 16)

        self.up_conv1 = nn.ConvTranspose2d(channel_size * 16, channel_size * 8, kernel_size=4, stride=2, padding=1)
        self.up_bn1 = nn.BatchNorm2d(channel_size * 8)
        
        self.up_conv2 = nn.ConvTranspose2d(channel_size * 8, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.up_bn2 = nn.BatchNorm2d(channel_size * 4)


        self.up_conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size, kernel_size=9, stride=1, padding=4)
        
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

    def forward(self, rgb_in, depth_in):
        rgb_x = F.relu(self.rgb_bn1(self.rgb_conv1(rgb_in)))
        rgb_x = F.relu(self.rgb_bn2(self.rgb_conv2(rgb_x)))
        rgb_x = F.relu(self.rgb_bn3(self.rgb_conv3(rgb_x)))
        depth_x = depth_in.repeat_interleave(3,1)
        depth_x = F.relu(self.depth_bn1(self.depth_conv1(depth_x)))
        depth_x = F.relu(self.depth_bn2(self.depth_conv2(depth_x)))
        depth_x = F.relu(self.depth_bn3(self.depth_conv3(depth_x)))
        
        x = torch.cat([rgb_x, depth_x], 1)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
       
        x = F.relu(self.up_bn1(self.up_conv1(x)))
        x = F.relu(self.up_bn2(self.up_conv2(x)))
        x = F.relu(self.up_bn3(self.up_conv3(x)))
        x = self.up_conv4(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

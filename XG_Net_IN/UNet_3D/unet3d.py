import torch
from torch import nn


class VGGBlock3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size=2, stride=1, padding=1, dilation=2)
        self.bn1 = nn.InstanceNorm3d(middle_channels)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size=2, stride=1, padding=1, dilation=2)
        self.bn2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        # print('out1',out.shape)
        out = self.bn1(out)
        out = self.relu(out)
        # print('out2', out.shape)

        out = self.conv2(out)
        # print('out3', out.shape)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet3D(nn.Module):
    def __init__(self, num_classes=2, input_channels=1, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool3d(2, 2)
        self.up1 = nn.Upsample(scale_factor=1, mode='nearest', align_corners=None)
        self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.conv0_0 = VGGBlock3D(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock3D(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock3D(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock3D(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock3D(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock3D(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock3D(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock3D(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock3D(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print(input.shape)
        x0_0 = self.conv0_0(input)
        # print(x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print(x1_0.shape)
        x2_0 = self.conv2_0(self.pool(x1_0))
        # print(x2_0.shape)
        x3_0 = self.conv3_0(self.pool(x2_0))
        # print(x3_0.shape)
        x4_0 = self.conv4_0(x3_0)
        # print(x4_0.shape)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up1(x4_0)], 1))
        # print(x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        # print(x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        # print(x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))
        # print(x0_4.shape)

        output = self.final(x0_4)
        # print(output.shape)
        return output


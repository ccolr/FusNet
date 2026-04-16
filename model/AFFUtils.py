import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_CAM(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # global attention
        self.glob_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_local = self.local_att(x)
        x_glob = self.glob_att(x)
        x_local_and_glob = x_local + x_glob
        return x * self.sigmoid(x_local_and_glob)


class AFF(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels=128, r=4):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels_1, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels_2, out_channels, kernel_size=1, stride=1, padding=0)

        inter_channels = int(out_channels // r)

        # local attention
        self.local_att = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        # global attention
        self.glob_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # 对齐通道数, 先检查通道数是否等于out_channels
        if x.shape[1] != self.out_channels:
            x = self.conv1(x)
        if y.shape[1] != self.out_channels:
            y = self.conv2(y)

        sum = x + y
        sum_glob = self.glob_att(sum)
        sum_local = self.local_att(sum)
        att_sum = sum_glob + sum_local

        return x * self.sigmoid(att_sum) + y * (1 - self.sigmoid(att_sum))


class iAFF(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, r=4):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels_1, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels_2, out_channels, kernel_size=1, stride=1, padding=0)

        inter_channels = int(out_channels // r)

        # local attention1
        self.local_att_1 = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        # global attention1
        self.glob_att_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()

        # local attention2
        self.local_att_2 = nn.Sequential(
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )

        # global attention2
        self.glob_att_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # 对齐通道数, 先检查通道数是否等于out_channels
        if x.shape[1] != self.out_channels:
            x = self.conv1(x)
        if y.shape[1] != self.out_channels:
            y = self.conv2(y)

        sum_1 = x + y
        sum_local_1 = self.local_att_1(sum_1)
        sum_glob_1 = self.glob_att_1(sum_1)
        sum_local_and_glob_1 = sum_local_1 + sum_glob_1
        xi = x * self.sigmoid(sum_local_and_glob_1) + y * (1 - self.sigmoid(sum_local_and_glob_1))

        sum_local_2 = self.local_att_2(xi)
        sum_glob_2 = self.glob_att_2(xi)
        sum_local_and_glob_2 = sum_local_2 + sum_glob_2
        return x * self.sigmoid(sum_local_and_glob_2) + y * (1 - self.sigmoid(sum_local_and_glob_2))


# class AttentionFusion(nn.Module):
#     """
#     将两个通道数统一
#     """
#     def __init__(self, in_channels_1, in_channels_2, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels_1, out_channels, kernel_size=1, stride=1, padding=0)
#         self.conv2 = nn.Conv2d(in_channels_2, out_channels, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x, y):
#         x = self.conv1(x)
#         y = self.conv2(y)
#         return x, y


if __name__ == "__main__":
    model = iAFF(64, 32, 128)
    x = torch.randn(2, 64, 128, 128)
    y = torch.randn(2, 32, 128, 128)
    z = model(x, y)

    print(z.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_vision import mamba_vision_S
from model.res2net import res2net50_v1b_26w_4s
import torchvision
from model.swin import swin_tiny_patch4_window7_224

# from model.Samba import samba_new
from model.AFFUtils import iAFF, AFF, MS_CAM


class RGBChannelAttention(nn.Module):
    """
    对 RGB 三个通道分别加权增强，使网络对绿色区域更加敏感。
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # 每个通道按权重放大或缩小


class TextureConv(nn.Module):
    """
    增强局部纹理特征，对竹叶颗粒感敏感。
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class FusNet(nn.Module):
    # res2net, swin transformer, mamba融合的编码器-解码器网络
    def __init__(self, outchannel):
        super().__init__()

        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)  # 使用预训练的res2net作为特征提取网络

        self.rgb_channel_attention = RGBChannelAttention(64)  # RGB通道注意力模块
        self.texture_conv = TextureConv(64, 64)  # 纹理增强卷积模块

        # ---- Swin Transformer Backbone ----
        self.swin = swin_tiny_patch4_window7_224(pretrained=True)  # 使用预训练的swin transformer作为特征提取网络

        # ---- Mamba Encoder ----
        self.mamba = mamba_vision_S(
            pretrained=True,
            model_path="./model/pretrained/mambavision_small_1k.pth.tar",
        )  # 使用预训练的mamba作为特征提取网络

        # ---- 特征融合层 ----
        self.fuse1 = nn.Conv2d(
            in_channels=640, out_channels=256, kernel_size=1, stride=1, padding=0
        )  # 融合resnet, swin, mamba的输出
        self.fuse2 = nn.Conv2d(in_channels=896, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.fuse3 = nn.Conv2d(in_channels=1792, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.fuse4 = nn.Conv2d(in_channels=3584, out_channels=2048, kernel_size=1, stride=1, padding=0)

        # -----------------------
        # ---- 新的特征融合层! ----
        # -----------------------
        self.new_fuse1_1 = iAFF(in_channels_1=96, in_channels_2=96, out_channels=96, r=4)
        self.new_fuse1_2 = iAFF(in_channels_1=96, in_channels_2=256, out_channels=96, r=4)
        self.new_fuse1_3 = iAFF(in_channels_1=256, in_channels_2=96, out_channels=256, r=4)

        self.new_fuse2_1 = iAFF(in_channels_1=192, in_channels_2=192, out_channels=192, r=4)
        self.new_fuse2_2 = iAFF(in_channels_1=192, in_channels_2=512, out_channels=192, r=4)
        self.new_fuse2_3 = iAFF(in_channels_1=512, in_channels_2=192, out_channels=512, r=4)

        self.new_fuse3_1 = iAFF(in_channels_1=384, in_channels_2=384, out_channels=384, r=4)
        self.new_fuse3_2 = iAFF(in_channels_1=384, in_channels_2=1024, out_channels=384, r=4)
        self.new_fuse3_3 = iAFF(in_channels_1=1024, in_channels_2=384, out_channels=1024, r=4)

        self.new_fuse4_1 = iAFF(in_channels_1=768, in_channels_2=768, out_channels=768, r=4)
        self.new_fuse4_2 = iAFF(in_channels_1=768, in_channels_2=2048, out_channels=768, r=4)
        self.new_fuse4_3 = iAFF(in_channels_1=2048, in_channels_2=768, out_channels=2048, r=4)

        # ---- 解码器层 ----
        self.x5_dem_1 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x4_dem_1 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x3_dem_1 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x2_dem_1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x5_x4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x4_x3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x3_x2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x2_x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.x5_x4_x3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x4_x3_x2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x3_x2_x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.x5_x4_x3_x2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x4_x3_x2_x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x5_dem_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.x5_x4_x3_x2_x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.level3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.level1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.x5_dem_5 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.output4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.output3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.output2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.output1 = nn.Sequential(nn.Conv2d(64, outchannel, kernel_size=3, padding=1))  # 最终输出层

    def forward(self, x):  # (1, 3, 224, 224)
        input = x
        bs, C, H, W = x.shape

        # ---- Res2Net特征提取 ----
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x1 = self.resnet.maxpool(x)  # shape[1, 64, 56, 56]

        # ---- 插入 RGB + Texture 模块 ----
        x1_rgb_att = self.rgb_channel_attention(x1)  # 注意力增强
        x1_tex = self.texture_conv(x1)  # 纹理增强

        # 拼接三个特征(用打铁形象理解, 此处就是把RGB特征、RGB注意力增强特征、纹理增强特征打成一块铁块), 送入后续网络
        x1_enhanced = torch.cat([x1, x1_rgb_att, x1_tex], dim=1)  # shape[1, 64*3, 56, 56]

        # 此处理解为持续锻造, 让增强后的特征更好地融合在一起
        x2 = self.resnet.layer1(x1)  # shape[1, 256, 56, 56]
        x3 = self.resnet.layer2(x2)  # shape[1, 512, 28, 28]
        x4 = self.resnet.layer3(x3)  # shape[1, 1024, 14, 14]
        x5 = self.resnet.layer4(x4)  # shape[1, 2048, 7, 7]

        # ---- Swin Transformer特征提取 ----
        x, H, W = self.swin.patch_embed(input)
        x = self.swin.pos_drop(x)

        x1s = x.view(bs, 96, H, W)  # shape[1, 96, 56, 56]
        x, H, W = self.swin.layers[0](x, H, W)
        x2s = x.permute(0, 2, 1).view(bs, 192, H, W)  # shape[1, 192, 28, 28]
        x, H, W = self.swin.layers[1](x, H, W)
        x3s = x.permute(0, 2, 1).view(bs, 384, H, W)  # shape[1, 384, 14, 14]
        x, H, W = self.swin.layers[2](x, H, W)
        x4s = x.permute(0, 2, 1).view(bs, 768, H, W)  # shape[1, 768, 7, 7]

        # ---- Mamba特征提取 ----
        mamba_x1 = self.mamba.patch_embed(input)  # [1, 96, 56, 56]
        mamba_x2 = self.mamba.levels[0](mamba_x1)  # shape[1, 192, 28, 28]
        mamba_x3 = self.mamba.levels[1](mamba_x2)  # shape[1, 384, 14, 14]
        mamba_x4 = self.mamba.levels[2](mamba_x3)  # shape[1, 768, 7, 7]  # 获取mamba的多层次特征

        # ---- 融合特征 ----
        new_fuse1_1 = self.new_fuse1_1(mamba_x1, x1s) # [1, 96, 56, 56]
        new_fuse1_2 = self.new_fuse1_2(x1s, x2) # [1, 96, 56, 56]
        new_fuse1_3 = self.new_fuse1_3(x2, mamba_x1) # [1, 256, 56, 56]

        new_fuse2_1 = self.new_fuse2_1(mamba_x2, x2s)
        new_fuse2_2 = self.new_fuse2_2(x2s, x3)
        new_fuse2_3 = self.new_fuse2_3(x3, mamba_x2)

        new_fuse3_1 = self.new_fuse3_1(mamba_x3, x3s)
        new_fuse3_2 = self.new_fuse3_2(x3s, x4)
        new_fuse3_3 = self.new_fuse3_3(x4, mamba_x3)

        new_fuse4_1 = self.new_fuse4_1(mamba_x4, x4s)
        new_fuse4_2 = self.new_fuse4_2(x4s, x5)
        new_fuse4_3 = self.new_fuse4_3(x5, mamba_x4)

        # x2 = self.fuse1(torch.cat([new_fuse1_1, new_fuse1_2, new_fuse1_3], dim=1))
        x2 = self.fuse1(torch.cat([new_fuse1_1, new_fuse1_2, new_fuse1_3, x1_enhanced], dim=1))
        x3 = self.fuse2(torch.cat([new_fuse2_1, new_fuse2_2, new_fuse2_3], dim=1))
        x4 = self.fuse3(torch.cat([new_fuse3_1, new_fuse3_2, new_fuse3_3], dim=1))
        x5 = self.fuse4(torch.cat([new_fuse4_1, new_fuse4_2, new_fuse4_3], dim=1))

        # ---- 解码特征，消除冗余 ----
        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_4 = self.x5_x4(abs(F.interpolate(x5_dem_1, size=x4.size()[2:], mode="bilinear") - x4_dem_1))
        x4_3 = self.x4_x3(abs(F.interpolate(x4_dem_1, size=x3.size()[2:], mode="bilinear") - x3_dem_1))
        x3_2 = self.x3_x2(abs(F.interpolate(x3_dem_1, size=x2.size()[2:], mode="bilinear") - x2_dem_1))
        x2_1 = self.x2_x1(abs(F.interpolate(x2_dem_1, size=x1.size()[2:], mode="bilinear") - x1))

        x5_4_3 = self.x5_x4_x3(abs(F.interpolate(x5_4, size=x4_3.size()[2:], mode="bilinear") - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.interpolate(x4_3, size=x3_2.size()[2:], mode="bilinear") - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.interpolate(x3_2, size=x2_1.size()[2:], mode="bilinear") - x2_1))

        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.interpolate(x5_4_3, size=x4_3_2.size()[2:], mode="bilinear") - x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.interpolate(x4_3_2, size=x3_2_1.size()[2:], mode="bilinear") - x3_2_1))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(F.interpolate(x5_dem_4, size=x4_3_2_1.size()[2:], mode="bilinear") - x4_3_2_1)
        )

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.interpolate(x5_dem_5, size=level4.size()[2:], mode="bilinear") + level4)
        output3 = self.output3(F.interpolate(output4, size=level3.size()[2:], mode="bilinear") + level3)
        output2 = self.output2(F.interpolate(output3, size=level2.size()[2:], mode="bilinear") + level2)
        output1 = self.output1(F.interpolate(output2, size=level1.size()[2:], mode="bilinear") + level1)

        output = F.interpolate(output1, size=input.size()[2:], mode="bilinear")

        return output

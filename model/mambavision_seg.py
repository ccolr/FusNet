import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_vision import mamba_vision_S


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.skip_proj = (
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False)
            if skip_channels > 0 else None
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + (out_channels if skip_channels > 0 else 0),
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        if skip is not None and self.skip_proj is not None:
            skip = self.skip_proj(skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MambaVisionSeg(nn.Module):
    """MambaVision Small + FPN decoder baseline for bamboo segmentation.

    Encoder stages:
        patch_embed:  (B, 96,  56×56)
        levels[0]:    (B, 192, 28×28)
        levels[1]:    (B, 384, 14×14)
        levels[2]:    (B, 768, 7×7)
    """

    def __init__(self, num_classes=2, pretrained=True,
                 model_path="./model/pretrained/mambavision_small_1k.pth.tar"):
        super().__init__()
        self.encoder = mamba_vision_S(pretrained=pretrained, model_path=model_path)

        self.lat1 = nn.Conv2d(96,  256, kernel_size=1, bias=False)
        self.lat2 = nn.Conv2d(192, 256, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(384, 256, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(768, 256, kernel_size=1, bias=False)

        self.decoder = nn.ModuleList([
            DecodeBlock(256, skip_channels=256, out_channels=256, scale_factor=2),  # 7→14
            DecodeBlock(256, skip_channels=256, out_channels=128, scale_factor=2),  # 14→28
            DecodeBlock(128, skip_channels=256, out_channels=64,  scale_factor=2),  # 28→56
            DecodeBlock(64,  skip_channels=0,   out_channels=64,  scale_factor=4),  # 56→224
        ])

        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def _extract_features(self, x):
        f1 = self.encoder.patch_embed(x)
        f2 = self.encoder.levels[0](f1)
        f3 = self.encoder.levels[1](f2)
        f4 = self.encoder.levels[2](f3)
        return f1, f2, f3, f4

    def forward(self, x):
        f1, f2, f3, f4 = self._extract_features(x)

        p1 = self.lat1(f1)
        p2 = self.lat2(f2)
        p3 = self.lat3(f3)
        p4 = self.lat4(f4)

        skips = [p3, p2, p1, None]
        x_dec = p4
        for decode_block, skip in zip(self.decoder, skips):
            x_dec = decode_block(x_dec, skip)

        return self.seg_head(x_dec)

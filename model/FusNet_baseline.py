# FusNet_baseline.py
#
# FusNet 的简单融合对比版本，用于与 FusNet（HCABlock）做消融对比。
# 将四阶段 HCABlock 替换为 AddFuseBlock 或 CatFuseBlock：
#   "add" — 各路 1×1conv 投影到 dim_feat 后直接相加，无跨路交互
#   "cat" — 各路 1×1conv 投影后拼接，再 1×1conv 压回 dim_feat
# Backbone、解码器、输出头、active_branches 消融接口与 FusNet.py 完全相同。

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_vision import mamba_vision_S
from model.res2net import res2net50_v1b_26w_4s
from model.swin import swin_tiny_patch4_window7_224

ALL_BRANCHES = ["res2net", "swin", "mamba"]

STAGE_CHANNELS = {
    "res2net": [256,  512,  1024, 2048],
    "swin":    [96,   192,  384,  768],
    "mamba":   [96,   192,  384,  768],
}


# ─────────────────────────────────────────────
# 融合模块（替换 HCABlock）
# ─────────────────────────────────────────────


class AddFuseBlock(nn.Module):
    """
    各路 1×1conv 投影到 dim_feat 后直接相加，再 BN+ReLU。
    无任何跨路交互，是 HCABlock 的极简基线。
    """

    def __init__(self, in_channels: dict, out_channels: int):
        super().__init__()
        self.projs = nn.ModuleDict(
            {b: nn.Conv2d(ch, out_channels, kernel_size=1, bias=False) for b, ch in in_channels.items()}
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, **feats):
        out = None
        for b, proj in self.projs.items():
            y = proj(feats[f"f_{b}"])
            out = y if out is None else out + y
        return self.act(self.norm(out))


class CatFuseBlock(nn.Module):
    """
    各路 1×1conv+BN+ReLU 投影到 dim_feat 后拼接，
    再 1×1conv+BN+ReLU 压回 dim_feat。
    比 Add 多一个跨通道线性混合，但仍无空间级跨路交互。
    """

    def __init__(self, in_channels: dict, out_channels: int):
        super().__init__()
        n = len(in_channels)
        self.projs = nn.ModuleDict(
            {
                b: nn.Sequential(
                    nn.Conv2d(ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for b, ch in in_channels.items()
            }
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(n * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, **feats):
        parts = [self.projs[b](feats[f"f_{b}"]) for b in self.projs]
        return self.fuse(torch.cat(parts, dim=1))


# ─────────────────────────────────────────────
# 解码模块（与 FusNet.py 完全相同）
# ─────────────────────────────────────────────


class DecodeBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.skip_proj = (
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False) if skip_channels > 0 else None
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


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────


class FusNetBaseline(nn.Module):
    """
    FusNet 简单融合对比版本。

    与 FusNet（HCABlock）的唯一区别：
        将四阶段 HCABlock 替换为 AddFuseBlock 或 CatFuseBlock，
        其余结构（backbone / decoder / 输出头 / active_branches 接口）完全相同。

    Args:
        num_classes    : 分割类别数
        dim_feat       : 投影对齐的公共特征维度（默认256）
        fusion_mode    : "add" 或 "cat"
        active_branches: 激活的 backbone 列表，默认三路全开。
                         消融示例：["res2net", "swin"] 去掉 MambaVision

    使用示例:
        model_add = FusNetBaseline(fusion_mode="add")
        model_cat = FusNetBaseline(fusion_mode="cat")
        model_cat_2b = FusNetBaseline(fusion_mode="cat", active_branches=["res2net", "swin"])
    """

    def __init__(self, num_classes=2, dim_feat=256, fusion_mode="add", active_branches=None):
        super().__init__()
        assert fusion_mode in ("add", "cat"), f"fusion_mode 应为 'add' 或 'cat'，得到 '{fusion_mode}'"

        if active_branches is None:
            active_branches = ALL_BRANCHES
        assert 2 <= len(active_branches) <= 3
        for b in active_branches:
            assert b in ALL_BRANCHES, f"未知 branch: {b}"
        self.active_branches = list(active_branches)
        self.dim_feat = dim_feat

        # ── Backbones（只初始化激活的，与 FusNet.py 相同）────────
        self.resnet = res2net50_v1b_26w_4s(pretrained=True) if "res2net" in self.active_branches else None
        self.swin = swin_tiny_patch4_window7_224(pretrained=True) if "swin" in self.active_branches else None
        self.mamba = (
            mamba_vision_S(pretrained=True, model_path="./model/pretrained/mambavision_small_1k.pth.tar")
            if "mamba" in self.active_branches
            else None
        )

        # ── 四阶段简单融合模块 ──────────────────────────────────
        FuseBlock = AddFuseBlock if fusion_mode == "add" else CatFuseBlock
        self.fuse_stages = nn.ModuleList(
            [
                FuseBlock(
                    in_channels={b: STAGE_CHANNELS[b][stage_idx] for b in self.active_branches},
                    out_channels=dim_feat,
                )
                for stage_idx in range(4)
            ]
        )

        # ── 解码器（与 FusNet.py 完全相同）────────────────────
        self.decoder = nn.ModuleList(
            [
                DecodeBlock(dim_feat, skip_channels=dim_feat, out_channels=256, scale_factor=2),  # 7→14
                DecodeBlock(256, skip_channels=dim_feat, out_channels=128, scale_factor=2),  # 14→28
                DecodeBlock(128, skip_channels=dim_feat, out_channels=64, scale_factor=2),  # 28→56
                DecodeBlock(64, skip_channels=0, out_channels=64, scale_factor=4),  # 56→224
            ]
        )

        # ── 分割输出头 ──────────────────────────────────────────
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    # ──────────────────────────────────────────────────────────
    def _extract_res2net(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        f1 = self.resnet.layer1(x)   # (B, 256,  56×56)
        f2 = self.resnet.layer2(f1)  # (B, 512,  28×28)
        f3 = self.resnet.layer3(f2)  # (B, 1024, 14×14)
        f4 = self.resnet.layer4(f3)  # (B, 2048, 7×7)
        return f1, f2, f3, f4

    def _extract_swin(self, x):
        B = x.shape[0]
        tokens, H, W = self.swin.patch_embed(x)
        tokens = self.swin.pos_drop(tokens)
        f1 = tokens.permute(0, 2, 1).view(B, 96, H, W)     # (B, 96,  56×56)
        tokens, H, W = self.swin.layers[0](tokens, H, W)
        f2 = tokens.permute(0, 2, 1).view(B, 192, H, W)    # (B, 192, 28×28)
        tokens, H, W = self.swin.layers[1](tokens, H, W)
        f3 = tokens.permute(0, 2, 1).view(B, 384, H, W)    # (B, 384, 14×14)
        tokens, H, W = self.swin.layers[2](tokens, H, W)
        f4 = tokens.permute(0, 2, 1).view(B, 768, H, W)    # (B, 768, 7×7)
        return f1, f2, f3, f4

    def _extract_mamba(self, x):
        f1 = self.mamba.patch_embed(x)       # (B, 96,  56×56)
        f2 = self.mamba.levels[0](f1)        # (B, 192, 28×28)
        f3 = self.mamba.levels[1](f2)        # (B, 384, 14×14)
        f4 = self.mamba.levels[2](f3)        # (B, 768, 7×7)
        return f1, f2, f3, f4

    # ──────────────────────────────────────────────────────────
    def forward(self, x):
        # ── 1. 只提取激活路特征 ────────────────────────────────
        stage_feats = {b: [None] * 4 for b in ALL_BRANCHES}

        if "res2net" in self.active_branches:
            for i, f in enumerate(self._extract_res2net(x)):
                stage_feats["res2net"][i] = f
        if "swin" in self.active_branches:
            for i, f in enumerate(self._extract_swin(x)):
                stage_feats["swin"][i] = f
        if "mamba" in self.active_branches:
            for i, f in enumerate(self._extract_mamba(x)):
                stage_feats["mamba"][i] = f

        # ── 2. 四阶段简单融合 ───────────────────────────────────
        fused = []
        for stage_idx, fuse in enumerate(self.fuse_stages):
            kwargs = {f"f_{b}": stage_feats[b][stage_idx] for b in self.active_branches}
            fused.append(fuse(**kwargs))
        # fused[0]: (B, dim_feat, 56×56)
        # fused[1]: (B, dim_feat, 28×28)
        # fused[2]: (B, dim_feat, 14×14)
        # fused[3]: (B, dim_feat, 7×7)

        # ── 3. FPN 式解码（与 FusNet.py 完全相同）───────────────
        skips = [fused[2], fused[1], fused[0], None]
        x_dec = fused[3]
        for decode_block, skip in zip(self.decoder, skips):
            x_dec = decode_block(x_dec, skip)

        # ── 4. 分割输出 ────────────────────────────────────────
        return self.seg_head(x_dec)  # (B, num_classes, 224×224)

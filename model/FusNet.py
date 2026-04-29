import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_vision import mamba_vision_S
from model.res2net import res2net50_v1b_26w_4s
from model.swin import swin_tiny_patch4_window7_224
from model.HCABlock import HCABlock

ALL_BRANCHES = ["res2net", "swin", "mamba"]

# 各阶段每路通道数
STAGE_CHANNELS = {
    "res2net": [256, 512,  1024, 2048],
    "swin":    [96,  192,  384,  768],
    "mamba":   [96,  192,  384,  768],
}


# ─────────────────────────────────────────────
# 子模块
# ─────────────────────────────────────────────


class DecodeBlock(nn.Module):
    """
    单步解码：上采样 → 与跳跃连接融合 → conv提炼。
    skip_channels=0 表示该步骤没有跳跃连接。
    """

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


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────


class FusNet(nn.Module):
    """
    可配置路数的多编码器融合分割网络（用于消融实验）。
    各阶段通道数：
        Stage1 (56×56): Res2Net layer1=256,  Swin stem=96,   Mamba patch_embed=96
        Stage2 (28×28): Res2Net layer2=512,  Swin layer0=192, Mamba level0=192
        Stage3 (14×14): Res2Net layer3=1024, Swin layer1=384, Mamba level1=384
        Stage4 (7×7):   Res2Net layer4=2048, Swin layer2=768, Mamba level2=768

    Args:
        num_classes    : 分割类别数
        dim_feat       : HCA 输出公共特征维度（默认256）
        num_heads      : HCA 多头注意力头数（默认8）
        active_branches: 激活的 backbone 列表，可选值为
                         "res2net" / "swin" / "mamba"
                         默认 None → 三路全开
                         消融示例：
                           ["res2net", "swin"]    去掉 MambaVision
                           ["res2net", "mamba"]   去掉 Swin-T
                           ["swin",    "mamba"]   去掉 Res2Net

    使用示例:
        model = FusNet(active_branches=["res2net", "swin"])
        out = model(x)   # 正常调用，内部自动只走激活的路
    """

    def __init__(self, num_classes=2, dim_feat=256, num_heads=8, active_branches=None):
        super().__init__()

        # ── 解析 active_branches ──────────────────────────────
        if active_branches is None:
            active_branches = ALL_BRANCHES
        assert 2 <= len(active_branches) <= 3
        for b in active_branches:
            assert b in ALL_BRANCHES, f"未知 branch: {b}"
        self.active_branches = list(active_branches)

        self.dim_feat = dim_feat

        # ── Backbones（只初始化激活的）────────────────────────
        self.resnet = res2net50_v1b_26w_4s(pretrained=True) if "res2net" in self.active_branches else None
        self.swin   = swin_tiny_patch4_window7_224(pretrained=True) if "swin" in self.active_branches else None
        self.mamba  = (
            mamba_vision_S(
                pretrained=True,
                model_path="./model/pretrained/mambavision_small_1k.pth.tar",
            )
            if "mamba" in self.active_branches else None
        )

        # ── 四阶段 HCA 融合模块 ─────────────────────────────────
        # 各阶段 in_channels 只包含激活路
        stage_kv_strides = [2, 2, 1, 1]
        stage_use_pos    = [False, False, True, True]

        self.hca_stages = nn.ModuleList([
            HCABlock(
                in_channels={b: STAGE_CHANNELS[b][stage_idx] for b in self.active_branches},
                out_channels=dim_feat,
                num_heads=num_heads,
                kv_stride=stage_kv_strides[stage_idx],
                ffn_ratio=4.0,
                use_pos_bias=stage_use_pos[stage_idx],
                active_branches=self.active_branches,
            )
            for stage_idx in range(4)
        ])

        # ── 解码器（FPN式，结构不变）───────────────────────────
        self.decoder = nn.ModuleList([
            DecodeBlock(dim_feat, skip_channels=dim_feat, out_channels=256, scale_factor=2),  # 7→14
            DecodeBlock(256,      skip_channels=dim_feat, out_channels=128, scale_factor=2),  # 14→28
            DecodeBlock(128,      skip_channels=dim_feat, out_channels=64,  scale_factor=2),  # 28→56
            DecodeBlock(64,       skip_channels=0,        out_channels=64,  scale_factor=4),  # 56→224
        ])

        # ── 分割输出头 ──────────────────────────────────────────
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    # ──────────────────────────────────────────────────────────
    def _extract_res2net(self, x):
        """
        提取 Res2Net 四个阶段特征。
        Returns:
            f1: (B, 256,  56×56)  layer1
            f2: (B, 512,  28×28)  layer2
            f3: (B, 1024, 14×14)  layer3
            f4: (B, 2048, 7×7)    layer4
        """
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
        """
        提取 Swin-T 四个阶段特征。
        Returns:
            f1: (B, 96,  56×56)  patch_embed 后（stem）
            f2: (B, 192, 28×28)  layers[0] 后
            f3: (B, 384, 14×14)  layers[1] 后
            f4: (B, 768, 7×7)    layers[2] 后
        """
        B = x.shape[0]
        tokens, H, W = self.swin.patch_embed(x)
        tokens = self.swin.pos_drop(tokens)
        f1 = tokens.permute(0, 2, 1).view(B, 96, H, W)    # (B, 96,  56×56)
        tokens, H, W = self.swin.layers[0](tokens, H, W)
        f2 = tokens.permute(0, 2, 1).view(B, 192, H, W)   # (B, 192, 28×28)
        tokens, H, W = self.swin.layers[1](tokens, H, W)
        f3 = tokens.permute(0, 2, 1).view(B, 384, H, W)   # (B, 384, 14×14)
        tokens, H, W = self.swin.layers[2](tokens, H, W)
        f4 = tokens.permute(0, 2, 1).view(B, 768, H, W)   # (B, 768, 7×7)
        return f1, f2, f3, f4

    def _extract_mamba(self, x):
        """
        提取 MambaVision 四个阶段特征。
        Returns:
            f1: (B, 96,  56×56)  patch_embed 后（stem）
            f2: (B, 192, 28×28)  levels[0] 后
            f3: (B, 384, 14×14)  levels[1] 后
            f4: (B, 768, 7×7)    levels[2] 后
        """
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
            feats = self._extract_res2net(x)
            for i, f in enumerate(feats):
                stage_feats["res2net"][i] = f

        if "swin" in self.active_branches:
            feats = self._extract_swin(x)
            for i, f in enumerate(feats):
                stage_feats["swin"][i] = f

        if "mamba" in self.active_branches:
            feats = self._extract_mamba(x)
            for i, f in enumerate(feats):
                stage_feats["mamba"][i] = f

        # ── 2. 四阶段 HCA 融合 ─────────────────────────────────
        fused = []
        for stage_idx, hca in enumerate(self.hca_stages):
            kwargs = {
                f"f_{b}": stage_feats[b][stage_idx]
                for b in self.active_branches
            }
            fused.append(hca(**kwargs))
        # fused[0]: (B, dim_feat, 56×56)
        # fused[1]: (B, dim_feat, 28×28)
        # fused[2]: (B, dim_feat, 14×14)
        # fused[3]: (B, dim_feat, 7×7)

        # ── 3. FPN 式解码 ──────────────────────────────────────
        skips = [fused[2], fused[1], fused[0], None]
        x_dec = fused[3]
        for decode_block, skip in zip(self.decoder, skips):
            x_dec = decode_block(x_dec, skip)

        # ── 4. 分割输出 ────────────────────────────────────────
        return self.seg_head(x_dec)  # (B, num_classes, 224×224)

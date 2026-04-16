import torch
import torch.nn as nn
import torch.nn.functional as F

from model.mamba_vision import mamba_vision_S
from model.res2net import res2net50_v1b_26w_4s
from model.swin import swin_tiny_patch4_window7_224
from model.AFFUtils import iAFF


# ─────────────────────────────────────────────
# 子模块
# ─────────────────────────────────────────────


class SpatialGate(nn.Module):
    """
    用Res2Net layer0的输出生成空间注意力权重图。
    输出shape: (B, 1, H, W)，用于调制patch_embed的输出。
    """

    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.gate(x)  # (B, 1, H, W)


class ProjectionAdd(nn.Module):
    """
    将三路不同通道数的特征分别用1×1conv投影到同一维度后相加。
    """

    def __init__(self, ch_res, ch_swin, ch_mamba, dim_out):
        super().__init__()
        self.proj_res = nn.Conv2d(ch_res, dim_out, kernel_size=1, bias=False)
        self.proj_swin = nn.Conv2d(ch_swin, dim_out, kernel_size=1, bias=False)
        self.proj_mamba = nn.Conv2d(ch_mamba, dim_out, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat_res, feat_swin, feat_mamba):
        return self.act(self.norm(self.proj_res(feat_res) + self.proj_swin(feat_swin) + self.proj_mamba(feat_mamba)))


class GlobalFeedbackGate(nn.Module):
    """
    用最深层的Mamba全局特征生成通道门控，调制浅层融合特征。
    只在编码器末尾做一次，不是每个stage都做。
    """

    def __init__(self, ch_global, ch_local):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(nn.Linear(ch_global, ch_local, bias=False), nn.Sigmoid())

    def forward(self, feat_global, feat_local):
        # feat_global: (B, ch_global, H, W)
        # feat_local:  (B, ch_local,  H, W)
        g = self.gap(feat_global).flatten(1)  # (B, ch_global)
        w = self.gate(g).unsqueeze(-1).unsqueeze(-1)  # (B, ch_local, 1, 1)
        return feat_local * w


class DecodeBlock(nn.Module):
    """
    单步解码：上采样 → 与跳跃连接融合 → conv提炼。
    skip_channels=0 表示该步骤没有跳跃连接。
    """

    def __init__(self, in_channels, skip_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        # fuse_channels = in_channels + skip_channels

        # 跳跃连接通道对齐（1×1conv压缩到out_channels）
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


class FusNet(nn.Module):
    """
    三路编码器（Res2Net + Swin-T + MambaVision）串行iAFF融合
    + FPN式解码器用于遥感竹林分割。

    Args:
        num_classes:  分割类别数
        dim_feat:     三路投影对齐的公共特征维度（默认256）
        iaff_r:       iAFF模块的reduction ratio（默认4）
    """

    def __init__(self, num_classes=2, dim_feat=256, iaff_r=4):
        super().__init__()

        self.dim_feat = dim_feat

        # ── Backbones ──────────────────────────────────────────
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.swin = swin_tiny_patch4_window7_224(pretrained=True)
        self.mamba = mamba_vision_S(pretrained=True, model_path="./model/pretrained/mambavision_small_1k.pth.tar")

        # ── 空间注意力引导（Res2Net layer0 → patch_embed引导）──
        self.spatial_gate = SpatialGate(in_channels=256)

        # ── 串行iAFF（每个stage两个：Res→Swin, Swin→Mamba）──
        # 通道数参考：
        #   stage2: res=512,  swin=192, mamba=192
        #   stage3: res=1024, swin=384, mamba=384
        #   stage4: res=2048, swin=768, mamba=768
        stage_channels = [
            (512, 192, 192),
            (1024, 384, 384),
            (2048, 768, 768),
        ]
        self.iaff_res_swin = nn.ModuleList()
        self.iaff_swin_mamba = nn.ModuleList()
        for ch_res, ch_swin, ch_mamba in stage_channels:
            self.iaff_res_swin.append(iAFF(in_channels_1=ch_swin, in_channels_2=ch_res, out_channels=ch_swin, r=iaff_r))
            self.iaff_swin_mamba.append(
                iAFF(in_channels_1=ch_mamba, in_channels_2=ch_swin, out_channels=ch_mamba, r=iaff_r)
            )

        # ── 三路投影相加（每个stage一个ProjectionAdd）──────────
        self.proj_add = nn.ModuleList(
            [ProjectionAdd(ch_res, ch_swin, ch_mamba, dim_feat) for ch_res, ch_swin, ch_mamba in stage_channels]
        )

        # ── 全局反馈门控（Mamba stage4 → 调制stage2/3的融合特征）
        self.global_gate_s2 = GlobalFeedbackGate(ch_global=768, ch_local=dim_feat)
        self.global_gate_s3 = GlobalFeedbackGate(ch_global=768, ch_local=dim_feat)

        # ── 跳跃连接投影（Res2Net压到dim_feat再与fused相加）───
        # res_f2: 1024→dim_feat，用于14×14
        # res_f1: 512→dim_feat， 用于28×28
        self.skip_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1024, dim_feat, kernel_size=1, bias=False),
                    nn.BatchNorm2d(dim_feat),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, dim_feat, kernel_size=1, bias=False), nn.BatchNorm2d(dim_feat), nn.ReLU(inplace=True)
                ),
            ]
        )

        # ── 解码器（FPN式，Res2Net各层作为跳跃连接）────────────
        # 输入特征维度：dim_feat（7×7）
        # 跳跃连接维度：Res2Net layer2=1024(14×14), layer1=512(28×28), layer0=256(56×56)
        self.decoder = nn.ModuleList(
            [
                DecodeBlock(dim_feat, skip_channels=dim_feat, out_channels=256, scale_factor=2),  # 7→14
                DecodeBlock(256, skip_channels=dim_feat, out_channels=128, scale_factor=2),  # 14→28
                DecodeBlock(128, skip_channels=256, out_channels=64, scale_factor=2),  # 28→56
                DecodeBlock(64, skip_channels=0, out_channels=64, scale_factor=4),  # 56→224
            ]
        )

        # ── 分割输出头 ──────────────────────────────────────────
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    # ──────────────────────────────────────────────────────────
    def _extract_res2net(self, x):
        """提取Res2Net各层特征，返回layer0~layer3的输出。"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        f0 = self.resnet.layer1(x)  # (B, 256,  56×56)
        f1 = self.resnet.layer2(f0)  # (B, 512,  28×28)
        f2 = self.resnet.layer3(f1)  # (B, 1024, 14×14)
        f3 = self.resnet.layer4(f2)  # (B, 2048, 7×7)
        return f0, f1, f2, f3

    def _extract_swin(self, x, spatial_guide):
        """提取Swin各层特征，patch_embed后注入空间引导。"""
        B = x.shape[0]
        tokens, H, W = self.swin.patch_embed(x)  # (B, N, 96)
        tokens = self.swin.pos_drop(tokens)

        # 空间引导：将gate插值到patch尺度后调制token
        guide = F.interpolate(spatial_guide, size=(H, W), mode="bilinear", align_corners=False)  # (B,1,H,W)
        guide = guide.flatten(2).transpose(1, 2)  # (B, N, 1)
        tokens = tokens * guide

        tokens, H, W = self.swin.layers[0](tokens, H, W)
        f1 = tokens.permute(0, 2, 1).view(B, 192, H, W)  # (B, 192, 28×28)
        tokens, H, W = self.swin.layers[1](tokens, H, W)
        f2 = tokens.permute(0, 2, 1).view(B, 384, H, W)  # (B, 384, 14×14)
        tokens, H, W = self.swin.layers[2](tokens, H, W)
        f3 = tokens.permute(0, 2, 1).view(B, 768, H, W)  # (B, 768, 7×7)
        return f1, f2, f3

    def _extract_mamba(self, x, spatial_guide):
        """提取MambaVision各层特征，patch_embed后注入空间引导。"""
        feat = self.mamba.patch_embed(x)  # (B, 96, 56×56)

        # 空间引导
        guide = F.interpolate(spatial_guide, size=feat.shape[2:], mode="bilinear", align_corners=False)
        feat = feat * guide

        f1 = self.mamba.levels[0](feat)  # (B, 192, 28×28)
        f2 = self.mamba.levels[1](f1)  # (B, 384, 14×14)
        f3 = self.mamba.levels[2](f2)  # (B, 768, 7×7)
        return f1, f2, f3

    # ──────────────────────────────────────────────────────────
    def forward(self, x):
        # ── 1. 特征提取 ────────────────────────────────────────
        res_f0, res_f1, res_f2, res_f3 = self._extract_res2net(x)

        # 用Res2Net layer0生成空间引导
        spatial_guide = self.spatial_gate(res_f0)  # (B, 1, 56×56)

        swin_f1, swin_f2, swin_f3 = self._extract_swin(x, spatial_guide)
        mamba_f1, mamba_f2, mamba_f3 = self._extract_mamba(x, spatial_guide)

        # ── 2. 串行iAFF融合（三个stage）────────────────────────
        res_feats = [res_f1, res_f2, res_f3]
        swin_feats = [swin_f1, swin_f2, swin_f3]
        mamba_feats = [mamba_f1, mamba_f2, mamba_f3]

        fused = []
        for i in range(3):
            # Step1: Res2Net → Swin（Swin是主体，被Res2Net调制）
            swin_enh = self.iaff_res_swin[i](swin_feats[i], res_feats[i])

            # Step2: Swin_enhanced → Mamba（Mamba是主体，被增强后的Swin调制）
            mamba_enh = self.iaff_swin_mamba[i](mamba_feats[i], swin_enh)

            # Step3: 三路投影对齐后相加
            f = self.proj_add[i](res_feats[i], swin_enh, mamba_enh)
            fused.append(f)  # 每个 (B, dim_feat, H, W)

        # fused[0]: (B, dim_feat, 28×28)
        # fused[1]: (B, dim_feat, 14×14)
        # fused[2]: (B, dim_feat, 7×7)

        # ── 3. 全局反馈门控（用mamba stage4调制浅层融合特征）──
        fused[0] = self.global_gate_s2(mamba_f3, fused[0])
        fused[1] = self.global_gate_s3(mamba_f3, fused[1])

        # ── 4. 跳跃连接：Res2Net压维后与fused相加 ─────────────
        # 14×14：res_f2(1024) → dim_feat，再加fused[1]
        # 28×28：res_f1(512)  → dim_feat，再加fused[0]
        skip_14 = self.skip_proj[0](res_f2) + fused[1]  # (B, dim_feat, 14×14)
        skip_28 = self.skip_proj[1](res_f1) + fused[0]  # (B, dim_feat, 28×28)
        skip_56 = res_f0  # (B, 256,      56×56)

        # ── 5. FPN式解码器 ──────────────────────────────────────
        # 跳跃连接来自Res2Net原始输出（未经iAFF，保留完整细节）
        skips = [skip_14, skip_28, skip_56, None]

        x_dec = fused[2]  # 从最深层开始 (B, dim_feat, 7×7)
        for i, (decode_block, skip) in enumerate(zip(self.decoder, skips)):
            x_dec = decode_block(x_dec, skip)

        # ── 6. 分割输出 ─────────────────────────────────────────
        out = self.seg_head(x_dec)  # (B, num_classes, 224×224)
        return out

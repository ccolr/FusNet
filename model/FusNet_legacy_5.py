# iAFF之后传入下一层, 跳跃连接是两路增强特征（swin_enh + mamba_enh）相加

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


class TwoBranchAdd(nn.Module):
    """
    将swin_enh和mamba_enh两路特征分别用1×1conv投影到同一维度后相加。
    """

    def __init__(self, ch_swin, ch_mamba, dim_out):
        super().__init__()
        self.proj_swin = nn.Conv2d(ch_swin, dim_out, kernel_size=1, bias=False)
        self.proj_mamba = nn.Conv2d(ch_mamba, dim_out, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat_swin, feat_mamba):
        return self.act(self.norm(self.proj_swin(feat_swin) + self.proj_mamba(feat_mamba)))


class DecodeBlock(nn.Module):
    """
    单步解码：上采样 → 与跳跃连接融合 → conv提炼。
    skip_channels=0 表示该步骤没有跳跃连接。
    """

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


class FusNet(nn.Module):
    """
    三路编码器（Res2Net + Swin-T + MambaVision）串行iAFF融合
    + FPN式解码器用于遥感竹林分割。

    融合策略：iAFF增强后取swin_enh和mamba_enh两路投影相加，去掉res分支的直接参与。

    Args:
        num_classes:  分割类别数
        dim_feat:     两路投影对齐的公共特征维度（默认256）
        iaff_r:       iAFF模块的reduction ratio（默认4）
    """

    def __init__(self, num_classes=2, dim_feat=256, iaff_r=4):
        super().__init__()

        self.dim_feat = dim_feat

        # ── Backbones ──────────────────────────────────────────
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.swin = swin_tiny_patch4_window7_224(pretrained=True)
        self.mamba = mamba_vision_S(pretrained=True, model_path="./model/pretrained/mambavision_small_1k.pth.tar")

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

        # ── 两路投影相加（swin_enh + mamba_enh，每个stage一个TwoBranchAdd）──
        swin_mamba_channels = [(ch_swin, ch_mamba) for _, ch_swin, ch_mamba in stage_channels]
        self.proj_add = nn.ModuleList(
            [TwoBranchAdd(ch_swin, ch_mamba, dim_feat) for ch_swin, ch_mamba in swin_mamba_channels]
        )

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
                    nn.Conv2d(512, dim_feat, kernel_size=1, bias=False),
                    nn.BatchNorm2d(dim_feat),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        # ── 解码器（FPN式，Res2Net各层作为跳跃连接）────────────
        self.decoder = nn.ModuleList(
            [
                DecodeBlock(dim_feat, skip_channels=dim_feat, out_channels=256, scale_factor=2),  # 7→14
                DecodeBlock(256, skip_channels=dim_feat, out_channels=128, scale_factor=2),       # 14→28
                DecodeBlock(128, skip_channels=256, out_channels=64, scale_factor=2),             # 28→56
                DecodeBlock(64, skip_channels=0, out_channels=64, scale_factor=4),               # 56→224
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
        f0 = self.resnet.layer1(x)   # (B, 256,  56×56)
        f1 = self.resnet.layer2(f0)  # (B, 512,  28×28)
        f2 = self.resnet.layer3(f1)  # (B, 1024, 14×14)
        f3 = self.resnet.layer4(f2)  # (B, 2048, 7×7)
        return f0, f1, f2, f3

    def _swin_stage(self, tokens, H, W, layer_idx):
        """运行Swin单个stage，接受token输入，返回增强后的feature map和新token。"""
        B = tokens.shape[0]
        tokens, H, W = self.swin.layers[layer_idx](tokens, H, W)
        ch = tokens.shape[-1]
        feat = tokens.permute(0, 2, 1).view(B, ch, H, W)
        return tokens, H, W, feat

    def _mamba_stage(self, feat, level_idx):
        """运行MambaVision单个level，接受feature map，返回下一level的feature map。"""
        return self.mamba.levels[level_idx](feat)

    def forward(self, x):
        # ── 1. Res2Net完整提取 ─────────────────────────────────────
        res_f0, res_f1, res_f2, res_f3 = self._extract_res2net(x)

        # ── 2. Swin patch_embed ────────────────────────────────────
        swin_tokens, H, W = self.swin.patch_embed(x)  # (B, N, 96)
        swin_tokens = self.swin.pos_drop(swin_tokens)

        # ── 3. Mamba patch_embed ───────────────────────────────────
        mamba_feat = self.mamba.patch_embed(x)  # (B, 96, 56×56)

        # ── 4. 逐stage交替提取 + iAFF融合 ─────────────────────────
        res_feats = [res_f1, res_f2, res_f3]
        fused = []

        for i in range(3):
            # Swin 第 i 个stage
            swin_tokens, H, W, swin_feat = self._swin_stage(swin_tokens, H, W, layer_idx=i)

            # Mamba 第 i 个level
            mamba_feat = self._mamba_stage(mamba_feat, level_idx=i)

            # Step1: Res2Net → Swin iAFF（Swin主体，Res2Net调制）
            swin_enh = self.iaff_res_swin[i](swin_feat, res_feats[i])

            # Step2: Swin_enhanced → Mamba iAFF（Mamba主体，增强后的Swin调制）
            mamba_enh = self.iaff_swin_mamba[i](mamba_feat, swin_enh)

            # Step3: 两路投影对齐后相加（swin_enh + mamba_enh，不含res）
            f = self.proj_add[i](swin_enh, mamba_enh)
            fused.append(f)

            # 将增强后的特征回写，作为下一stage的输入
            swin_tokens = swin_enh.flatten(2).transpose(1, 2)  # (B, N, C)
            mamba_feat = mamba_enh

        # ── 5. 跳跃连接 ────────────────────────────────────────────
        skip_14 = self.skip_proj[0](res_f2) + fused[1]
        skip_28 = self.skip_proj[1](res_f1) + fused[0]
        skip_56 = res_f0

        # ── 6. FPN式解码器 ──────────────────────────────────────────
        skips = [skip_14, skip_28, skip_56, None]
        x_dec = fused[2]
        for decode_block, skip in zip(self.decoder, skips):
            x_dec = decode_block(x_dec, skip)

        # ── 7. 分割输出 ─────────────────────────────────────────────
        out = self.seg_head(x_dec)
        return out

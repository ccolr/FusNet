"""
Heterogeneous Cross-Attention (HCA) Fusion Module
==================================================
用于三路异构backbone（Res2Net / Swin-T / MambaVision）的中间特征融合。

设计原则：
  - 每路特征作为Q，另外两路concat后作为KV
  - 各路独立LayerNorm，处理异构特征分布差异
  - 残差连接保护每路backbone的主体特征
  - 支持多头注意力
  - 支持高分辨率下的KV空间降采样（控制显存）

使用示例：
    model = HCAFusion(
        in_channels=[256, 384, 320],   # 三路输入通道，可以不同
        out_channels=256,
        num_heads=8,
        kv_stride=2,                   # H/16及以上的stage建议设2
    )
    out = model(f_res2net, f_swin, f_mamba)  # 输出 (B, 256, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 1. 2D 可学习相对位置偏置
# ─────────────────────────────────────────────────────────────────────────────
class RelativePositionBias2D(nn.Module):
    """
    为 cross-attention 的 attention map 添加可学习的 2D 相对位置偏置。
    仅在 kv_stride=1（Q 和 KV 空间尺寸相同）时启用。
    """

    def __init__(self, num_heads: int, max_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.max_size = max_size
        table_size = (2 * max_size - 1) ** 2
        self.bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns: bias (num_heads, H*W, H*W)"""
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)  # (HW, 2)
        rel = coords[:, None, :] - coords[None, :, :]  # (HW, HW, 2)
        rel[:, :, 0] += self.max_size - 1
        rel[:, :, 1] += self.max_size - 1
        rel[:, :, 0] *= 2 * self.max_size - 1
        idx = rel.sum(-1)  # (HW, HW)
        bias = self.bias_table[idx]  # (HW, HW, heads)
        return bias.permute(2, 0, 1)  # (heads, HW, HW)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 单路 Cross-Attention
# ─────────────────────────────────────────────────────────────────────────────
class SingleBranchCrossAttention(nn.Module):
    """
    Q 来自本路，KV 来自另外两路按通道 concat 后联合投影。

    Args:
        dim         : 对齐后的统一通道数
        num_heads   : 注意力头数
        kv_stride   : KV 空间降采样步长（>1 节省显存）
        attn_drop   : attention map dropout
        proj_drop   : 输出 projection dropout
        use_pos_bias: 是否使用 2D 相对位置偏置
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kv_stride: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pos_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.kv_stride = kv_stride

        self.q_proj = nn.Linear(dim, dim)
        # 两路 KV 按通道 concat → 2*dim，联合投影到 dim
        self.k_proj_1 = nn.Linear(dim, dim)  # 第一路 KV 独立投影
        self.k_proj_2 = nn.Linear(dim, dim)
        self.v_proj_1 = nn.Linear(dim, dim)
        self.v_proj_2 = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.use_pos_bias = use_pos_bias and (kv_stride == 1)
        self.use_pos_bias = False
        if self.use_pos_bias:
            self.pos_bias = RelativePositionBias2D(num_heads)

        if kv_stride > 1:
            self.kv_pool = nn.AvgPool2d(kernel_size=kv_stride, stride=kv_stride)

    def forward(self, x_q, x_k1, x_k2):
        """
        Args:
            x_q  : (B, C, H, W)  本路特征，作为 Query
            x_k1 : (B, C, H, W)  第一路 KV 来源
            x_k2 : (B, C, H, W)  第二路 KV 来源
        Returns:
            out  : (B, C, H, W)  增强后的特征（含残差）
        """
        B, C, H, W = x_q.shape

        # Query
        q = rearrange(x_q, "b c h w -> b (h w) c")
        q = self.q_proj(q)  # (B, HW, C)

        # KV 可选降采样
        if self.kv_stride > 1:
            x_k1 = self.kv_pool(x_k1)
            x_k2 = self.kv_pool(x_k2)

        # 两路 KV 按通道 concat → 联合 K/V 投影
        kv1 = rearrange(x_k1, "b c h w -> b (h w) c")
        kv2 = rearrange(x_k2, "b c h w -> b (h w) c")

        k = self.k_proj_1(kv1) + self.k_proj_2(kv2)  # 分开投影后相加
        v = self.v_proj_1(kv1) + self.v_proj_2(kv2)

        # kv1 = rearrange(x_k1, "b c h w -> b (h w) c")  # (B, HkWk, C)
        # kv2 = rearrange(x_k2, "b c h w -> b (h w) c")  # (B, HkWk, C)
        # kv = torch.cat([kv1, kv2], dim=-1)  # (B, HkWk, 2C)
        # k = self.k_proj(kv)  # (B, HkWk, C)
        # v = self.v_proj(kv)  # (B, HkWk, C)

        # 多头 reshape
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)

        # Attention
        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale

        if self.use_pos_bias:
            bias = self.pos_bias(H, W, x_q.device)  # (heads, HW, HW)
            attn = attn + bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # 聚合 + 输出投影
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)  # (B, heads, HW, d)
        out = rearrange(out, "b h n d -> b n (h d)")  # (B, HW, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        # 还原空间维度 + 残差
        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return x_q + out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 完整 HCA Fusion 模块
# ─────────────────────────────────────────────────────────────────────────────
class HCABlock(nn.Module):
    """
    Heterogeneous Cross-Attention Fusion

    三路异构 backbone 特征融合模块：
      Step 1 : 通道对齐（1x1 Conv）
      Step 2 : 各路独立 LayerNorm（处理异构分布）
      Step 3 : 三路并行 Cross-Attention（每路 Q 询问其他两路 KV）
      Step 4 : 各路独立 FFN（Pre-Norm + 残差）
      Step 5 : 三路 concat + 1x1 Conv 降维输出

    Args:
        in_channels  : 三路输入通道数，如 [256, 384, 320]（可不同）
        out_channels : 输出通道数
        num_heads    : 多头注意力头数
        kv_stride    : KV 空间降采样步长（H/8 分辨率建议 2，H/32 可设 1）
        ffn_ratio    : FFN 隐层相对于 out_channels 的扩张比
        attn_drop    : attention dropout
        proj_drop    : projection dropout
        use_pos_bias : 是否使用 2D 相对位置偏置
    """

    def __init__(
        self,
        in_channels: list,
        out_channels: int,
        num_heads: int = 8,
        kv_stride: int = 1,
        ffn_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pos_bias: bool = True,
    ):
        super().__init__()
        assert len(in_channels) == 3
        C_r, C_s, C_m = in_channels
        dim = out_channels

        # Step 1: 通道对齐
        self.proj_r = self._make_proj(C_r, dim)
        self.proj_s = self._make_proj(C_s, dim)
        self.proj_m = self._make_proj(C_m, dim)

        # Step 2: 各路独立 LayerNorm
        self.norm_r = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)

        # Step 3: 三路对称 Cross-Attention
        ca_kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            kv_stride=kv_stride,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_pos_bias=use_pos_bias,
        )
        self.ca_r = SingleBranchCrossAttention(**ca_kwargs)
        self.ca_s = SingleBranchCrossAttention(**ca_kwargs)
        self.ca_m = SingleBranchCrossAttention(**ca_kwargs)

        # Step 4: 各路 FFN（Pre-Norm）
        hidden = int(dim * ffn_ratio)
        self.norm_r2 = nn.LayerNorm(dim)
        self.norm_s2 = nn.LayerNorm(dim)
        self.norm_m2 = nn.LayerNorm(dim)
        self.ffn_r = self._make_ffn(dim, hidden)
        self.ffn_s = self._make_ffn(dim, hidden)
        self.ffn_m = self._make_ffn(dim, hidden)

        # Step 5: 三路 concat → 降维
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(dim * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.post_ca_norm_r = nn.LayerNorm(dim)  # 注意这里 dim 变量名跟你的一致用 dim
        self.post_ca_norm_s = nn.LayerNorm(dim)
        self.post_ca_norm_m = nn.LayerNorm(dim)

    @staticmethod
    def _make_proj(in_c, out_c):
        if in_c == out_c:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
        )

    @staticmethod
    def _make_ffn(dim, hidden):
        return nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def _spatial_norm(self, x, norm):
        """(B, C, H, W) → LayerNorm on C → (B, C, H, W)"""
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = norm(x)
        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

    def _apply_ffn(self, x, norm, ffn):
        """Pre-Norm FFN with residual"""
        B, C, H, W = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x_flat = x_flat + ffn(norm(x_flat))
        return rearrange(x_flat, "b (h w) c -> b c h w", h=H, w=W)

    def forward(self, f_res2net, f_swin, f_mamba):
        """
        Args:
            f_res2net : (B, C_r, H, W)
            f_swin    : (B, C_s, H, W)
            f_mamba   : (B, C_m, H, W)
        Returns:
            fused     : (B, out_channels, H, W)
        """
        # Step 1: 通道对齐
        r = self.proj_r(f_res2net)
        s = self.proj_s(f_swin)
        m = self.proj_m(f_mamba)

        # Step 2: 各路独立 LayerNorm
        r = self._spatial_norm(r, self.norm_r)
        s = self._spatial_norm(s, self.norm_s)
        m = self._spatial_norm(m, self.norm_m)

        # Step 3: 三路并行 Cross-Attention
        r_out = self.ca_r(r, s, m)  # Q=Res2Net, KV=Swin+Mamba
        s_out = self.ca_s(s, r, m)  # Q=Swin,    KV=Res2Net+Mamba
        m_out = self.ca_m(m, r, s)  # Q=Mamba,   KV=Res2Net+Swin

        r_out = self._spatial_norm(r_out, self.post_ca_norm_r)
        s_out = self._spatial_norm(s_out, self.post_ca_norm_s)
        m_out = self._spatial_norm(m_out, self.post_ca_norm_m)

        # Step 4: 各路 FFN
        r_out = self._apply_ffn(r_out, self.norm_r2, self.ffn_r)
        s_out = self._apply_ffn(s_out, self.norm_s2, self.ffn_s)
        m_out = self._apply_ffn(m_out, self.norm_m2, self.ffn_m)

        # Step 5: concat + 降维
        fused = torch.cat([r_out, s_out, m_out], dim=1)  # (B, 3*dim, H, W)
        return self.fusion_proj(fused)  # (B, out_channels, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 快速功能测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, H, W = 2, 32, 32

    f_r = torch.randn(B, 256, H, W)  # Res2Net
    f_s = torch.randn(B, 384, H, W)  # Swin-T
    f_m = torch.randn(B, 320, H, W)  # MambaVision

    model = HCABlock(
        in_channels=[256, 384, 320],
        out_channels=256,
        num_heads=8,
        kv_stride=2,
        ffn_ratio=4.0,
        attn_drop=0.1,
        proj_drop=0.1,
        use_pos_bias=True,
    )

    out = model(f_r, f_s, f_m)
    print(f"输入:  Res2Net {f_r.shape}, Swin {f_s.shape}, Mamba {f_m.shape}")
    print(f"输出:  {out.shape}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"参数量: {total / 1e6:.2f} M")

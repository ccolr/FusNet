"""
Heterogeneous Cross-Attention (HCA) Fusion Module
==================================================
支持三路或任意两路的可配置融合，用于消融实验。

active_branches 参数决定使用哪些路：
    ["res2net", "swin", "mamba"]   → 三路（原始）
    ["res2net", "swin"]            → 消融：去掉 MambaVision
    ["res2net", "mamba"]           → 消融：去掉 Swin-T
    ["swin",    "mamba"]           → 消融：去掉 Res2Net

使用示例:
    model = HCABlock(
        in_channels={"res2net": 256, "swin": 96, "mamba": 96},
        out_channels=256,
        num_heads=8,
        kv_stride=2,
        active_branches=["res2net", "swin"],   # ← 消融时改这里
    )
    out = model(f_res2net=f_r, f_swin=f_s)     # 只传激活的两路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

ALL_BRANCHES = ["res2net", "swin", "mamba"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 2D 可学习相对位置偏置
# ─────────────────────────────────────────────────────────────────────────────
class RelativePositionBias2D(nn.Module):
    def __init__(self, num_heads: int, max_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.max_size = max_size
        table_size = (2 * max_size - 1) ** 2
        self.bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).reshape(-1, 2)
        rel = coords[:, None, :] - coords[None, :, :]
        rel[:, :, 0] += self.max_size - 1
        rel[:, :, 1] += self.max_size - 1
        rel[:, :, 0] *= 2 * self.max_size - 1
        idx = rel.sum(-1)
        bias = self.bias_table[idx]
        return bias.permute(2, 0, 1)  # (heads, HW, HW)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 单路 Cross-Attention（支持单KV源 or 双KV源）
# ─────────────────────────────────────────────────────────────────────────────
class SingleBranchCrossAttention(nn.Module):
    """
    Q 来自本路，KV 来自另外 n_kv 路（1 或 2）。

    当 n_kv=1 时（两路消融），只有一组 k_proj / v_proj。
    当 n_kv=2 时（三路），两组分别投影后相加，与原始实现一致。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        kv_stride: int = 1,
        n_kv: int = 2,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pos_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        assert n_kv in (1, 2), "n_kv 只支持 1 或 2"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_stride = kv_stride
        self.n_kv = n_kv

        self.q_proj = nn.Linear(dim, dim)
        # 为每路 KV 建立独立的 k/v 投影
        self.k_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_kv)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_kv)])
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_pos_bias = False  # 与原实现保持一致，暂关闭
        if self.use_pos_bias:
            self.pos_bias = RelativePositionBias2D(num_heads)

        if kv_stride > 1:
            self.kv_pool = nn.AvgPool2d(kernel_size=kv_stride, stride=kv_stride)

    def forward(self, x_q, *x_kvs):
        """
        Args:
            x_q    : (B, C, H, W)  本路特征，作为 Query
            *x_kvs : n_kv 个 (B, C, H, W)，作为 KV 来源
        Returns:
            out    : (B, C, H, W)  增强后的特征（含残差）
        """
        assert len(x_kvs) == self.n_kv
        B, C, H, W = x_q.shape

        q = rearrange(x_q, "b c h w -> b (h w) c")
        q = self.q_proj(q)

        # KV 降采样
        x_kvs_pooled = []
        for xk in x_kvs:
            if self.kv_stride > 1:
                xk = self.kv_pool(xk)
            x_kvs_pooled.append(rearrange(xk, "b c h w -> b (h w) c"))

        # 各路 KV 独立投影后相加（n_kv=1 时退化为单路）
        k = sum(self.k_projs[i](x_kvs_pooled[i]) for i in range(self.n_kv))
        v = sum(self.v_projs[i](x_kvs_pooled[i]) for i in range(self.n_kv))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)

        attn = torch.einsum("bhnd,bhmd->bhnm", q, k) * self.scale

        if self.use_pos_bias:
            bias = self.pos_bias(H, W, x_q.device)
            attn = attn + bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        out = self.proj_drop(out)

        out = rearrange(out, "b (h w) c -> b c h w", h=H, w=W)
        return x_q + out


# ─────────────────────────────────────────────────────────────────────────────
# 3. 完整 HCA Fusion 模块（可配置路数）
# ─────────────────────────────────────────────────────────────────────────────
class HCABlock(nn.Module):
    """
    Heterogeneous Cross-Attention Fusion（支持 2 路或 3 路）

    Args:
        in_channels    : dict，键为 branch 名称，值为通道数
                         e.g. {"res2net": 256, "swin": 96, "mamba": 96}
                         也可直接传 list（按 res2net/swin/mamba 顺序，兼容旧接口）
        out_channels   : 输出通道数
        num_heads      : 多头注意力头数
        kv_stride      : KV 空间降采样步长
        ffn_ratio      : FFN 扩张比
        attn_drop      : attention dropout
        proj_drop      : projection dropout
        use_pos_bias   : 是否使用 2D 相对位置偏置（当前实现内部强制关闭）
        active_branches: 激活的 branch 列表，如 ["res2net", "swin"]
                         None 表示全部三路（默认行为）
    """

    def __init__(
        self,
        in_channels,
        out_channels: int,
        num_heads: int = 8,
        kv_stride: int = 1,
        ffn_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_pos_bias: bool = True,
        active_branches=None,
    ):
        super().__init__()

        # ── 解析 active_branches ──────────────────────────────
        if active_branches is None:
            active_branches = ALL_BRANCHES
        assert 2 <= len(active_branches) <= 3, "active_branches 需包含 2~3 个 branch"
        for b in active_branches:
            assert b in ALL_BRANCHES, f"未知 branch: {b}"
        self.active_branches = list(active_branches)
        n_branches = len(self.active_branches)
        n_kv = n_branches - 1  # 每路的 KV 来源数量

        # ── 解析 in_channels（兼容 list 旧接口）──────────────
        if isinstance(in_channels, (list, tuple)):
            assert len(in_channels) == 3, "list 形式需按 [res2net, swin, mamba] 顺序传入三路通道"
            in_ch_dict = dict(zip(ALL_BRANCHES, in_channels))
        else:
            in_ch_dict = in_channels
        self.in_ch_dict = in_ch_dict

        dim = out_channels

        # ── Step 1: 通道对齐（只建激活路）────────────────────
        self.proj = nn.ModuleDict({
            b: self._make_proj(in_ch_dict[b], dim)
            for b in self.active_branches
        })

        # ── Step 2 & 后处理: LayerNorm ──────────────────────
        self.norm_pre = nn.ModuleDict({b: nn.LayerNorm(dim) for b in self.active_branches})
        self.norm_post_ca = nn.ModuleDict({b: nn.LayerNorm(dim) for b in self.active_branches})
        self.norm_ffn = nn.ModuleDict({b: nn.LayerNorm(dim) for b in self.active_branches})

        # ── Step 3: Cross-Attention（每路 Q 询问其余路 KV）──
        ca_kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            kv_stride=kv_stride,
            n_kv=n_kv,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_pos_bias=use_pos_bias,
        )
        self.ca = nn.ModuleDict({
            b: SingleBranchCrossAttention(**ca_kwargs)
            for b in self.active_branches
        })

        # ── Step 4: FFN ──────────────────────────────────────
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.ModuleDict({
            b: self._make_ffn(dim, hidden)
            for b in self.active_branches
        })

        # ── Step 5: concat + 降维 ────────────────────────────
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(dim * n_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

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
        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = norm(x)
        return rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

    def _apply_ffn(self, x, norm, ffn):
        B, C, H, W = x.shape
        x_flat = rearrange(x, "b c h w -> b (h w) c")
        x_flat = x_flat + ffn(norm(x_flat))
        return rearrange(x_flat, "b (h w) c -> b c h w", h=H, w=W)

    def forward(self, f_res2net=None, f_swin=None, f_mamba=None):
        """
        Args:
            f_res2net : (B, C_r, H, W) 或 None（若未激活）
            f_swin    : (B, C_s, H, W) 或 None
            f_mamba   : (B, C_m, H, W) 或 None
        Returns:
            fused     : (B, out_channels, H, W)
        """
        inputs = {"res2net": f_res2net, "swin": f_swin, "mamba": f_mamba}

        # 检查激活路都有输入
        for b in self.active_branches:
            assert inputs[b] is not None, f"branch '{b}' 已激活但未传入特征"

        # Step 1: 通道对齐
        feats = {b: self.proj[b](inputs[b]) for b in self.active_branches}

        # Step 2: Pre-Norm
        feats = {b: self._spatial_norm(feats[b], self.norm_pre[b]) for b in self.active_branches}

        # Step 3: Cross-Attention
        # 对每路 b，其 KV 来自其余路的有序列表
        ca_out = {}
        for b in self.active_branches:
            kv_sources = [feats[other] for other in self.active_branches if other != b]
            ca_out[b] = self.ca[b](feats[b], *kv_sources)

        # Post-CA Norm
        ca_out = {b: self._spatial_norm(ca_out[b], self.norm_post_ca[b]) for b in self.active_branches}

        # Step 4: FFN
        out = {b: self._apply_ffn(ca_out[b], self.norm_ffn[b], self.ffn[b]) for b in self.active_branches}

        # Step 5: concat + 降维（按固定顺序保证可复现）
        ordered = [out[b] for b in self.active_branches]
        fused = torch.cat(ordered, dim=1)
        return self.fusion_proj(fused)


# ─────────────────────────────────────────────────────────────────────────────
# 快速功能测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, H, W = 2, 32, 32

    f_r = torch.randn(B, 256, H, W)
    f_s = torch.randn(B, 384, H, W)
    f_m = torch.randn(B, 320, H, W)

    for branches in [["res2net", "swin"], ["res2net", "mamba"], ["swin", "mamba"], ["res2net", "swin", "mamba"]]:
        in_ch = {"res2net": 256, "swin": 384, "mamba": 320}
        model = HCABlock(
            in_channels=in_ch,
            out_channels=256,
            num_heads=8,
            kv_stride=2,
            active_branches=branches,
        )
        kwargs = {}
        if "res2net" in branches: kwargs["f_res2net"] = f_r
        if "swin"    in branches: kwargs["f_swin"]    = f_s
        if "mamba"   in branches: kwargs["f_mamba"]   = f_m

        out = model(**kwargs)
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"branches={branches}  输出={out.shape}  参数={total/1e6:.2f}M")

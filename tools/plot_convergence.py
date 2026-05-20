"""
Dot plots of val mIoU and IoU (Bamboo) across all models.
X-axis: model name, Y-axis: metric value.
FusNet (Res+Swin+Mamba) is highlighted with a red star.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
})

# ── Data ──────────────────────────────────────────────────────────────────────
# (label, best_iou, iou_ep, best_f1, f1_ep, best_miou, miou_ep, best_recall, recall_ep)
EXPERIMENTS = [
    # Standard baselines
    ("DeepLabV3",               0.804773, 90,  0.891827, 90,  0.835981, 109, 0.924292, 23),
    ("DeepLabV3+",              0.810907, 95,  0.895581, 95,  0.841893, 81,  0.942071, 2),
    ("SwinUNet",                0.818530, 85,  0.900211, 85,  0.848210, 85,  0.956127, 11),
    # Single-backbone models
    ("Res2Net",                 0.806253, 96,  0.892735, 96,  0.837736, 96,  0.934519, 51),
    ("Swin-T",                  0.809181, 107, 0.894527, 107, 0.840291, 107, 0.978039, 1),
    ("MambaVision",             0.818072, 61,  0.899933, 61,  0.848131, 87,  0.948982, 6),
    # FusNet baseline fusion variants
    ("FusNet-Base(add)",        0.820210, 50,  0.901225, 50,  0.849000, 55,  0.942053, 32),
    ("FusNet-Base(cat)",        0.819274, 109, 0.900660, 109, 0.848134, 113, 0.991136, 1),
    # Cross-attention ablations (2-backbone)
    ("Res+Swin",                0.810552, 63,  0.895365, 63,  0.841757, 78,  0.975696, 1),
    ("Res+Mamba",               0.819352, 99,  0.900708, 99,  0.848477, 98,  0.942908, 3),
    ("Swin+Mamba",              0.821405, 63,  0.901946, 63,  0.850784, 63,  0.943722, 1),
    # Full FusNet — MUST remain last
    ("FusNet\n(Res+Swin+Mamba)", 0.822783, 36, 0.902777, 36, 0.851179, 67, 0.958380, 9),
]

BASELINE_COLORS = [
    "#4878CF",  # DeepLabV3
    "#2196F3",  # DeepLabV3+
    "#6ACC65",  # SwinUNet
    "#FF5722",  # Res2Net
    "#00ACC1",  # Swin-T
    "#9C27B0",  # MambaVision
    "#FF9800",  # FusNet-Base(add)
    "#FFC107",  # FusNet-Base(cat)
    "#D65F5F",  # Res+Swin
    "#B47CC7",  # Res+Mamba
    "#C4AD66",  # Swin+Mamba
]
FUSNET_COLOR = "#D62728"


def _draw_dot_panel(ax, val_col, ylabel, title):
    labels = [e[0].replace("\n", " ") for e in EXPERIMENTS]
    values = [e[val_col] for e in EXPERIMENTS]
    n = len(labels)
    x = np.arange(n)

    all_vals = values
    val_min, val_max = min(all_vals), max(all_vals)
    margin = (val_max - val_min) * 0.5

    # Dashed horizontal reference line at FusNet value
    fusnet_val = values[-1]
    ax.axhline(fusnet_val, color=FUSNET_COLOR, linewidth=0.9,
               linestyle="--", alpha=0.5, zorder=1)

    # Vertical grid lines (light)
    for xi in x:
        ax.axvline(xi, color="#cccccc", linewidth=0.5, linestyle=":", zorder=0)

    # Baseline models
    for i in range(n - 1):
        ax.scatter(x[i], values[i],
                   color=BASELINE_COLORS[i], marker="o", s=60, zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.annotate(f"{values[i]:.4f}",
                    xy=(x[i], values[i]),
                    xytext=(0, -8), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5,
                    color=BASELINE_COLORS[i])

    # FusNet (red star, larger)
    ax.scatter(x[-1], fusnet_val,
               color=FUSNET_COLOR, marker="*", s=340, zorder=5,
               edgecolors="white", linewidths=0.4)
    ax.annotate(f"{fusnet_val:.4f}",
                xy=(x[-1], fusnet_val),
                xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", fontsize=7,
                color=FUSNET_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(val_min - margin, val_max + margin)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    ax.grid(True, axis="y", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure: two separate files ────────────────────────────────────────────────
PANELS = [
    dict(val_col=5, ylabel="Val mIoU",      title="Val mIoU by Model",       stem="convergence_miou"),
    dict(val_col=1, ylabel="IoU (Bamboo)",  title="IoU (Bamboo) by Model",   stem="convergence_iou"),
]

for panel in PANELS:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_dot_panel(ax, val_col=panel["val_col"],
                    ylabel=panel["ylabel"], title=panel["title"])
    fig.tight_layout(pad=2.0)
    for fmt in ("pdf", "png"):
        path = f"illustrator/{panel['stem']}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)

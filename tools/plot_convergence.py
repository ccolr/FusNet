"""
Plot best IoU(Bamboo) and F1 vs. convergence epoch for each experiment.
FusNet (Res+Swin+Mamba) is highlighted with a red star to show it reaches
the highest metric values at the fewest epochs.
"""

import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
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
# (label, best_iou, iou_epoch, best_f1, f1_epoch)
EXPERIMENTS = [
    ("DeepLabV3",       0.804773, 90,  0.891827, 90),
    ("SwinUNet",        0.818530, 85,  0.900211, 85),
    ("Res+Swin",        0.810552, 63,  0.895365, 63),
    ("Res+Mamba",       0.819352, 99,  0.900708, 99),
    ("Swin+Mamba",      0.821405, 63,  0.901946, 63),
    ("FusNet\n(Res+Swin+Mamba)", 0.822783, 36, 0.902777, 36),
]

BASELINE_STYLES = [
    dict(color="#4878CF", marker="o",  s=70,   zorder=3),   # DeepLabV3
    dict(color="#6ACC65", marker="s",  s=60,   zorder=3),   # SwinUNet
    dict(color="#D65F5F", marker="^",  s=70,   zorder=3),   # Res+Swin
    dict(color="#B47CC7", marker="D",  s=60,   zorder=3),   # Res+Mamba
    dict(color="#C4AD66", marker="p",  s=70,   zorder=3),   # Swin+Mamba
]
FUSNET_STYLE = dict(color="#D62728", marker="*", s=380, zorder=5)

# ── Annotation offsets: (dx, dy, ha)
# FusNet and SwinUNet → bottom-left (dx<0, ha="right")
# others              → bottom-right (dx>0, ha="left")
IOU_OFFSETS = [
    ( 1, -0.0010, "left"),    # DeepLabV3
    (-2, -0.0010, "right"),   # SwinUNet   ← left-down
    ( 1, -0.0010, "left"),    # Res+Swin
    ( 1, -0.0010, "left"),    # Res+Mamba
    ( 1, -0.0010, "left"),    # Swin+Mamba
    (-2, -0.0011, "right"),   # FusNet     ← left-down
]
F1_OFFSETS = [
    ( 1, -0.0010, "left"),
    (-2, -0.0010, "right"),   # SwinUNet   ← left-down
    ( 1, -0.0010, "left"),
    ( 1, -0.0010, "left"),
    ( 1, -0.0010, "left"),
    (-2, -0.0011, "right"),   # FusNet     ← left-down
]


def _draw_panel(ax, epoch_col, val_col, offsets, ylabel, title):
    baselines = EXPERIMENTS[:-1]
    fusnet    = EXPERIMENTS[-1]

    # ── Dashed horizontal line at FusNet best value ───────────────────────────
    fusnet_val = fusnet[val_col]
    ax.axhline(fusnet_val, color="#D62728", linewidth=0.8,
               linestyle="--", alpha=0.45, zorder=1)

    # ── Baseline models ───────────────────────────────────────────────────────
    for i, exp in enumerate(baselines):
        label, *cols = exp
        ep  = cols[epoch_col - 1]
        val = cols[val_col   - 1]
        st  = BASELINE_STYLES[i]
        ax.scatter(ep, val, label=label.replace("\n", " "), **st)
        dx, dy, ha = offsets[i]
        ax.annotate(f"{val:.4f}",
                    xy=(ep, val), xytext=(ep + dx, val + dy),
                    fontsize=8.5, color=st["color"],
                    arrowprops=None, va="top", ha=ha)

    # ── FusNet ────────────────────────────────────────────────────────────────
    ep_f  = fusnet[epoch_col]
    val_f = fusnet[val_col]
    ax.scatter(ep_f, val_f,
               label="FusNet (Res+Swin+Mamba)", **FUSNET_STYLE)
    dx, dy, ha = offsets[-1]
    ax.annotate(f"{val_f:.4f}",
                xy=(ep_f, val_f), xytext=(ep_f + dx, val_f + dy),
                fontsize=8.5, fontweight="bold", color="#D62728",
                va="top", ha=ha)

    # ── Epoch arrow annotation for FusNet ────────────────────────────────────
    ax.annotate(f"Epoch {ep_f}",
                xy=(ep_f, val_f),
                xytext=(ep_f + 5, val_f - 0.004),
                fontsize=8, color="#D62728", style="italic",
                arrowprops=dict(arrowstyle="-", color="#D62728",
                                lw=0.8, alpha=0.7))

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Epoch", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlim(0, 125)
    all_vals = [e[val_col] for e in EXPERIMENTS]
    margin = (max(all_vals) - min(all_vals)) * 0.6
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
    ax.grid(True, which="major", linestyle="--")
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_single(epoch_col, val_col, offsets, ylabel, title, stem):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    _draw_panel(ax, epoch_col=epoch_col, val_col=val_col,
                offsets=offsets, ylabel=ylabel, title=title)
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels,
                    loc="lower right", ncol=2,
                    frameon=True, framealpha=0.9,
                    edgecolor="#cccccc",
                    handletextpad=0.4, columnspacing=0.8,
                    fontsize=8.5)
    for handle, lbl in zip(leg.legend_handles, labels):
        if "FusNet" in lbl:
            handle.set_sizes([80])
    for fmt in ("pdf", "png"):
        path = f"tools/{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)


# epoch_col / val_col index into each row tuple (after label):
#   row = (label, iou_val, iou_ep, f1_val, f1_ep)
#   index 1 = iou_val, 2 = iou_ep, 3 = f1_val, 4 = f1_ep
_save_single(epoch_col=2, val_col=1, offsets=IOU_OFFSETS,
             ylabel="IoU (Bamboo)",
             title="Best IoU (Bamboo) vs. Convergence Epoch",
             stem="convergence_iou")

_save_single(epoch_col=4, val_col=3, offsets=F1_OFFSETS,
             ylabel="F1 Score",
             title="Best F1 Score vs. Convergence Epoch",
             stem="convergence_f1")

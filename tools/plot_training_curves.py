"""
Validation mIoU and F1 training curves for all experiments.
Curves are lightly smoothed (rolling mean, window=5) for readability.
FusNet (Res+Swin+Mamba) is drawn in red with a thicker line.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

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

# ── Data sources ──────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "..", "output")
MODELS = [
    # Standard baselines
    ("DeepLabV3",             "deeplabv3_outputs",              "#4878CF", "-",   1.2),
    ("DeepLabV3+",            "deeplabv3plus_outputs",          "#2196F3", "--",  1.2),
    ("SwinUNet",              "swinunet_outputs",               "#6ACC65", "--",  1.2),
    # Single-backbone models
    ("Res2Net",               "res2net_outputs",                "#FF5722", "-.",  1.2),
    ("Swin-T",                "swin_outputs",                   "#00ACC1", ":",   1.3),
    ("MambaVision",           "mambavision_outputs",            "#9C27B0", "-",   1.2),
    # FusNet baseline fusion variants
    ("FusNet-Base(add)",      "fusnet_baseline_add_outputs",    "#FF9800", "--",  1.2),
    ("FusNet-Base(cat)",      "fusnet_baseline_cat_outputs",    "#FFC107", "-.",  1.2),
    # FusNet legacy/ablation architectures
    ("Legacy-1",              "fusnet_legacy_1_outputs",        "#E53935", "-",   1.2),
    ("Legacy-2",              "fusnet_legacy_2_outputs",        "#E91E63", "--",  1.2),
    ("Legacy-3",              "fusnet_legacy_3_outputs",        "#673AB7", "-.",  1.2),
    ("Legacy-4",              "fusnet_legacy_4_outputs",        "#3F51B5", ":",   1.2),
    ("Legacy-5",              "fusnet_legacy_5_outputs",        "#03A9F4", "-",   1.2),
    # Cross-attention ablations (2-backbone)
    ("Res+Swin",              "fusnet_outputs_res_swin",        "#D65F5F", "-.",  1.2),
    ("Res+Mamba",             "fusnet_outputs_res_mamba",       "#B47CC7", ":",   1.4),
    ("Swin+Mamba",            "fusnet_outputs_swin_mamba",      "#C4AD66", "--",  1.2),
    # Full FusNet
    ("FusNet (Res+Swin+Mamba)", "fusnet_outputs_res_swin_mamba", "#D62728", "-", 2.2),
]
SMOOTH_WIN = 5


def _load(folder):
    path = os.path.join(BASE, folder, "log.txt")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, on_bad_lines="skip")
    df = df[pd.to_numeric(df["Epoch"], errors="coerce").notna()].copy()
    df["Epoch"] = df["Epoch"].astype(int)
    return df


def _smooth(s):
    return s.rolling(window=SMOOTH_WIN, min_periods=1, center=True).mean()


def _save_curve(col, ylabel, title, stem):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for label, folder, color, ls, lw in MODELS:
        df = _load(folder)
        if df is None:
            print(f"Skipping {label}: {folder}/log.txt not found")
            continue
        epochs = df["Epoch"]
        vals   = _smooth(df[col])
        is_fusnet = label.startswith("FusNet (")
        ax.plot(epochs, vals,
                color=color, linestyle=ls, linewidth=lw,
                label=label,
                zorder=5 if is_fusnet else 3,
                alpha=1.0 if is_fusnet else 0.80)

        # mark best point
        best_idx = df[col].idxmax()
        best_ep  = df.loc[best_idx, "Epoch"]
        best_val = df.loc[best_idx, col]
        marker   = "*" if is_fusnet else "o"
        msize    = 12  if is_fusnet else 5
        ax.scatter(best_ep, best_val,
                   color=color, marker=marker, s=msize**2,
                   zorder=6 if is_fusnet else 4,
                   edgecolors="white" if not is_fusnet else color,
                   linewidths=0.6)

    ax.set_xlabel("Epoch", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, pad=8, fontweight="bold")
    ax.set_xlim(1, 128)
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True, which="major", linestyle="--")
    ax.grid(True, which="minor", linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right", frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fontsize=8.0, handlelength=2.2,
              handletextpad=0.5, ncol=3, columnspacing=1.0)

    for fmt in ("pdf", "png"):
        path = os.path.join(os.path.dirname(__file__), f"{stem}.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"Saved: {path}")
    plt.close(fig)


_save_curve("Val_mIoU", "Val mIoU",      "Validation mIoU over Training Epochs",      "training_curve_miou")
_save_curve("Val_F1",   "Val F1 Score",  "Validation F1 Score over Training Epochs",  "training_curve_f1")
_save_curve("Val_Acc",  "Val Accuracy",  "Validation Accuracy over Training Epochs",  "training_curve_acc")

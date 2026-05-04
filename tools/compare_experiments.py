"""
Compare training metrics across all 6 experiments in output/.
Generates a multi-panel figure and prints per-metric best results to terminal.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")

EXPERIMENTS = {
    "DeepLabV3": "deeplabv3_outputs",
    "SwinUNet": "swinunet_outputs",
    "Res+Swin": "fusnet_outputs_res_swin",
    "Res+Mamba": "fusnet_outputs_res_mamba",
    "Swin+Mamba": "fusnet_outputs_swin_mamba",
    "Res+Swin+Mamba": "fusnet_outputs_res_swin_mamba",
}

# Validation metrics to compare (lower-is-better metrics handled separately)
VAL_METRICS = [
    ("Val_Loss",        "Val Loss",         True),   # (col, label, lower_is_better)
    ("Val_Acc",         "Val Accuracy",     False),
    ("Val_mIoU",        "Val mIoU",         False),
    ("Val_IoU_Bamboo",  "Val IoU (Bamboo)", False),
    ("Val_F1",          "Val F1",           False),
    ("Val_Recall",      "Val Recall",       False),
]

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3", "#a65628"]
LINESTYLES = ["-", "-", "--", "--", "-.", "-."]


def load_logs():
    dfs = {}
    for name, folder in EXPERIMENTS.items():
        path = os.path.join(OUTPUT_ROOT, folder, "log.txt")
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path}")
            continue
        df = pd.read_csv(path)
        # Drop duplicate header rows that some trainers emit on resume
        df = df[df["Epoch"] != "Epoch"].reset_index(drop=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        dfs[name] = df
    return dfs


def plot_curves(dfs, save_path):
    n_metrics = len(VAL_METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    exp_names = list(dfs.keys())

    for ax_idx, (col, label, lower_better) in enumerate(VAL_METRICS):
        ax = axes[ax_idx]
        for i, name in enumerate(exp_names):
            df = dfs[name]
            if col not in df.columns:
                continue
            ax.plot(
                df["Epoch"],
                df[col],
                label=name,
                color=COLORS[i % len(COLORS)],
                linestyle=LINESTYLES[i % len(LINESTYLES)],
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best")

        # Mark global best epoch for best experiment
        best_name = None
        best_val = None
        for name in exp_names:
            df = dfs[name]
            if col not in df.columns:
                continue
            series = df[col].dropna()
            v = series.min() if lower_better else series.max()
            if best_val is None or (lower_better and v < best_val) or (not lower_better and v > best_val):
                best_val = v
                best_epoch = df.loc[series.idxmin() if lower_better else series.idxmax(), "Epoch"]
                best_name = name
        if best_name is not None:
            ax.axhline(
                best_val,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
                label=f"best={best_val:.4f}",
            )

    # Hide unused axes
    for ax_idx in range(n_metrics, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("Experiment Comparison — Validation Metrics", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {save_path}\n")
    plt.show()


def print_summary(dfs):
    # Collect best values per metric per experiment
    rows = []
    exp_names = list(dfs.keys())

    print("=" * 80)
    print("EXPERIMENT METRICS SUMMARY (Validation)")
    print("=" * 80)

    for col, label, lower_better in VAL_METRICS:
        print(f"\n{'─' * 60}")
        print(f"  {label}  ({'lower is better' if lower_better else 'higher is better'})")
        print(f"{'─' * 60}")
        print(f"  {'Experiment':<22} {'Best Value':>12}  {'Epoch':>6}")
        print(f"  {'─'*22} {'─'*12}  {'─'*6}")

        results = []
        for name in exp_names:
            df = dfs[name]
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            if lower_better:
                best_val = series.min()
                best_epoch = int(df.loc[series.idxmin(), "Epoch"])
            else:
                best_val = series.max()
                best_epoch = int(df.loc[series.idxmax(), "Epoch"])
            results.append((name, best_val, best_epoch))

        if lower_better:
            results.sort(key=lambda x: x[1])
        else:
            results.sort(key=lambda x: x[1], reverse=True)

        for rank, (name, val, epoch) in enumerate(results):
            marker = " <<<  BEST" if rank == 0 else ""
            print(f"  {name:<22} {val:>12.6f}  {epoch:>6}{marker}")

    print(f"\n{'=' * 80}")
    print("OVERALL WINNER PER METRIC")
    print("=" * 80)

    win_count = {n: 0 for n in exp_names}
    for col, label, lower_better in VAL_METRICS:
        best_name = None
        best_val = None
        for name in exp_names:
            df = dfs[name]
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            v = series.min() if lower_better else series.max()
            if best_val is None or (lower_better and v < best_val) or (not lower_better and v > best_val):
                best_val = v
                best_name = name
        if best_name:
            win_count[best_name] += 1
            print(f"  {label:<25} → {best_name}  ({best_val:.6f})")

    print(f"\n{'─' * 50}")
    print("  Win counts:")
    for name, cnt in sorted(win_count.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        print(f"    {name:<22} {cnt:>2}  {bar}")
    print("=" * 80)


def main():
    dfs = load_logs()
    if not dfs:
        print("No log files found.")
        return

    save_path = os.path.join(OUTPUT_ROOT, "..", "tools", "experiment_comparison.png")
    save_path = os.path.normpath(save_path)

    print_summary(dfs)
    plot_curves(dfs, save_path)


if __name__ == "__main__":
    main()

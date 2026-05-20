"""
Summarize best validation metrics for every experiment under output/.
Auto-discovers all subdirectories that contain a log.txt, then prints a
ranked table per metric (same style as compare_experiments_output.txt) and
saves the result to output/all_experiments_summary.txt.
"""

import os
import sys
import pandas as pd

OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "output")

# Human-readable names for known directories; anything not listed falls back
# to the raw directory name.
NAME_MAP = {
    "deeplabv3_outputs":            "DeepLabV3",
    "deeplabv3plus_outputs":        "DeepLabV3+",
    "fusnet_baseline_add_outputs":  "FusNet-Base(Add)",
    "fusnet_baseline_cat_outputs":  "FusNet-Base(Cat)",
    "fusnet_legacy_1_outputs":      "FusNet-Legacy-v1",
    "fusnet_legacy_2_outputs":      "FusNet-Legacy-v2",
    "fusnet_legacy_3_outputs":      "FusNet-Legacy-v3",
    "fusnet_legacy_4_outputs":      "FusNet-Legacy-v4",
    "fusnet_legacy_5_outputs":      "FusNet-Legacy-v5",
    "fusnet_outputs_res_mamba":     "Res+Mamba",
    "fusnet_outputs_res_swin":      "Res+Swin",
    "fusnet_outputs_res_swin_mamba":"Res+Swin+Mamba",
    "fusnet_outputs_swin_mamba":    "Swin+Mamba",
    "mambavision_outputs":          "MambaVision",
    "res2net_outputs":              "Res2Net",
    "swin_outputs":                 "Swin",
    "swinunet_outputs":             "SwinUNet",
}

VAL_METRICS = [
    ("Val_Loss",        "Val Loss",         True),   # (col, label, lower_is_better)
    ("Val_Acc",         "Val Accuracy",     False),
    ("Val_mIoU",        "Val mIoU",         False),
    ("Val_IoU_Bamboo",  "Val IoU (Bamboo)", False),
    ("Val_F1",          "Val F1",           False),
    ("Val_Recall",      "Val Recall",       False),
]


def discover_experiments():
    """Return OrderedDict: display_name -> log_path, sorted by display name."""
    experiments = {}
    for entry in sorted(os.listdir(OUTPUT_ROOT)):
        folder = os.path.join(OUTPUT_ROOT, entry)
        if not os.path.isdir(folder):
            continue
        log_path = os.path.join(folder, "log.txt")
        if not os.path.exists(log_path):
            continue
        name = NAME_MAP.get(entry, entry)
        experiments[name] = log_path
    return experiments


def load_logs(experiments):
    dfs = {}
    for name, path in experiments.items():
        try:
            df = pd.read_csv(path)
            # Drop any duplicate header rows emitted on resume
            df = df[df["Epoch"] != "Epoch"].reset_index(drop=True)
            df = df.apply(pd.to_numeric, errors="coerce")
            if "Epoch" not in df.columns:
                print(f"[WARN] No Epoch column in {path}, skipping.")
                continue
            dfs[name] = df
        except Exception as exc:
            print(f"[WARN] Cannot load {path}: {exc}")
    return dfs


def build_summary(dfs):
    lines = []
    W = 80

    lines.append("=" * W)
    lines.append("EXPERIMENT METRICS SUMMARY (Validation)")
    lines.append("=" * W)

    win_count = {n: 0 for n in dfs}
    winners = {}  # metric label -> (name, val)

    for col, label, lower_better in VAL_METRICS:
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  {label}  ({'lower is better' if lower_better else 'higher is better'})")
        lines.append(f"{'─' * 60}")
        lines.append(f"  {'Experiment':<26} {'Best Value':>12}  {'Epoch':>6}")
        lines.append(f"  {'─'*26} {'─'*12}  {'─'*6}")

        results = []
        for name, df in dfs.items():
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

        if not results:
            lines.append("  (no data)")
            continue

        results.sort(key=lambda x: x[1], reverse=(not lower_better))

        for rank, (name, val, epoch) in enumerate(results):
            marker = " <<<  BEST" if rank == 0 else ""
            lines.append(f"  {name:<26} {val:>12.6f}  {epoch:>6}{marker}")

        best_name, best_val, _ = results[0]
        win_count[best_name] += 1
        winners[label] = (best_name, best_val)

    lines.append(f"\n{'=' * W}")
    lines.append("OVERALL WINNER PER METRIC")
    lines.append("=" * W)
    for _, label, _ in VAL_METRICS:
        if label in winners:
            name, val = winners[label]
            lines.append(f"  {label:<25} → {name}  ({val:.6f})")

    lines.append(f"\n{'─' * 50}")
    lines.append("  Win counts:")
    for name, cnt in sorted(win_count.items(), key=lambda x: -x[1]):
        bar = "█" * cnt
        lines.append(f"    {name:<26} {cnt:>2}  {bar}")
    lines.append("=" * W)

    return "\n".join(lines)


def main():
    experiments = discover_experiments()
    if not experiments:
        print("No experiment log files found under output/.")
        sys.exit(1)

    print(f"Found {len(experiments)} experiments: {', '.join(experiments)}\n")

    dfs = load_logs(experiments)
    summary = build_summary(dfs)

    print(summary)

    save_path = os.path.join(OUTPUT_ROOT, "all_experiments_summary.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nSaved to: {os.path.normpath(save_path)}")


if __name__ == "__main__":
    main()

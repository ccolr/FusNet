"""Extract validation metrics from a training log file."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


def extract_val_metrics(log_path: str, output_path: Optional[str] = None) -> None:
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Error: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    val_columns = ["Epoch", "Val_Loss", "Val_Acc", "Val_mIoU", "Val_IoU_Bamboo", "Val_F1", "Val_Recall"]

    rows = []
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in val_columns if c not in reader.fieldnames]
        if missing:
            print(f"Error: columns not found in log: {missing}", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            rows.append({c: row[c] for c in val_columns})

    out = open(output_path, "w", newline="") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=val_columns)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output_path:
            out.close()

    if output_path:
        print(f"Saved {len(rows)} epochs → {output_path}")
    else:
        # also print best epoch summary to stderr so it doesn't pollute piped CSV
        valid_rows = [r for r in rows if r["Val_mIoU"] is not None]
        best = max(valid_rows, key=lambda r: float(r["Val_mIoU"]))
        print(
            f"\nBest epoch by Val_mIoU: {best['Epoch']} "
            f"(mIoU={best['Val_mIoU']}, F1={best['Val_F1']}, Acc={best['Val_Acc']})",
            file=sys.stderr,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract validation metrics from log.txt")
    parser.add_argument("log", help="Path to log.txt")
    parser.add_argument("-o", "--output", help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()
    extract_val_metrics(args.log, args.output)

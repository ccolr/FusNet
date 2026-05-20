"""
Compare heatmaps from three single-backbone models side by side,
with the original image on the far left.

Output layout per image (1 row × 4 columns):
  Original  |  Res2Net  |  Swin-T  |  MambaVision

Usage:
    python tools/compare_heatmaps.py [--output_dir visualization/heatmap_compare]
                                     [--image_dir bamboo/images]
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# ── Typography ────────────────────────────────────────────────────────────────
FONT = "LXGW WenKai"
rcParams["font.family"] = FONT
rcParams["axes.unicode_minus"] = False

# ── Layout constants (inches) ─────────────────────────────────────────────────
IMG_W    = 3.5
IMG_H    = 3.5
GAP      = 0.35
MARGIN_L = 0.30
MARGIN_T = 0.60
MARGIN_B = 0.30

N_PANELS = 4
FIG_W = MARGIN_L * 2 + N_PANELS * IMG_W + (N_PANELS - 1) * GAP
FIG_H = MARGIN_T + IMG_H + MARGIN_B

def _panel_x(col):
    return (MARGIN_L + col * (IMG_W + GAP)) / FIG_W

PANEL_Y = MARGIN_B / FIG_H
PANEL_W = IMG_W / FIG_W
PANEL_H = IMG_H / FIG_H

# ── Style ─────────────────────────────────────────────────────────────────────
LABEL_FONTSIZE = 11
TITLE_FONTSIZE = 9
LABEL_COLOR    = "#111111"
TITLE_COLOR    = "#555555"
SPINE_COLOR    = "#cccccc"
SPINE_LW       = 0.6

HEATMAP_DIRS = {
    "Res2Net":     Path("visualization/res2net_seg/heatmaps"),
    "Swin-T":      Path("visualization/swin_seg/heatmaps"),
    "MambaVision": Path("visualization/mambavision_seg/heatmaps"),
}


def _read_tif_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def _heatmap_stem(fname: str) -> str:
    """bamboo_images_image1_..._heatmap.png  →  image1_..."""
    return fname.replace("bamboo_images_", "").replace("_heatmap.png", "")


def _make_figure(fname: str, image_dir: Path) -> plt.Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
    fig.patch.set_facecolor("white")

    stem = _heatmap_stem(fname)
    panels = [("原图", image_dir / f"{stem}.tif", True)] + [
        (label, d / fname, False)
        for label, d in HEATMAP_DIRS.items()
    ]

    for col, (label, path, is_tif) in enumerate(panels):
        ax = fig.add_axes([_panel_x(col), PANEL_Y, PANEL_W, PANEL_H])

        if is_tif:
            img = _read_tif_rgb(path)
        else:
            img = mpimg.imread(path)

        ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
            spine.set_linewidth(SPINE_LW)

        fig.text(
            _panel_x(col) + PANEL_W / 2,
            PANEL_Y - 0.025,
            label,
            ha="center", va="top",
            fontsize=LABEL_FONTSIZE,
            fontfamily=FONT,
            color=LABEL_COLOR,
        )

    title = stem
    fig.text(
        0.5, 1 - (MARGIN_T * 0.28) / FIG_H,
        title,
        ha="center", va="top",
        fontsize=TITLE_FONTSIZE,
        fontfamily=FONT,
        color=TITLE_COLOR,
    )

    return fig


def main(output_dir: Path, image_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    name_sets = [set(p.name for p in d.iterdir() if p.suffix == ".png")
                 for d in HEATMAP_DIRS.values()]
    common = sorted(name_sets[0].intersection(*name_sets[1:]))

    if not common:
        print("No common heatmap filenames found across the three folders.")
        return

    missing_orig = 0
    for fname in tqdm(common, desc="heatmap_compare", unit="img"):
        stem = _heatmap_stem(fname)
        orig_path = image_dir / f"{stem}.tif"
        if not orig_path.exists():
            tqdm.write(f"  [skip] original not found: {orig_path}")
            missing_orig += 1
            continue

        fig = _make_figure(fname, image_dir)
        out_path = output_dir / fname.replace("_heatmap.png", "_compare.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)

    saved = len(common) - missing_orig
    print(f"\nDone. {saved}/{len(common)} comparison images written to {output_dir}/")
    if missing_orig:
        print(f"  ({missing_orig} skipped — original TIF not found)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="visualization/heatmap_compare")
    parser.add_argument("--image_dir",  default="bamboo/images")
    args = parser.parse_args()
    main(Path(args.output_dir), Path(args.image_dir))

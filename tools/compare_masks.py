"""
Generate four groups of mask comparison figures.

Group 1 — single backbones (1 output folder):
  原图 | GT | Res2Net | Swin-T | MambaVision

Group 2 — each two-branch fusion vs its component singles (3 output folders):
  2a  原图 | GT | Res2Net | Swin-T | Res2Net+Swin-T
  2b  原图 | GT | Res2Net | MambaVision | Res2Net+MambaVision
  2c  原图 | GT | Swin-T | MambaVision | Swin-T+MambaVision

Group 3 — three two-branch fusions (1 output folder):
  原图 | GT | Res2Net+Swin-T | Res2Net+MambaVision | Swin-T+MambaVision

Group 4 — FusNet (3-branch) vs three two-branch fusions (1 output folder):
  原图 | GT | Res2Net+Swin-T | Res2Net+MambaVision | Swin-T+MambaVision | FusNet (3路)

Usage:
    python tools/compare_masks.py [--vis_root visualization] \
                                  [--image_dir bamboo/images] \
                                  [--gt_dir bamboo/labels]
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

# ── Style constants ───────────────────────────────────────────────────────────
IMG_W          = 3.0
IMG_H          = 3.0
GAP            = 0.30
MARGIN_L       = 0.28
MARGIN_T       = 0.65
MARGIN_B       = 0.32
LABEL_FONTSIZE = 24
TITLE_FONTSIZE = 20
LABEL_COLOR    = "#111111"
TITLE_COLOR    = "#555555"
SPINE_COLOR    = "#cccccc"
SPINE_LW       = 0.6

# ── Model directories (relative to vis_root) ──────────────────────────────────
RES2NET      = Path("res2net_seg")
SWIN         = Path("swin_seg")
MAMBA        = Path("mambavision_seg")
RES_SWIN     = Path("fusnet_res_swin")
RES_MAMBA    = Path("fusnet_res_mamba")
SWIN_MAMBA   = Path("fusnet_swin_mamba")
FUSNET       = Path("fusnet_res_swin_mamba")
BASELINE_ADD = Path("fusnet_baseline_add")
BASELINE_CAT = Path("fusnet_baseline_cat")

# ── Four comparison groups ────────────────────────────────────────────────────
GROUPS = [
    # (group_id, output_subdir, panels_list)
    # panels_list: list of (label, model_rel_dir | None)
    #   None → this slot is GT (handled separately)
    (
        "1_single_backbones",
        "compare_single_backbones",
        [
            ("Res2Net",    RES2NET),
            ("Swin-T",     SWIN),
            ("MambaVision",MAMBA),
        ],
    ),
    (
        "2a_res_swin_vs_singles",
        "compare_res_swin_vs_singles",
        [
            ("Res2Net",        RES2NET),
            ("Swin-T",         SWIN),
            ("Res2Net+Swin-T", RES_SWIN),
        ],
    ),
    (
        "2b_res_mamba_vs_singles",
        "compare_res_mamba_vs_singles",
        [
            ("Res2Net",           RES2NET),
            ("MambaVision",       MAMBA),
            ("Res2Net+MambaVision",RES_MAMBA),
        ],
    ),
    (
        "2c_swin_mamba_vs_singles",
        "compare_swin_mamba_vs_singles",
        [
            ("Swin-T",              SWIN),
            ("MambaVision",         MAMBA),
            ("Swin-T+MambaVision",  SWIN_MAMBA),
        ],
    ),
    (
        "3_two_branch",
        "compare_2branch_masks",
        [
            ("Res2Net+Swin-T",       RES_SWIN),
            ("Res2Net+MambaVision",  RES_MAMBA),
            ("Swin-T+MambaVision",   SWIN_MAMBA),
        ],
    ),
    (
        "4_fusnet_vs_2branch",
        "compare_fusnet_vs_2branch_masks",
        [
            ("Res2Net+Swin-T",       RES_SWIN),
            ("Res2Net+MambaVision",  RES_MAMBA),
            ("Swin-T+MambaVision",   SWIN_MAMBA),
            ("FusNet (3路)",          FUSNET),
        ],
    ),
    (
        "5_fusnet_variants",
        "compare_fusnet_variants",
        [
            ("FusNet (3路)",  FUSNET),
            ("FusNet-Add",   BASELINE_ADD),
            ("FusNet-Cat",   BASELINE_CAT),
        ],
    ),
    (
        "6_fusnet_vs_methods",
        "compare_fusnet_vs_methods",
        [
            ("FusNet (3路)", FUSNET),
            ("DeepLabV3",   Path("deeplabv3")),
            ("DeepLabV3+",  Path("deeplabv3plus")),
            ("Swin-UNet",   Path("fusnet_swin_unet")),
        ],
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mask_stem(fname: str) -> str:
    """bamboo_images_image1_..._pred_mask.png  →  image1_..."""
    return fname.replace("bamboo_images_", "").replace("_pred_mask.png", "")


def _read_tif_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def _read_gt_mask(path: Path) -> np.ndarray:
    """Read a binary label TIF and return an H×W uint8 array (0/255)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.uint8)
    arr = (arr > 0).astype(np.uint8) * 255
    return arr


def _build_fig(stem: str, panels: list, image_dir: Path, gt_dir: Path) -> plt.Figure:
    """
    panels: list of (label, img_array)
    Full row: 原图 | GT | *panels
    """
    n_cols = 2 + len(panels)
    fig_w = MARGIN_L * 2 + n_cols * IMG_W + (n_cols - 1) * GAP
    fig_h = MARGIN_T + IMG_H + MARGIN_B

    def panel_x(col: int) -> float:
        return (MARGIN_L + col * (IMG_W + GAP)) / fig_w

    panel_y = MARGIN_B / fig_h
    panel_w = IMG_W / fig_w
    panel_h = IMG_H / fig_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")

    orig_path = image_dir / f"{stem}.tif"
    gt_path   = gt_dir   / f"{stem}_mask.tif"

    all_panels = [
        ("原图", _read_tif_rgb(orig_path)),
        ("GT",   _read_gt_mask(gt_path)),
    ] + panels

    for col, (label, img) in enumerate(all_panels):
        ax = fig.add_axes([panel_x(col), panel_y, panel_w, panel_h])
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", interpolation="lanczos")
        else:
            ax.imshow(img, interpolation="lanczos")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
            spine.set_linewidth(SPINE_LW)

        fig.text(
            panel_x(col) + panel_w / 2,
            panel_y - 0.025,
            label,
            ha="center", va="top",
            fontsize=LABEL_FONTSIZE,
            fontfamily=FONT,
            color=LABEL_COLOR,
        )

    fig.text(
        0.5, 1 - (MARGIN_T * 0.28) / fig_h,
        stem,
        ha="center", va="top",
        fontsize=TITLE_FONTSIZE,
        fontfamily=FONT,
        color=TITLE_COLOR,
    )

    return fig


# ── Per-group runner ──────────────────────────────────────────────────────────

def _run_group(group_id: str, out_subdir: str, model_panels: list,
               vis_root: Path, image_dir: Path, gt_dir: Path, output_root: Path):
    out_dir = output_root / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_dirs = [(label, vis_root / rel / "masks") for label, rel in model_panels]

    name_sets = [
        set(p.name for p in d.iterdir() if p.suffix == ".png")
        for _, d in mask_dirs
    ]
    common = sorted(name_sets[0].intersection(*name_sets[1:]))

    if not common:
        print(f"[{group_id}] No common mask files found — skipping.")
        return

    saved = 0
    for fname in tqdm(common, desc=group_id, unit="img"):
        stem = _mask_stem(fname)
        orig_path = image_dir / f"{stem}.tif"
        gt_path   = gt_dir   / f"{stem}_mask.tif"

        if not orig_path.exists():
            tqdm.write(f"  [skip] original not found: {orig_path}")
            continue
        if not gt_path.exists():
            tqdm.write(f"  [skip] GT not found: {gt_path}")
            continue

        panels = []
        ok = True
        for label, mask_dir in mask_dirs:
            p = mask_dir / fname
            if not p.exists():
                tqdm.write(f"  [skip] mask not found: {p}")
                ok = False
                break
            panels.append((label, mpimg.imread(p)))
        if not ok:
            continue

        fig = _build_fig(stem, panels, image_dir, gt_dir)
        out_path = out_dir / fname.replace("_pred_mask.png", "_compare.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        saved += 1

    print(f"[{group_id}] {saved}/{len(common)} saved → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_GROUP_IDS = [g[0] for g in GROUPS]


def main(vis_root: Path, image_dir: Path, gt_dir: Path, output_root: Path,
         selected: list):
    for group_id, out_subdir, model_panels in GROUPS:
        if group_id not in selected:
            continue
        _run_group(group_id, out_subdir, model_panels,
                   vis_root, image_dir, gt_dir, output_root)
    print("\nAll selected groups done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mask comparison figures.")
    parser.add_argument("--vis_root",    default="visualization",
                        help="Root of per-model visualization dirs")
    parser.add_argument("--image_dir",   default="bamboo/images",
                        help="Original TIF images directory")
    parser.add_argument("--gt_dir",      default="bamboo/labels",
                        help="Ground-truth label TIF directory")
    parser.add_argument("--output_root", default="visualization/Masks_Compare",
                        help="Root directory for all output subfolders")
    parser.add_argument("--groups", nargs="+", default=ALL_GROUP_IDS,
                        metavar="GROUP_ID",
                        help=(
                            "Which groups to run. Default: all. "
                            f"Choices: {ALL_GROUP_IDS}"
                        ))
    args = parser.parse_args()

    invalid = set(args.groups) - set(ALL_GROUP_IDS)
    if invalid:
        parser.error(f"Unknown group(s): {invalid}. Valid: {ALL_GROUP_IDS}")

    main(
        Path(args.vis_root),
        Path(args.image_dir),
        Path(args.gt_dir),
        Path(args.output_root),
        selected=args.groups,
    )

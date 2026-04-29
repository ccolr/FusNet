"""
compare.py
==========
为每张图片生成「原图 | GT mask 叠加 | 预测 mask 叠加」三联对比图（白底 PNG）。

用法示例：
    python compare.py \
        --image_dir   data/images \
        --gt_dir      data/labels \
        --pred_dir    predictions/masks \
        --output_dir  comparisons \
        --gt_suffix   _mask.tif \
        --pred_suffix _pred_mask.png
"""

import argparse
import os
import glob

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont


# ── 叠加参数 ────────────────────────────────────────────────────────────────
OVERLAY_COLOR = (220, 30, 30)   # 红色叠加
OVERLAY_ALPHA = 0.45             # 叠加透明度
GAP           = 20               # 三图之间间隔（像素）
PADDING       = 30               # 整张图四周留白
LABEL_HEIGHT  = 32               # 标签文字区高度
DISPLAY_SIZE  = (256, 256)       # 每个子图统一显示尺寸
BG_COLOR      = (255, 255, 255)  # 白底


def read_rgb(path: str) -> np.ndarray:
    """返回 uint8 HWC RGB numpy array，自动处理多波段 tif。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            arr = src.read()            # (C, H, W)
        arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr
    else:
        img = Image.open(path).convert("RGB")
        return np.array(img, dtype=np.uint8)


def read_mask(path: str) -> np.ndarray:
    """返回二值 mask (H, W)，前景=True。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            m = src.read(1).astype(np.float32)
        return m > 0
    else:
        m = np.array(Image.open(path).convert("L"), dtype=np.float32)
        return m > 127


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA) -> np.ndarray:
    """将 mask 区域以指定颜色半透明叠加到图像上，返回 uint8 RGB。"""
    out = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.where(
            mask,
            out[:, :, c] * (1 - alpha) + color_arr[c] * alpha,
            out[:, :, c],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def to_pil(arr: np.ndarray, size=DISPLAY_SIZE) -> Image.Image:
    img = Image.fromarray(arr, mode="RGB")
    img = img.resize(size, Image.BILINEAR)
    return img


def make_label_strip(text: str, width: int, height: int = LABEL_HEIGHT) -> Image.Image:
    strip = Image.new("RGB", (width, height), BG_COLOR)
    draw  = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (height - th) // 2), text, fill=(40, 40, 40), font=font)
    return strip


def build_comparison(orig_img: np.ndarray,
                     gt_mask: np.ndarray,
                     pred_mask: np.ndarray,
                     heatmap: np.ndarray) -> Image.Image:
    """返回三联对比 PIL Image（白底）。"""
    gt_overlay   = overlay_mask(orig_img, gt_mask)
    pred_overlay = overlay_mask(orig_img, pred_mask)

    panels_data = [
        ("Original",          orig_img),
        ("GT Mask Overlay",   gt_overlay),
        ("Pred Mask Overlay", pred_overlay),
        ("Heatmap Overlay",   heatmap),
    ]

    W, H    = DISPLAY_SIZE
    n       = len(panels_data)
    total_w = PADDING * 2 + n * W + (n - 1) * GAP
    total_h = PADDING * 2 + H + LABEL_HEIGHT + 6

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)

    for i, (label, arr) in enumerate(panels_data):
        panel = to_pil(arr)
        x = PADDING + i * (W + GAP)
        canvas.paste(panel, (x, PADDING))
        label_img = make_label_strip(label, W)
        canvas.paste(label_img, (x, PADDING + H + 6))

    return canvas


def find_image(image_dir: str, stem: str) -> str | None:
    """在 image_dir 中找与 stem 最匹配的文件（忽略扩展名大小写）。"""
    for ext in ("tif", "tiff", "png", "jpg", "jpeg", "TIF", "TIFF", "PNG", "JPG"):
        candidate = os.path.join(image_dir, f"{stem}.{ext}")
        if os.path.exists(candidate):
            return candidate
    # glob fallback
    hits = glob.glob(os.path.join(image_dir, f"{stem}.*"))
    return hits[0] if hits else None


def main():
    parser = argparse.ArgumentParser(description="FusNet Compare")
    parser.add_argument("--image_dir",   type=str, required=True,
                        help="原始图像文件夹")
    parser.add_argument("--gt_dir",      type=str, required=True,
                        help="GT mask 文件夹")
    parser.add_argument("--pred_dir",    type=str, required=True,
                        help="预测 mask 文件夹（predict.py 的输出）")
    parser.add_argument("--heatmap_dir",    type=str, required=True,
                        help="热力图文件夹（heatmap.py 的输出）")
    parser.add_argument("--output_dir",  type=str, default="comparisons",
                        help="对比图输出文件夹")
    parser.add_argument("--gt_suffix",   type=str, default="_mask.tif",
                        help="GT mask 文件的后缀（含扩展名），如 _mask.tif")
    parser.add_argument("--pred_suffix", type=str, default="_pred_mask.png",
                        help="预测 mask 文件的后缀，如 _pred_mask.png")
    parser.add_argument("--heatmap_suffix", type=str, default="_heatmap.png",
                        help="热力图文件的后缀，如 _heatmap.png")
    parser.add_argument("--display_size", type=int, nargs=2, default=[256, 256],
                        metavar=("W", "H"), help="每个子图的显示尺寸（默认 256 256）")
    args = parser.parse_args()

    global DISPLAY_SIZE
    DISPLAY_SIZE = tuple(args.display_size)

    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历 pred_dir 中的所有预测 mask
    pred_files = sorted(
        f for f in os.listdir(args.pred_dir)
        if f.endswith(args.pred_suffix)
    )
    if not pred_files:
        print(f"[WARN] No predicted masks found in {args.pred_dir} with suffix '{args.pred_suffix}'")
        return

    print(f"Found {len(pred_files)} predicted masks.")

    ok, skip = 0, 0
    for pred_fname in pred_files:
        # 1. 定义需要去除的前缀
        prefix = "bamboo_images_"
        
        # 2. 从预测文件名推断 stem
        # 先去掉后缀，再去掉前缀
        # stem = pred_fname[: -len(args.pred_suffix)]
        # if stem.startswith(prefix):
        #     stem = stem[len(prefix):]
            
        # --- 或者在 Python 3.9+ 中可以使用更优雅的写法： ---
        stem = pred_fname.removesuffix(args.pred_suffix).removeprefix("bamboo_images_")

        # 3. 找原始图像
        img_path = find_image(args.image_dir, stem)
        if img_path is None:
            print(f"  [SKIP] Image not found for stem '{stem}' (original: {pred_fname})")
            skip += 1
            continue

        # 4. 找 GT mask
        gt_fname = stem + args.gt_suffix
        gt_path  = os.path.join(args.gt_dir, gt_fname)
        
        if not os.path.exists(gt_path):
            # fallback: 在 gt_dir 中做 glob
            hits = glob.glob(os.path.join(args.gt_dir, f"{stem}*"))
            gt_path = hits[0] if hits else None

        if gt_path is None or not os.path.exists(gt_path):
            print(f"  [SKIP] GT mask not found for stem '{stem}'")
            skip += 1
            continue

        pred_path = os.path.join(args.pred_dir, pred_fname)
        heatmap_path = os.path.join(args.heatmap_dir, f"bamboo_images_{stem}{args.heatmap_suffix}")

        try:
            orig_img  = read_rgb(img_path)
            gt_mask   = read_mask(gt_path)
            pred_mask = read_mask(pred_path)
            heatmap   = read_rgb(heatmap_path)

            comparison = build_comparison(orig_img, gt_mask, pred_mask, heatmap)
            out_path   = os.path.join(args.output_dir, f"{stem}_compare.png")
            comparison.save(out_path)
            print(f"  Saved: {out_path}")
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            skip += 1

    print(f"\nDone. {ok} comparisons saved to: {args.output_dir}  ({skip} skipped)")


if __name__ == "__main__":
    main()

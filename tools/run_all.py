"""
run_all.py
==========
一次性运行 predict → compare → heatmap 三个步骤。

用法示例（最简）：
    python run_all.py \
        --weights     fusnet_outputs/best_model.pth \
        --test_txt    test.txt \
        --data_dir    . \
        --image_dir   data/images \
        --gt_dir      data/labels \
        --output_root results

所有子目录会自动创建在 --output_root 下：
    results/
      masks/        ← predict.py 输出
      comparisons/  ← compare.py 输出
      heatmaps/     ← heatmap.py 输出

高级选项（每个步骤的参数都可以单独覆盖，不传则用默认值）：
    --pred_dir    results/masks          # 若你已有预测结果，可跳过 predict
    --gt_suffix   _mask.tif
    --pred_suffix _pred_mask.png
    --heatmap_suffix _heatmap.png
    --display_size 256 256
    --input_size  224
    --alpha       0.5
    --skip_predict  # 跳过 predict 步骤
    --skip_compare  # 跳过 compare 步骤
    --skip_heatmap  # 跳过 heatmap 步骤
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], step_name: str):
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"  CMD : {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[WARN] '{step_name}' exited with code {result.returncode}. "
              "Pipeline continues, but check the output above.")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="FusNet 一键推理 + 对比图 + 热力图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 公共参数 ─────────────────────────────────────────────────────────────
    parser.add_argument("--weights",     type=str, required=True,
                        help="模型权重路径，如 fusnet_outputs/best_model.pth")
    parser.add_argument("--test_txt",    type=str, required=True,
                        help="test.txt 路径，每行为相对 data_dir 的图片路径")
    parser.add_argument("--data_dir",    type=str, default=".",
                        help="项目根目录（predict / heatmap 的 data_dir）")
    parser.add_argument("--image_dir",   type=str, required=True,
                        help="原始图像文件夹（compare.py 用）")
    parser.add_argument("--gt_dir",      type=str, required=True,
                        help="GT mask 文件夹（compare.py 用）")
    parser.add_argument("--output_root", type=str, default="results",
                        help="所有输出的根目录")

    # ── 细粒度覆盖 ───────────────────────────────────────────────────────────
    parser.add_argument("--pred_dir",    type=str, default=None,
                        help="预测 mask 目录（默认 output_root/masks）；"
                             "设置此项并配合 --skip_predict 可跳过推理步骤")
    parser.add_argument("--gt_suffix",   type=str, default="_mask.tif",
                        help="GT mask 文件后缀")
    parser.add_argument("--pred_suffix", type=str, default="_pred_mask.png",
                        help="预测 mask 文件后缀")
    parser.add_argument("--heatmap_suffix", type=str, default="_heatmap.png",
                        help="热力图文件的后缀，如 _heatmap.png")
    parser.add_argument("--display_size", type=int, nargs=2, default=[256, 256],
                        metavar=("W", "H"), help="compare 子图显示尺寸")
    parser.add_argument("--input_size",  type=int, default=224,
                        help="模型输入分辨率")
    parser.add_argument("--alpha",       type=float, default=0.5,
                        help="热力图叠加透明度")

    # ── 跳过开关 ─────────────────────────────────────────────────────────────
    parser.add_argument("--skip_predict",  action="store_true",
                        help="跳过 predict 步骤")
    parser.add_argument("--skip_compare",  action="store_true",
                        help="跳过 compare 步骤")
    parser.add_argument("--skip_heatmap",  action="store_true",
                        help="跳过 heatmap 步骤")

    args = parser.parse_args()

    # ── 目录规划 ─────────────────────────────────────────────────────────────
    mask_dir   = args.pred_dir or os.path.join(args.output_root, "masks")
    comp_dir   = os.path.join(args.output_root, "comparisons")
    hmap_dir   = os.path.join(args.output_root, "heatmaps")

    os.makedirs(args.output_root, exist_ok=True)

    python = sys.executable  # 使用当前 Python 环境

    # ── 脚本路径（假设与 run_all.py 同目录）─────────────────────────────────
    here         = os.path.dirname(os.path.abspath(__file__))
    predict_py   = os.path.join(here, "predict.py")
    compare_py   = os.path.join(here, "compare.py")
    heatmap_py   = os.path.join(here, "heatmap.py")

    codes = {}

    # ── Step 1: Predict ──────────────────────────────────────────────────────
    if not args.skip_predict:
        cmd = [
            python, predict_py,
            "--weights",    args.weights,
            "--test_txt",   args.test_txt,
            "--data_dir",   args.data_dir,
            "--output_dir", mask_dir,
            "--input_size", str(args.input_size),
        ]
        codes["predict"] = run(cmd, "Predict masks")
    else:
        print("\n[SKIP] predict step")

    # ── Step 2: Compare ──────────────────────────────────────────────────────
    if not args.skip_heatmap:
        cmd = [
            python, heatmap_py,
            "--weights",    args.weights,
            "--test_txt",   args.test_txt,
            "--data_dir",   args.data_dir,
            "--output_dir", hmap_dir,
            "--input_size", str(args.input_size),
            "--alpha",      str(args.alpha),
        ]
        codes["heatmap"] = run(cmd, "Generate heatmaps")
    else:
        print("\n[SKIP] heatmap step")

    # ── Step 3: Heatmap ──────────────────────────────────────────────────────
    if not args.skip_compare:
        cmd = [
            python, compare_py,
            "--image_dir",   args.image_dir,
            "--gt_dir",      args.gt_dir,
            "--pred_dir",    mask_dir,
            "--heatmap_dir", hmap_dir,
            "--output_dir",  comp_dir,
            "--gt_suffix",   args.gt_suffix,
            "--pred_suffix", args.pred_suffix,
            "--heatmap_suffix", args.heatmap_suffix,
            "--display_size", str(args.display_size[0]), str(args.display_size[1]),
        ]
        codes["compare"] = run(cmd, "Generate comparisons")
    else:
        print("\n[SKIP] compare step")


    # ── 总结 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ALL STEPS COMPLETE")
    print(f"  Output root : {os.path.abspath(args.output_root)}")
    print(f"  Masks       : {mask_dir}")
    print(f"  Heatmaps    : {hmap_dir}")
    print(f"  Comparisons : {comp_dir}")
    if any(c != 0 for c in codes.values()):
        failed = [k for k, v in codes.items() if v != 0]
        print(f"\n  [WARN] These steps had non-zero exit codes: {failed}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

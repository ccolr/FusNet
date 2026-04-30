# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FusNet is a bamboo segmentation deep learning model for remote sensing imagery. It fuses three backbone architectures (Res2Net50, Swin Transformer Tiny, MambaVision Small) via a novel **Heterogeneous Cross-Attention (HCA)** mechanism to perform binary semantic segmentation (Bamboo vs. Non-Bamboo).

## Common Commands

**Training:**
```bash
python train.py --batch_size 8 --epochs 120 --data_dir .
```

**Web Demo (Streamlit):**
```bash
streamlit run app.py
```

**Batch Prediction:**
```bash
python tools/predict.py \
    --weights fusnet_outputs/best_model.pth \
    --test_txt test.txt \
    --data_dir . \
    --output_dir predictions/masks
```

**Full Analysis Pipeline:**
```bash
python tools/run_all.py \
    --weights fusnet_outputs/best_model.pth \
    --test_txt test.txt \
    --data_dir . \
    --image_dir bamboo/images \
    --gt_dir bamboo/labels \
    --output_root results
```

**GradCAM Heatmaps:**
```bash
python tools/heatmap.py --weights <path> --test_txt test.txt --data_dir . --output_dir heatmaps
```

**Visual Comparison (orig | GT | pred side-by-side):**
```bash
python tools/compare.py --image_dir bamboo/images --gt_dir bamboo/labels --pred_dir predictions/masks --output_dir comparisons
```

**Compute Dataset Statistics:**
```bash
python tools/compute_mean_std.py --data_dir bamboo/images
```

## Architecture

### Model (`model/`)

**Three Encoders** — all extract 4 stages (56×56 → 28×28 → 14×14 → 7×7):
- `res2net.py` — Res2Net50 CNN backbone with multi-scale residual blocks
- `swin.py` — Swin Transformer Tiny with shifted-window attention
- `mamba_vision.py` — MambaVision Small using state-space models

**Fusion** (`HCABlock.py`):
- `HCABlock` aligns each branch's channels to `dim_feat=256` via 1×1 conv, then performs **cross-attention** where each branch queries the other branches as key-value sources
- Supports ablation via `active_branches` parameter (e.g., `["res2net", "swin"]` to disable MambaVision)

**Decoder** (`FusNet.py`):
- FPN-style progressive upsampling from 7×7 → 224×224
- `DecodeBlock` modules with skip connections from each encoder stage
- Final 1×1 conv → 2-class output

`FusNet_origin.py` — alternative implementation using iAFF attention (AFFUtils.py) instead of the standard HCA; kept for reference.

### Data

Images are TIF/TIFF multi-band remote sensing files; labels are binary TIF masks. Image lists are in `train.txt`, `valid.txt`, `test.txt` (relative paths).

**Normalization:** MEAN=[56.777, 67.8952, 59.0113], STD=[37.9482, 34.2724, 30.4053] (per-channel for training dataset).

**Input resolution:** 224×224 (resized from originals during training; upsampled back to original resolution at inference).

### Training Details

- **Loss:** 0.5 × BCEWithLogits + 0.5 × Dice
- **Differential LR:** Pretrained backbones at 1e-5, new modules (HCA, decoder) at 1e-4
- **Scheduler:** Warmup + CosineAnnealing
- **Mixed precision (AMP):** Enabled on CUDA
- **Gradient clipping:** max_norm=5.0
- **Augmentation:** albumentations pipeline — RandomResizedCrop, flips, rotations, elastic distortions, color jitter, Gaussian noise/blur, CoarseDropout

Outputs (checkpoints, logs, curves, confusion matrix) go to `fusnet_outputs/` by default; ablation runs use `output/fusnet_outputs_<branch_combo>/`.

### Ablation Studies

To train with a subset of backbones, pass `active_branches` to `FusNet`:
```python
model = FusNet(num_classes=2, active_branches=["res2net", "swin"])  # disable MambaVision
```
Completed ablations are in `output/` (res_swin, res_mamba, swin_mamba, res_swin_mamba).

## Key Dependencies

- `torch`, `torchvision` — training and inference
- `albumentations` — augmentation pipeline
- `rasterio` — GeoTIFF I/O
- `einops` — tensor reshaping in attention blocks
- `mamba_ssm` — state-space model ops for MambaVision
- `streamlit` — web demo
- `opencv-python`, `PIL`, `numpy`, `matplotlib`

MambaVision requires pretrained weights at `model/pretrained/mambavision_small_1k.pth.tar`.

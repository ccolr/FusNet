"""
predict.py
==========
对 test.txt 中列出的图片运行 FusNet 推理，将预测 mask 保存为 PNG。

用法示例：
    python predict.py \
        --weights fusnet_outputs/best_model.pth \
        --test_txt test.txt \
        --data_dir . \
        --output_dir predictions/masks
"""

import argparse
import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MEAN = [55.7578 / 255, 67.4502 / 255, 58.6568 / 255]
STD  = [37.5201 / 255, 34.2345 / 255, 30.3007 / 255]


def load_model(weights_path: str, device: torch.device):
    from model.FusNet import FusNet
    model = FusNet(num_classes=2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def read_image(img_path: str) -> np.ndarray:
    """读取遥感图像，返回 uint8 HWC RGB numpy array。"""
    with rasterio.open(img_path) as src:
        image = src.read()          # (C, H, W)
    image = np.moveaxis(image, 0, -1).astype(np.uint8)   # (H, W, C)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def preprocess(image: np.ndarray, size: int = 224) -> torch.Tensor:
    """归一化并 resize 为模型输入 tensor (1, 3, H, W)。"""
    h, w = image.shape[:2]
    img_f = image.astype(np.float32) / 255.0           # [0,1]

    # resize
    pil = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    img_f = np.array(pil).astype(np.float32) / 255.0

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    tensor = tfm(img_f)                                 # (3, H, W)  float32
    return tensor.unsqueeze(0)                          # (1, 3, H, W)


@torch.no_grad()
def predict_mask(model, tensor: torch.Tensor, orig_h: int, orig_w: int,
                 device: torch.device) -> np.ndarray:
    """返回与原图尺寸相同的二值 mask (uint8, 0/255)。"""
    tensor = tensor.to(device)
    logits = model(tensor)                              # (1, 2, H, W)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # 还原到原始分辨率
    logits = F.interpolate(logits, size=(orig_h, orig_w),
                           mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[:, 1, :, :]   # 竹林类概率
    binary = (probs >= 0.5).squeeze(0).cpu().numpy().astype(np.uint8) * 255
    return binary


def main():
    parser = argparse.ArgumentParser(description="FusNet Predict")
    parser.add_argument("--weights",    type=str, required=True,
                        help="模型权重路径，如 fusnet_outputs/best_model.pth")
    parser.add_argument("--test_txt",   type=str, required=True,
                        help="test.txt 路径，每行为相对项目根目录的图片路径")
    parser.add_argument("--data_dir",   type=str, default=".",
                        help="项目根目录（默认当前目录）")
    parser.add_argument("--output_dir", type=str, default="predictions/masks",
                        help="mask PNG 输出目录")
    parser.add_argument("--input_size", type=int, default=224,
                        help="模型输入分辨率（默认 224）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.weights, device)
    print(f"Loaded weights: {args.weights}")

    with open(args.test_txt, "r", encoding="utf-8") as f:
        rel_paths = [l.strip() for l in f if l.strip()]

    print(f"Total images to predict: {len(rel_paths)}")

    for rel_path in rel_paths:
        img_path = os.path.join(args.data_dir, rel_path)
        if not os.path.exists(img_path):
            print(f"  [WARN] Not found: {img_path}, skip.")
            continue

        image = read_image(img_path)
        orig_h, orig_w = image.shape[:2]
        tensor = preprocess(image, args.input_size)
        mask = predict_mask(model, tensor, orig_h, orig_w, device)

        # 输出文件名：保留相对路径结构（把路径分隔符替换为 _）
        stem = os.path.splitext(rel_path.replace(os.sep, "_").replace("/", "_"))[0]
        out_path = os.path.join(args.output_dir, f"{stem}_pred_mask.png")
        Image.fromarray(mask, mode="L").save(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nDone. Masks saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

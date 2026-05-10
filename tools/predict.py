"""
predict.py
==========
对 test.txt 中列出的图片运行推理，将预测 mask 保存为 PNG。
支持 FusNet、DeepLabV3(+) 和 Swin-Unet 三种模型。

用法示例：
    python predict.py \
        --weights fusnet_outputs/best_model.pth \
        --test_txt test.txt \
        --data_dir . \
        --output_dir predictions/masks

    # DeepLabV3+
    python predict.py \
        --model deeplab \
        --arch  deeplabv3plus_resnet50 \
        --weights deeplab_outputs/best_model.pth \
        --test_txt test.txt --data_dir . --output_dir predictions/masks

    # Swin-Unet
    python predict.py \
        --model swin_unet \
        --weights swinunet_outputs/best_model.pth \
        --test_txt test.txt --data_dir . --output_dir predictions/masks
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


def _resolve_deeplab_arch(state: dict, user_arch: str) -> str:
    """根据 checkpoint 的 key 自动判断 V3 还是 V3+，backbone 沿用 user_arch 中的部分。"""
    # DeepLabHeadV3Plus 有独立的 project/aspp 属性；DeepLabHead 没有
    is_plus = any(k.startswith("model.classifier.project.") for k in state)
    if user_arch.startswith("deeplabv3plus_"):
        backbone = user_arch[len("deeplabv3plus_"):]
    elif user_arch.startswith("deeplabv3_"):
        backbone = user_arch[len("deeplabv3_"):]
    else:
        backbone = "resnet50"
    resolved = f"{'deeplabv3plus' if is_plus else 'deeplabv3'}_{backbone}"
    if resolved != user_arch:
        print(f"[INFO] DeepLab arch auto-corrected: {user_arch!r} → {resolved!r}")
    return resolved


def load_model(weights_path: str, device: torch.device,
               model_type: str = "fusnet",
               arch: str = "deeplabv3plus_resnet50"):
    state = torch.load(weights_path, map_location=device)
    if model_type == "fusnet":
        from model.FusNet import FusNet
        model = FusNet()
    elif model_type == "fusnet_legacy_1":
        from model.FusNet_legacy_1 import FusNet
        model = FusNet(outchannel=2)
    elif model_type == "fusnet_legacy_2":
        from model.FusNet_legacy_2 import FusNet
        model = FusNet()
    elif model_type == "fusnet_legacy_3":
        from model.FusNet_legacy_3 import FusNet
        model = FusNet()
    elif model_type == "fusnet_legacy_4":
        from model.FusNet_legacy_4 import FusNet
        model = FusNet()
    elif model_type == "deeplab":
        arch = _resolve_deeplab_arch(state, arch)
        from model.deeplabv3_seg import DeepLabV3Seg
        model = DeepLabV3Seg(arch=arch, pretrained_backbone=False)
    elif model_type == "swin_unet":
        from model.swin_unet_seg import SwinUnetSeg
        model = SwinUnetSeg()
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            "Choose from: fusnet, fusnet_legacy_1/2/3/4, deeplab, swin_unet"
        )
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
    parser = argparse.ArgumentParser(description="Segmentation Predict")
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
    parser.add_argument("--model",      type=str, default="fusnet",
                        choices=["fusnet", "fusnet_legacy_1", "fusnet_legacy_2",
                                 "fusnet_legacy_3", "fusnet_legacy_4",
                                 "deeplab", "swin_unet"],
                        help="模型类型（默认 fusnet）")
    parser.add_argument("--arch",       type=str, default="deeplabv3plus_resnet50",
                        help="DeepLab 架构变体，仅当 --model deeplab 时有效，"
                             "如 deeplabv3plus_resnet50 / deeplabv3_resnet101 等")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.weights, device, args.model, args.arch)
    print(f"Loaded weights: {args.weights}  (model={args.model})")

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

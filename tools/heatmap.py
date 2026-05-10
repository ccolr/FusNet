"""
heatmap.py
==========
对 test.txt 中的每张图片生成 GradCAM 热力图，叠加到原图上输出为 PNG。
支持 FusNet、DeepLabV3(+) 和 Swin-Unet 三种模型。

GradCAM 目标层：
  fusnet    — decoder 最后一个 DecodeBlock 的 conv（64-ch, 224×224）
  deeplab   — classifier head 最终分类卷积前的 ReLU 层（256-ch）
  swin_unet — 最终 Conv2d（swin_unet.output）的输入特征（96-ch, 224×224）

用法示例：
    python heatmap.py \
        --weights    fusnet_outputs/best_model.pth \
        --test_txt   test.txt \
        --data_dir   . \
        --output_dir heatmaps

    python heatmap.py \
        --model deeplab --arch deeplabv3plus_resnet50 \
        --weights deeplab_outputs/best_model.pth \
        --test_txt test.txt --data_dir . --output_dir heatmaps

    python heatmap.py \
        --model swin_unet \
        --weights swinunet_outputs/best_model.pth \
        --test_txt test.txt --data_dir . --output_dir heatmaps
"""

import argparse
import os

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

MEAN = [55.7578 / 255, 67.4502 / 255, 58.6568 / 255]
STD  = [37.5201 / 255, 34.2345 / 255, 30.3007 / 255]


# ── GradCAM hook 工具 ───────────────────────────────────────────────────────

class GradCAM:
    """
    对指定 layer 做 GradCAM。

    use_input=False（默认）：hook 捕获目标层的输出激活（标准 GradCAM）。
    use_input=True：hook 捕获目标层的输入激活（用于 Swin-Unet 的最终 Conv2d，
                   使得特征图为 96-ch 空间张量而非 2-ch logits）。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module,
                 use_input: bool = False):
        self.model = model
        self.use_input = use_input
        self.feature: torch.Tensor | None = None

        if use_input:
            self._fwd_hook = target_layer.register_forward_pre_hook(self._save_input)
        else:
            self._fwd_hook = target_layer.register_forward_hook(self._save_feature)

    def _save_input(self, module, input):
        self.feature = input[0]
        self.feature.retain_grad()

    def _save_feature(self, module, input, output):
        self.feature = output
        output.retain_grad()

    def remove(self):
        self._fwd_hook.remove()

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            tensor : (1, 3, H, W)  已归一化的输入
        Returns:
            cam    : (H, W) float32  归一化到 [0, 1] 的热力图（模型输入尺寸）
        """
        self.model.zero_grad()
        logits = self.model(tensor)          # (1, 2, H, W)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        # 用竹林类 logits 的全局均值作为 scalar score
        score = logits[:, 1, :, :].mean()
        score.backward()

        # GradCAM 公式：alpha = gap(gradient), cam = relu(sum_c(alpha_c * feature_c))
        gradient = self.feature.grad                                           # (1, C, H', W')
        alpha    = gradient.mean(dim=(2, 3), keepdim=True)                    # (1, C, 1, 1)
        cam      = (alpha * self.feature.detach()).sum(dim=1, keepdim=True)   # (1, 1, H', W')
        cam      = F.relu(cam)
        cam      = cam.squeeze().cpu().numpy()                     # (H', W')

        # 归一化
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)


def get_target_layer(model, model_type: str = "fusnet",
                     arch: str = "deeplabv3plus_resnet50"):
    """
    返回 (target_layer, use_input) 元组供 GradCAM 使用。

    fusnet             : decoder[-1].conv — (B, 64, 224, 224)
    fusnet_legacy_1    : output2 Sequential — (B, 64, 56, 56)，最终 output1 的上一级
    fusnet_legacy_2/3/4: decoder[-1].conv — (B, 64, 224, 224)，与 fusnet 相同
    deeplab            : classifier.classifier[-2] — ReLU，(B, 256, H', W')
    swin_unet          : swin_unet.output (Conv2d) 的输入 — (B, 96, 224, 224)
    """
    if model_type in ("fusnet", "fusnet_legacy_2", "fusnet_legacy_3", "fusnet_legacy_4"):
        return model.decoder[-1].conv, False
    elif model_type == "fusnet_legacy_1":
        # legacy_1 使用平铺式解码器，output2 是送入 output1 之前的最后一个 64-ch Sequential
        return model.output2, False
    elif model_type == "deeplab":
        # 对 DeepLabHead 和 DeepLabHeadV3Plus 都适用：
        #   Sequential 中 [-2] 是最后一个 ReLU，[-1] 是最终 1×1 Conv。
        return model.model.classifier.classifier[-2], False
    elif model_type == "swin_unet":
        # 捕获送入最终 Conv2d 之前的 96-ch 空间特征图
        return model.swin_unet.output, True
    else:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            "Choose from: fusnet, fusnet_legacy_1/2/3/4, deeplab, swin_unet"
        )


# ── 图像 I/O ────────────────────────────────────────────────────────────────

def read_image(img_path: str) -> np.ndarray:
    ext = os.path.splitext(img_path)[1].lower()
    if ext in (".tif", ".tiff"):
        with rasterio.open(img_path) as src:
            arr = src.read()
        arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr
    else:
        return np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)


def preprocess(image: np.ndarray, size: int = 224) -> torch.Tensor:
    pil = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    img_f = np.array(pil).astype(np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return tfm(img_f).unsqueeze(0)


def apply_heatmap(image: np.ndarray, cam: np.ndarray,
                  alpha: float = 0.5) -> np.ndarray:
    """
    将 CAM 热力图（jet colormap）叠加到原图上。
    image : (H, W, 3) uint8 RGB
    cam   : (H', W') float32 [0,1]
    """
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_bgr = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )                                               # BGR
    heatmap_rgb = heatmap_bgr[:, :, ::-1]          # → RGB
    blended = (
        image.astype(np.float32) * (1 - alpha)
        + heatmap_rgb.astype(np.float32) * alpha
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── 主程序 ───────────────────────────────────────────────────────────────────

def _resolve_deeplab_arch(state: dict, user_arch: str) -> str:
    """根据 checkpoint 的 key 自动判断 V3 还是 V3+，backbone 沿用 user_arch 中的部分。"""
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


def main():
    parser = argparse.ArgumentParser(description="GradCAM Heatmap")
    parser.add_argument("--weights",    type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--test_txt",   type=str, required=True,
                        help="test.txt 路径，每行为相对项目根目录的图片路径")
    parser.add_argument("--data_dir",   type=str, default=".",
                        help="项目根目录（默认当前目录）")
    parser.add_argument("--output_dir", type=str, default="heatmaps",
                        help="热力图输出目录")
    parser.add_argument("--input_size", type=int, default=224,
                        help="模型输入分辨率（默认 224）")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="热力图叠加透明度（默认 0.5）")
    parser.add_argument("--model",      type=str, default="fusnet",
                        choices=["fusnet", "fusnet_legacy_1", "fusnet_legacy_2",
                                 "fusnet_legacy_3", "fusnet_legacy_4",
                                 "deeplab", "swin_unet"],
                        help="模型类型（默认 fusnet）")
    parser.add_argument("--arch",       type=str, default="deeplabv3plus_resnet50",
                        help="DeepLab 架构变体，仅当 --model deeplab 时有效")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # GradCAM 需要梯度，不能套 torch.no_grad()，eval() 仍需保留以固定 BN/Dropout
    model = load_model(args.weights, device, args.model, args.arch)
    print(f"Loaded weights: {args.weights}  (model={args.model})")

    target_layer, use_input = get_target_layer(model, args.model, args.arch)
    gradcam = GradCAM(model, target_layer, use_input=use_input)

    with open(args.test_txt, "r", encoding="utf-8") as f:
        rel_paths = [l.strip() for l in f if l.strip()]

    print(f"Total images: {len(rel_paths)}")

    for rel_path in rel_paths:
        img_path = os.path.join(args.data_dir, rel_path)
        if not os.path.exists(img_path):
            print(f"  [WARN] Not found: {img_path}, skip.")
            continue

        image = read_image(img_path)
        tensor = preprocess(image, args.input_size).to(device)
        tensor.requires_grad_(True)

        try:
            cam = gradcam(tensor)
            result = apply_heatmap(image, cam, alpha=args.alpha)

            stem = os.path.splitext(rel_path.replace(os.sep, "_").replace("/", "_"))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_heatmap.png")
            Image.fromarray(result, mode="RGB").save(out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  [ERROR] {rel_path}: {e}")

    gradcam.remove()
    print(f"\nDone. Heatmaps saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

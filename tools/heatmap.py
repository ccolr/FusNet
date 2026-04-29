"""
heatmap.py
==========
对 test.txt 中的每张图片生成 GradCAM 热力图，叠加到原图上输出为 PNG。

热力图来源：对分割 logits 的「竹林类」通道求全局平均作为标量 score，
再对 decoder 最终卷积前的特征图 (64 通道, 224×224) 做 GradCAM，
最终双线性上采样到原图尺寸并叠加。

用法示例：
    python heatmap.py \
        --weights    fusnet_outputs/best_model.pth \
        --test_txt   test.txt \
        --data_dir   . \
        --output_dir heatmaps
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
    target_layer 须是 nn.Module（不含 in-place 操作的 leaf module 为佳）。
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.feature: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_feature)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_feature(self, module, input, output):
        self.feature = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

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
        alpha   = self.gradient.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (alpha * self.feature).sum(dim=1, keepdim=True) # (1, 1, H', W')
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()                      # (H', W')

        # 归一化
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)


def get_target_layer(model):
    """
    返回 decoder 最后一个 DecodeBlock 中的 conv Sequential 作为目标层。
    这一层输出 (B, 64, 224, 224)，空间信息最丰富。
    """
    return model.decoder[-1].conv


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

def main():
    parser = argparse.ArgumentParser(description="FusNet GradCAM Heatmap")
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from model.FusNet import FusNet
    model = FusNet(num_classes=2)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    # 注意：GradCAM 需要梯度，不能 model.eval() + torch.no_grad()
    model.eval()

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

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

"""
app.py  ──  FusNet Web Demo
============================
启动方式：
    streamlit run app.py

依赖：
    pip install streamlit torch torchvision rasterio opencv-python pillow numpy einops
"""

import io
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import requests, base64, numpy as np

# ─── 页面配置（必须第一行） ───────────────────────────────────────────────────
st.set_page_config(
    page_title="FusNet Demo",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 全局样式 ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .block-container { padding-top: 2rem; padding-bottom: 2rem; }

  .img-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 8px;
  }

  .sample-divider {
    border: none;
    border-top: 1px solid #e4e7ec;
    margin: 2rem 0 1.5rem;
  }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
MEAN = [55.7578 / 255, 67.4502 / 255, 58.6568 / 255]
STD  = [37.5201 / 255, 34.2345 / 255, 30.3007 / 255]

OVERLAY_COLOR = (220, 30, 30)

# ─────────────────────────────────────────────────────────────────────────────
# ❶  在这里填写各路模型权重路径
# ─────────────────────────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "Res2Net + Swin-T + MambaVision (Full)": {
        "active_branches": ["res2net", "swin", "mamba"],
        "weight_path": "output/fusnet_outputs_res_swin_mamba/last_model.pth",
    },
    "Res2Net + Swin-T": {
        "active_branches": ["res2net", "swin"],
        "weight_path": "output/fusnet_outputs_res_swin/last_model.pth",
    },
    "Res2Net + MambaVision": {
        "active_branches": ["res2net", "mamba"],
        "weight_path": "output/fusnet_outputs_res_mamba/last_model.pth",
    },
    "Swin-T + MambaVision": {
        "active_branches": ["swin", "mamba"],
        "weight_path": "output/fusnet_outputs_swin_mamba/last_model.pth",
    },
}

CATEGORY_DIRS = {
    "original":   "originals",
    "mask":       "masks",
    "pred_blend": "pred_overlays",
    "heatmap":    "heatmaps",
}

# ─── GradCAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model    = model
        self.feature  = None
        self.gradient = None
        self._fh = target_layer.register_forward_hook(self._save_feature)
        self._bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_feature(self, m, i, o):    self.feature  = o.detach()
    def _save_gradient(self, m, gi, go): self.gradient = go[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        score = logits[:, 1, :, :].mean()
        score.backward()

        alpha = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam   = (alpha * self.feature).sum(dim=1, keepdim=True)
        cam   = F.relu(cam).squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)
        return cam.astype(np.float32)


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def read_image_bytes(uploaded) -> np.ndarray:
    """从 UploadedFile 读取 uint8 RGB HWC numpy array。"""
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".tif", ".tiff")):
        import rasterio, tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with rasterio.open(tmp_path) as src:
            arr = src.read()
        os.unlink(tmp_path)
        arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr
    else:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img, dtype=np.uint8)


def preprocess(image: np.ndarray, size: int = 224) -> torch.Tensor:
    pil = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return tfm(arr).unsqueeze(0)


def overlay_mask(image: np.ndarray, mask_bool: np.ndarray,
                 alpha: float) -> np.ndarray:
    """将 bool mask 以红色半透明叠加到原图，alpha 控制叠加浓度。"""
    out   = image.astype(np.float32).copy()
    color = np.array(OVERLAY_COLOR, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.where(
            mask_bool,
            out[:, :, c] * (1 - alpha) + color[c] * alpha,
            out[:, :, c],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_heatmap(image: np.ndarray, cam: np.ndarray,
                  alpha: float) -> np.ndarray:
    """将 GradCAM cam 以 jet colormap 叠加到原图，alpha 控制热力图浓度。"""
    h, w  = image.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hmap  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    blend = image.astype(np.float32) * (1 - alpha) + hmap.astype(np.float32) * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def pil_png_bytes(arr: np.ndarray, mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


@st.cache_resource
def load_model(config_name: str):
    """加载并缓存模型（按 config_name 键区分，切换配置才重新加载）。"""
    from model.FusNet import FusNet
    cfg    = MODEL_CONFIGS[config_name]
    model  = FusNet(active_branches=cfg["active_branches"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state  = torch.load(cfg["weight_path"], map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


@st.cache_data
def run_inference(_model, _device, config_name: str,
                  image_key: str, image: np.ndarray):
    """
    运行推理，返回原始中间结果（不做图像合成）。

    缓存键：config_name + image_key
      - 同一张图 + 同一模型 → 直接返回缓存，不重跑推理
      - 切换模型或换图 → 重新推理
    滑块变化时不触发此函数，只触发下游的合成步骤。

    注意：_model / _device 加下划线前缀，告知 Streamlit 不对其做哈希
          （torch.nn.Module 不可哈希），由 config_name 保证模型唯一性。

    Returns:
        mask_bool : (H, W) bool      二值预测 mask
        cam       : (H, W) float32   GradCAM 激活图（归一化 0~1）
    """
    orig_h, orig_w = image.shape[:2]
    tensor = preprocess(image, 224).to(_device)

    # ── 预测 mask ────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = F.interpolate(logits, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
        probs     = torch.softmax(logits, dim=1)[:, 1, :, :]
        mask_bool = (probs >= 0.5).squeeze(0).cpu().numpy()   # bool (H, W)

    # ── GradCAM ──────────────────────────────────────────────
    target_layer = _model.decoder[-1].conv
    gradcam      = GradCAM(_model, target_layer)
    tensor_grad  = preprocess(image, 224).to(_device)
    tensor_grad.requires_grad_(True)
    cam = gradcam(tensor_grad)   # float32 (H', W')
    gradcam.remove()

    return mask_bool, cam


# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 FusNet")
    st.caption("Bamboo Segmentation Demo")
    st.divider()

    config_name = st.selectbox(
        "Network configuration",
        list(MODEL_CONFIGS.keys()),
        help="选择 backbone 组合，对应不同权重文件",
    )

    cfg       = MODEL_CONFIGS[config_name]
    weight_ok = os.path.exists(cfg["weight_path"])
    if weight_ok:
        st.success("Weight found", icon="✅")
    else:
        st.warning(f"Weight not found:\n`{cfg['weight_path']}`", icon="⚠️")

    st.divider()
    st.markdown("**Branches active**")
    for b in cfg["active_branches"]:
        st.markdown(f"- `{b}`")

    st.divider()
    # 滑块变化只触发合成，不触发推理（推理结果已由 st.cache_data 缓存）
    heatmap_alpha = st.slider(
        "Heatmap blend α", 0.1, 0.9, 0.5, 0.05,
        help="热力图与原图的混合比例，越大热力图越突出",
    )
    overlay_alpha = st.slider(
        "Mask overlay α", 0.1, 0.9, 0.45, 0.05,
        help="预测 mask 红色叠加层的不透明度",
    )

# ─── 主内容区 ─────────────────────────────────────────────────────────────────
st.title("FusNet · Bamboo Segmentation")
st.caption(
    "Upload one or multiple images (`.tif`, `.tiff`, `.png`, `.jpg`) "
    "to run inference and visualise results."
)

uploaded_files = st.file_uploader(
    "Upload images",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if not uploaded_files:
    st.info("👆 Upload at least one image to get started.", icon="ℹ️")
    st.stop()

if not weight_ok:
    st.error(
        f"Cannot run inference: weight file not found at `{cfg['weight_path']}`.\n\n"
        "Please update `MODEL_CONFIGS` in `app.py` with the correct path.",
        icon="🚨",
    )
    st.stop()

# ─── 加载模型 ─────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model, device = load_model(config_name)
st.caption(f"Running on **{device}** · model: _{config_name}_")

# ─── 推理阶段：结果由 st.cache_data 缓存，滑块变化不会重跑 ───────────────────
raw_results = []

progress = st.progress(0, text="Running inference…")
for i, uf in enumerate(uploaded_files):
    progress.progress(i / len(uploaded_files), text=f"Inferring {uf.name}…")
    image     = read_image_bytes(uf)
    image_key = f"{config_name}::{uf.name}::{uf.size}"
    mask_bool, cam = run_inference(model, device, config_name, image_key, image)
    raw_results.append({
        "name":      Path(uf.name).stem,
        "original":  image,
        "mask_bool": mask_bool,
        "cam":       cam,
    })
progress.empty()

st.success(f"Done — {len(raw_results)} image(s) processed.", icon="✅")

# ─── 合成阶段：每次 rerun（包括滑块变化）都用当前 α 值重新合成 ───────────────
# 合成计算量极小（纯 numpy），rerun 时几乎无延迟
composed = []
for r in raw_results:
    mask_u8    = (r["mask_bool"].astype(np.uint8)) * 255
    pred_blend = overlay_mask(r["original"], r["mask_bool"], alpha=overlay_alpha)
    heatmap    = apply_heatmap(r["original"], r["cam"],      alpha=heatmap_alpha)
    composed.append({
        "name":       r["name"],
        "original":   r["original"],
        "mask":       mask_u8,
        "pred_blend": pred_blend,
        "heatmap":    heatmap,
    })

# ─── 批量下载区 ───────────────────────────────────────────────────────────────
with st.expander("📦 Batch download", expanded=False):
    st.markdown(
        "Select which output types to include in the zip. "
        "Files will be organised into sub-folders by type. "
        "**Downloads reflect the current α settings.**"
    )

    col_cb = st.columns(4)
    dl_original   = col_cb[0].checkbox("Original",     value=True)
    dl_mask       = col_cb[1].checkbox("Mask",         value=True)
    dl_pred_blend = col_cb[2].checkbox("Pred overlay", value=True)
    dl_heatmap    = col_cb[3].checkbox("Heatmap",      value=True)

    selected_types = []
    if dl_original:   selected_types.append("original")
    if dl_mask:       selected_types.append("mask")
    if dl_pred_blend: selected_types.append("pred_blend")
    if dl_heatmap:    selected_types.append("heatmap")

    if selected_types:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for c in composed:
                stem = c["name"]
                if "original" in selected_types:
                    zf.writestr(
                        f"{CATEGORY_DIRS['original']}/{stem}_original.png",
                        pil_png_bytes(c["original"]),
                    )
                if "mask" in selected_types:
                    zf.writestr(
                        f"{CATEGORY_DIRS['mask']}/{stem}_mask.png",
                        pil_png_bytes(c["mask"], mode="L"),
                    )
                if "pred_blend" in selected_types:
                    zf.writestr(
                        f"{CATEGORY_DIRS['pred_blend']}/{stem}_pred_overlay.png",
                        pil_png_bytes(c["pred_blend"]),
                    )
                if "heatmap" in selected_types:
                    zf.writestr(
                        f"{CATEGORY_DIRS['heatmap']}/{stem}_heatmap.png",
                        pil_png_bytes(c["heatmap"]),
                    )
        zip_buf.seek(0)
        st.download_button(
            label="⬇️  Download selected as ZIP",
            data=zip_buf,
            file_name="fusnet_results.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.warning("Select at least one output type above.")

st.divider()

# ─── 逐图展示 ─────────────────────────────────────────────────────────────────
PANEL_LABELS = {
    "original":   "Original",
    "mask":       "Predicted Mask",
    "pred_blend": "Mask Overlay",
    "heatmap":    "GradCAM Heatmap",
}

for idx, c in enumerate(composed):
    if idx > 0:
        st.markdown('<hr class="sample-divider">', unsafe_allow_html=True)

    st.markdown(f"### {c['name']}")

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    for col, key in zip(
        [col1, col2, col3, col4],
        ["original", "mask", "pred_blend", "heatmap"],
    ):
        with col:
            arr  = c[key]
            mode = "L" if key == "mask" else "RGB"
            st.markdown(
                f'<div class="img-card-title">{PANEL_LABELS[key]}</div>',
                unsafe_allow_html=True,
            )
            st.image(arr, use_container_width=True)
            st.download_button(
                label="⬇ Download",
                data=pil_png_bytes(arr, mode=mode),
                file_name=f"{c['name']}_{key}.png",
                mime="image/png",
                key=f"dl_{idx}_{key}",
                use_container_width=True,
            )

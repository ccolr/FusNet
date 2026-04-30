"""
app.py  ──  FusNet Web Demo（远程推理客户端）
=============================================
启动方式：
    streamlit run app.py

本地需要先做 SSH 端口转发（将远端 server.py 映射到本地）：
    ssh -L 8000:localhost:8000 user@remote-host

依赖（本地无需 torch / cuda）：
    pip install streamlit requests pillow numpy opencv-python rasterio
"""

import base64
import io
import os
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

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
OVERLAY_COLOR = (220, 30, 30)

CATEGORY_DIRS = {
    "original":   "originals",
    "mask":       "masks",
    "pred_blend": "pred_overlays",
    "heatmap":    "heatmaps",
}

PANEL_LABELS = {
    "original":   "Original",
    "mask":       "Predicted Mask",
    "pred_blend": "Mask Overlay",
    "heatmap":    "GradCAM Heatmap",
}


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def read_image_bytes(data: bytes, filename: str) -> np.ndarray:
    """从原始字节读取 uint8 RGB HWC numpy array（用于本地展示）。"""
    if filename.lower().endswith((".tif", ".tiff")):
        import rasterio, tempfile
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            with rasterio.open(tmp_path) as src:
                arr = src.read()
        finally:
            os.unlink(tmp_path)
        arr = np.moveaxis(arr, 0, -1).astype(np.uint8)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def overlay_mask(image: np.ndarray, mask_bool: np.ndarray, alpha: float) -> np.ndarray:
    out   = image.astype(np.float32).copy()
    color = np.array(OVERLAY_COLOR, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = np.where(
            mask_bool,
            out[:, :, c] * (1 - alpha) + color[c] * alpha,
            out[:, :, c],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    h, w  = image.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hmap  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    blend = image.astype(np.float32) * (1 - alpha) + hmap.astype(np.float32) * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def pil_png_bytes(arr: np.ndarray, mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


# ─── API 调用（结果由 st.cache_data 缓存，切换配置或换图才重新请求） ──────────
@st.cache_data(show_spinner=False)
def call_infer_api(
    server_url:  str,
    config_name: str,
    image_key:   str,   # 仅用作缓存键（filename::size），不发到服务器
    raw_bytes:   bytes,
    filename:    str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    向服务器发起推理请求，返回 (mask_bool, cam)。

    - 同一张图 + 同一配置 → 直接返回缓存，不重发请求
    - 切换模型或换图 → 重新请求
    - 滑块变化不触发此函数（合成在本地做）
    """
    resp = requests.post(
        f"{server_url}/infer",
        files={"file": (filename, raw_bytes)},
        data={"config_name": config_name},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()

    # 解码 mask（PNG grayscale → bool ndarray）
    mask_png  = base64.b64decode(data["mask"])
    mask_u8   = np.array(Image.open(io.BytesIO(mask_png)).convert("L"))
    mask_bool = mask_u8 > 127

    # 解码 cam（raw float32 bytes → ndarray）
    cam_bytes = base64.b64decode(data["cam"])
    cam_shape = tuple(data["cam_shape"])
    cam       = np.frombuffer(cam_bytes, dtype=np.float32).reshape(cam_shape)

    return mask_bool, cam


@st.cache_data(show_spinner=False, ttl=30)
def fetch_server_configs(server_url: str) -> Optional[dict]:
    """拉取服务器上各模型配置及权重状态。ttl=30s 自动刷新。"""
    try:
        resp = requests.get(f"{server_url}/configs", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def check_health(server_url: str) -> Optional[dict]:
    try:
        resp = requests.get(f"{server_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 FusNet")
    st.caption("Bamboo Segmentation Demo")
    st.divider()

    # 服务器地址
    server_url = st.text_input(
        "Inference server URL",
        value="http://localhost:8000",
        help="远端 server.py 的地址。通过 SSH 端口转发到本地后填 http://localhost:<port>",
    ).rstrip("/")

    # 连接状态
    health = check_health(server_url)
    if health:
        st.success(f"Server online · `{health.get('device', '?')}`", icon="✅")
    else:
        st.error("Server unreachable", icon="🔴")
        st.caption(f"无法连接 `{server_url}`，请检查 server.py 是否运行以及端口转发是否正确。")

    st.divider()

    # 拉取服务器配置
    server_configs = fetch_server_configs(server_url) or {}
    available_names = [
        name for name, info in server_configs.items()
        if info.get("weight_available", False)
    ]
    all_names = list(server_configs.keys())

    if not server_configs:
        st.warning("无法获取服务器配置", icon="⚠️")
        config_name = st.selectbox("Network configuration", [], disabled=True)
    else:
        display_names = all_names  # 全部显示，权重缺失的会警告
        config_name = st.selectbox(
            "Network configuration",
            display_names,
            help="选择 backbone 组合，对应服务器上不同权重文件",
        )

        cfg_info   = server_configs.get(config_name, {})
        weight_ok  = cfg_info.get("weight_available", False)
        if weight_ok:
            st.success("Weight available on server", icon="✅")
        else:
            st.warning("Weight not found on server", icon="⚠️")

        st.markdown("**Branches active**")
        for b in cfg_info.get("active_branches", []):
            st.markdown(f"- `{b}`")

    st.divider()

    # 可视化参数（滑块变化只触发本地合成，不触发服务器请求）
    heatmap_alpha = st.slider("Heatmap blend α", 0.1, 0.9, 0.5, 0.05,
                              help="热力图与原图的混合比例")
    overlay_alpha = st.slider("Mask overlay α", 0.1, 0.9, 0.45, 0.05,
                              help="预测 mask 红色叠加层的不透明度")

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

if not health:
    st.error("推理服务器不可达，请先启动 `server.py` 并确认端口转发。", icon="🚨")
    st.stop()

if not server_configs:
    st.error("无法从服务器获取配置，请检查服务器日志。", icon="🚨")
    st.stop()

if not weight_ok:
    st.error(
        f"服务器上找不到 `{config_name}` 的权重文件，请检查服务器路径配置。",
        icon="🚨",
    )
    st.stop()

# ─── 推理阶段：结果由 st.cache_data 缓存，滑块变化不触发请求 ─────────────────
raw_results = []

progress = st.progress(0, text="Sending to inference server…")
for i, uf in enumerate(uploaded_files):
    progress.progress(i / len(uploaded_files), text=f"Inferring {uf.name}…")

    raw_bytes = uf.read()
    image     = read_image_bytes(raw_bytes, uf.name)
    image_key = f"{config_name}::{uf.name}::{uf.size}"

    try:
        mask_bool, cam = call_infer_api(
            server_url, config_name, image_key, raw_bytes, uf.name
        )
    except requests.HTTPError as e:
        st.error(f"**{uf.name}** — 服务器返回错误: {e.response.text}", icon="🚨")
        continue
    except Exception as e:
        st.error(f"**{uf.name}** — 请求失败: {e}", icon="🚨")
        continue

    raw_results.append({
        "name":      Path(uf.name).stem,
        "original":  image,
        "mask_bool": mask_bool,
        "cam":       cam,
    })

progress.empty()

if not raw_results:
    st.stop()

st.success(f"Done — {len(raw_results)} image(s) processed.", icon="✅")

# ─── 合成阶段（纯本地 numpy，滑块变化即时响应） ───────────────────────────────
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

# ─── 批量下载 ─────────────────────────────────────────────────────────────────
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

    selected_types = [
        k for k, v in [
            ("original",   dl_original),
            ("mask",       dl_mask),
            ("pred_blend", dl_pred_blend),
            ("heatmap",    dl_heatmap),
        ] if v
    ]

    if selected_types:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for c in composed:
                stem = c["name"]
                if "original" in selected_types:
                    zf.writestr(f"{CATEGORY_DIRS['original']}/{stem}_original.png",
                                pil_png_bytes(c["original"]))
                if "mask" in selected_types:
                    zf.writestr(f"{CATEGORY_DIRS['mask']}/{stem}_mask.png",
                                pil_png_bytes(c["mask"], mode="L"))
                if "pred_blend" in selected_types:
                    zf.writestr(f"{CATEGORY_DIRS['pred_blend']}/{stem}_pred_overlay.png",
                                pil_png_bytes(c["pred_blend"]))
                if "heatmap" in selected_types:
                    zf.writestr(f"{CATEGORY_DIRS['heatmap']}/{stem}_heatmap.png",
                                pil_png_bytes(c["heatmap"]))
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
for idx, c in enumerate(composed):
    if idx > 0:
        st.markdown('<hr class="sample-divider">', unsafe_allow_html=True)

    st.markdown(f"### {c['name']}")

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    for col, key in zip([col1, col2, col3, col4],
                        ["original", "mask", "pred_blend", "heatmap"]):
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

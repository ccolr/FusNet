"""
app.py  ──  FusNet Web Demo v2（远程推理客户端）
================================================
模式：
  • Default  — 多模型预测，每张图 N 组 4-panel（原图 / 预测掩码 / 叠加 / GradCAM）
  • Research — 指定服务端 GT 文件夹，6-panel + 逐图指标表 + 汇总对比表与柱状图

启动：
    streamlit run app.py

本地需先做 SSH 端口转发：
    ssh -L 8000:localhost:8000 user@remote-host
"""

import base64
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    margin-bottom: 6px;
  }

  .model-header {
    font-size: 0.84rem;
    font-weight: 600;
    color: #1e3a5f;
    padding: 4px 0 3px;
    margin: 0.7rem 0 0.3rem;
    border-left: 3px solid #3b82f6;
    padding-left: 8px;
  }

  .sample-divider {
    border: none;
    border-top: 1px solid #e4e7ec;
    margin: 2.2rem 0 1.6rem;
  }

  /* ── paper-style table (booktabs, centered) ── */
  .paper-table-wrap { display: flex; justify-content: center; }
  .paper-table {
    border-collapse: collapse;
    width: auto;
    min-width: 55%;
    font-size: 0.88em;
    margin: 0.5em 0 1.2em;
  }
  .paper-table thead tr {
    border-top: 2.5px solid #111;
    border-bottom: 1.2px solid #555;
  }
  .paper-table thead th {
    padding: 7px 18px;
    text-align: center;
    font-weight: 600;
    background: transparent;
    letter-spacing: 0.04em;
  }
  .paper-table thead th:first-child { text-align: left; }
  .paper-table tbody td {
    padding: 5px 18px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }
  .paper-table tbody td:first-child { text-align: left; font-style: italic; }
  .paper-table tbody tr:last-child td { border-bottom: 2.5px solid #111; }
  .paper-table .best { font-weight: 800; }

  /* ── outer tab-list: center all tab buttons ── */
  [data-baseweb="tab-list"] {
    justify-content: center;
  }
  /* ── inner tab-lists: keep left-aligned ── */
  [data-baseweb="tab-panel"] [data-baseweb="tab-list"] {
    justify-content: flex-start;
  }
  /* ── outer tabs only: bold, slightly larger ── */
  [data-baseweb="tab"] {
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.03em;
  }
  /* ── inner tabs (inside a tab-panel): revert to defaults ── */
  [data-baseweb="tab-panel"] [data-baseweb="tab"] {
    font-size: revert;
    font-weight: revert;
    letter-spacing: revert;
  }

  footer { visibility: hidden; }

  /* ── image download button — mirrors Streamlit's fullscreen button style ── */
  .img-dl-float {
    height: 0;
    margin-top: -2.5rem;
    padding-right: 0.5rem;
    display: flex;
    justify-content: flex-end;
    position: relative;
    z-index: 100;
    pointer-events: none;
    overflow: visible;
  }
  .img-dl-btn {
    pointer-events: auto;
    flex-shrink: 0;
    width: 2rem;
    height: 2rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.8);
    border: 1px solid rgba(49, 51, 63, 0.1);
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
    color: #31333f;
    text-decoration: none;
    opacity: 0;
    transition: opacity 0.15s ease, background 0.1s;
    cursor: pointer;
  }
  .img-dl-btn:hover { opacity: 1 !important; background: rgba(255, 255, 255, 1); }
  /* reveal download button when hovering the adjacent stImage container */
  .element-container:has([data-testid="stImage"]):hover + .element-container .img-dl-btn {
    opacity: 1;
  }
</style>
""", unsafe_allow_html=True)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
COLOR_PRED  = (220, 30,  30)   # red overlay for predictions
COLOR_GT    = (220, 30,  30)   # red overlay for GT (same as prediction)
IMAGE_EXTS  = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
ALL_METRICS = ["Precision", "Recall", "F1", "IoU", "mIoU", "Accuracy"]

PANEL_NAMES_DEFAULT  = ["Original", "Pred_Mask", "Pred_Overlay", "GradCAM"]
PANEL_NAMES_RESEARCH = ["Original", "GT_Mask", "Pred_Mask", "GT_Overlay", "Pred_Overlay", "GradCAM"]

# ─── 磁盘缓存工具 ─────────────────────────────────────────────────────────────
_DISK_CACHE_DIR = Path(".fusnet_session")

def _fhash(name: str, size: int) -> str:
    return hashlib.md5(f"{name}:{size}".encode()).hexdigest()[:12]

def _chash(cfg: str) -> str:
    return hashlib.md5(cfg.encode()).hexdigest()[:8]

def _save_sidebar(state: dict):
    try:
        _DISK_CACHE_DIR.mkdir(exist_ok=True)
        (_DISK_CACHE_DIR / "sidebar.json").write_text(json.dumps(state))
    except Exception:
        pass

def _load_sidebar() -> dict:
    p = _DISK_CACHE_DIR / "sidebar.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}

def _save_stems_meta(stems: list):
    try:
        _DISK_CACHE_DIR.mkdir(exist_ok=True)
        (_DISK_CACHE_DIR / "stems.json").write_text(json.dumps(stems))
    except Exception:
        pass

def _load_stems_meta() -> list:
    p = _DISK_CACHE_DIR / "stems.json"
    try:
        return json.loads(p.read_text()) if p.exists() else []
    except Exception:
        return []

def _save_image_cache(name: str, size: int, arr: np.ndarray):
    d = _DISK_CACHE_DIR / "images"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{_fhash(name, size)}.npy", arr)

def _load_image_cache(name: str, size: int) -> Optional[np.ndarray]:
    p = _DISK_CACHE_DIR / "images" / f"{_fhash(name, size)}.npy"
    return np.load(p) if p.exists() else None

def _save_result_cache(name: str, size: int, cfg: str, probs: np.ndarray, cam: np.ndarray):
    d = _DISK_CACHE_DIR / "results"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{_fhash(name, size)}_{_chash(cfg)}.npz", probs=probs, cam=cam)

def _load_result_cache(name: str, size: int, cfg: str) -> Optional[dict]:
    p = _DISK_CACHE_DIR / "results" / f"{_fhash(name, size)}_{_chash(cfg)}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"probs": d["probs"], "cam": d["cam"]}

def _save_gt_cache(name: str, size: int, gt: Optional[np.ndarray]):
    if gt is None:
        return
    d = _DISK_CACHE_DIR / "gt"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{_fhash(name, size)}.npy", gt.astype(np.uint8))

def _load_gt_cache(name: str, size: int) -> Optional[np.ndarray]:
    p = _DISK_CACHE_DIR / "gt" / f"{_fhash(name, size)}.npy"
    return np.load(p).astype(bool) if p.exists() else None

def _clear_disk_cache():
    if _DISK_CACHE_DIR.exists():
        shutil.rmtree(_DISK_CACHE_DIR)


# ─── 图像工具 ─────────────────────────────────────────────────────────────────
def read_image_bytes(data: bytes, filename: str) -> np.ndarray:
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
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def pil_png_bytes(arr: np.ndarray, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    color: tuple = COLOR_PRED,
) -> np.ndarray:
    out = image.astype(np.float32).copy()
    c   = np.array(color, dtype=np.float32)
    for i in range(3):
        out[:, :, i] = np.where(
            mask,
            out[:, :, i] * (1 - alpha) + c[i] * alpha,
            out[:, :, i],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    h, w  = image.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hmap  = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    blend = image.astype(np.float32) * (1 - alpha) + hmap.astype(np.float32) * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def resize_mask_bool(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask
    return (
        np.array(
            Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(
                (w, h), Image.NEAREST
            )
        )
        > 127
    )


# ─── 指标计算 ─────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool).ravel(), gt.astype(bool).ravel()
    tp   = int(( p &  g).sum())
    fp   = int(( p & ~g).sum())
    fn   = int((~p &  g).sum())
    tn   = int((~p & ~g).sum())
    eps  = 1e-8
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fn + fp + eps)
    return {
        "Precision": tp / (tp + fp + eps),
        "Recall":    tp / (tp + fn + eps),
        "F1":        2 * tp / (2 * tp + fp + fn + eps),
        "IoU":       iou_fg,
        "mIoU":      (iou_fg + iou_bg) / 2,
        "Accuracy":  (tp + tn) / (tp + fp + fn + tn + eps),
    }


# ─── 论文级 HTML 表格（booktabs 风格，居中） ──────────────────────────────────
def paper_table_html(model_metrics: dict, metrics: list) -> str:
    if not model_metrics or not metrics:
        return ""
    best = {
        m: max(model_metrics, key=lambda mn: model_metrics[mn].get(m, 0))
        for m in metrics
    }
    header = "<thead><tr><th>Model</th>" + "".join(f"<th>{m}</th>" for m in metrics) + "</tr></thead>"
    rows   = ""
    for mn, mv in model_metrics.items():
        cells = ""
        for m in metrics:
            v   = mv.get(m, float("nan"))
            cls = ' class="best"' if mn == best[m] else ""
            cells += f"<td{cls}>{v:.4f}</td>"
        rows += f"<tr><td>{mn}</td>{cells}</tr>"
    return (
        f'<div class="paper-table-wrap">'
        f'<table class="paper-table">{header}<tbody>{rows}</tbody></table>'
        f'</div>'
    )


# ─── 论文级柱状图 ─────────────────────────────────────────────────────────────
def metrics_barchart_png(model_metrics: dict, metrics: list) -> bytes:
    models  = list(model_metrics.keys())
    n_m, nc = len(metrics), len(models)
    y       = np.arange(n_m)
    bw      = min(0.65 / max(nc, 1), 0.25)
    offsets = (np.arange(nc) - (nc - 1) / 2) * bw
    cmap    = plt.get_cmap("tab10").colors  # type: ignore

    fig_h = max(5.5, n_m * 0.8 + nc * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_vals = [model_metrics[mn].get(m, 0) for mn in models for m in metrics]
    xmax = min(1.15, max(all_vals, default=0) + 0.10)

    for i, (mn, off) in enumerate(zip(models, offsets)):
        vals = [model_metrics[mn].get(m, 0) for m in metrics]
        bars = ax.barh(y + off, vals, bw * 0.90,
                       label=mn, color=cmap[i % 10],
                       edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(
                min(b.get_width() + 0.005, xmax - 0.002),
                b.get_y() + b.get_height() / 2,
                f"{v:.3f}",
                ha="left", va="center", fontsize=8,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, xmax)

    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, color="#aaa")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_ncol = min(nc, 5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=legend_ncol,
        fontsize=9.5,
        framealpha=0.95,
        edgecolor="#ccc",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ─── API 调用 ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def call_infer_api(
    server_url:  str,
    config_name: str,
    image_key:   str,
    raw_bytes:   bytes,
    filename:    str,
) -> tuple[np.ndarray, np.ndarray]:
    resp = requests.post(
        f"{server_url}/infer",
        files={"file": (filename, raw_bytes)},
        data={"config_name": config_name},
        timeout=180,
    )
    resp.raise_for_status()
    d = resp.json()
    probs = np.frombuffer(
        base64.b64decode(d["probs"]), dtype=np.float32
    ).reshape(tuple(d["probs_shape"]))
    cam = np.frombuffer(
        base64.b64decode(d["cam"]), dtype=np.float32
    ).reshape(tuple(d["cam_shape"]))
    return probs, cam


@st.cache_data(show_spinner=False)
def call_gt_api(server_url: str, gt_dir: str, stem: str) -> Optional[np.ndarray]:
    """Request GT mask from server; returns bool ndarray or None if not found."""
    try:
        r = requests.get(
            f"{server_url}/gt",
            params={"gt_dir": gt_dir, "stem": stem},
            timeout=30,
        )
        if r.status_code == 200:
            d    = r.json()
            mask = np.frombuffer(
                base64.b64decode(d["mask"]), dtype=np.uint8
            ).reshape(tuple(d["shape"]))
            return mask.astype(bool)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=30)
def fetch_server_configs(server_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{server_url}/configs", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def check_health(server_url: str) -> Optional[dict]:
    try:
        r = requests.get(f"{server_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ─── ZIP 构建 ─────────────────────────────────────────────────────────────────
def build_results_zip(
    all_results:      dict,
    all_gt:           dict,
    selected_configs: list,
    conf_threshold:   float,
    overlay_alpha:    float,
    heatmap_alpha:    float,
    download_types:   list,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in all_results:
            raw_image = all_results[stem]["_raw"]
            h, w      = raw_image.shape[:2]
            gt_bool   = all_gt.get(stem)

            for cfg_name in selected_configs:
                if cfg_name not in all_results[stem]:
                    continue
                probs      = all_results[stem][cfg_name]["probs"]
                cam        = all_results[stem][cfg_name]["cam"]
                mask_bool  = resize_mask_bool(probs >= conf_threshold, h, w)
                mask_u8    = mask_bool.astype(np.uint8) * 255
                pred_blend = overlay_mask(raw_image, mask_bool, overlay_alpha, COLOR_PRED)
                heatmap    = apply_heatmap(raw_image, cam, heatmap_alpha)

                safe_cfg = cfg_name.replace("/", "_").replace(" ", "_")

                panels: dict[str, tuple[np.ndarray, str]] = {
                    "Original":     (raw_image,  "RGB"),
                    "Pred_Mask":    (mask_u8,    "L"),
                    "Pred_Overlay": (pred_blend, "RGB"),
                    "GradCAM":      (heatmap,    "RGB"),
                }
                if gt_bool is not None:
                    gt_r  = resize_mask_bool(gt_bool, h, w)
                    gt_u8 = gt_r.astype(np.uint8) * 255
                    gt_blend = overlay_mask(raw_image, gt_r, overlay_alpha, COLOR_GT)
                    panels["GT_Mask"]    = (gt_u8,    "L")
                    panels["GT_Overlay"] = (gt_blend, "RGB")

                for panel_name, (arr, mode_l) in panels.items():
                    if panel_name not in download_types:
                        continue
                    zf.writestr(
                        f"{safe_cfg}/{panel_name}/{stem}.png",
                        pil_png_bytes(arr, mode=mode_l),
                    )
    buf.seek(0)
    return buf.getvalue()


# ─── 从磁盘恢复 sidebar 状态（每个浏览器会话只初始化一次） ─────────────────────
if "_sb_initialized" not in st.session_state:
    for _k, _v in _load_sidebar().items():
        st.session_state[_k] = _v
    st.session_state["_sb_initialized"] = True

# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
selected_metrics: list = ALL_METRICS[:]
gt_dir_input:     str  = ""
selected_configs: list = []
download_types:   list = []

with st.sidebar:
    st.markdown("## 🌿 FusNet")
    st.caption("Bamboo Segmentation Demo")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Default", "Research"],
        key="sb_mode",
        horizontal=True,
        help="Default: 多模型预测  ·  Research: 含 GT 对比与指标",
    )
    st.divider()

    server_url = st.text_input(
        "Inference server URL",
        value="http://localhost:8000",
        key="sb_server_url",
        help="远端 server.py 的地址。SSH 端口转发后填 http://localhost:<port>",
    ).rstrip("/")

    health = check_health(server_url)
    if health:
        st.success(f"Server online · `{health.get('device', '?')}`", icon="✅")
    else:
        st.error("Server unreachable", icon="🔴")
        st.caption(f"无法连接 `{server_url}`，请检查 server.py 是否运行及端口转发。")

    st.divider()

    server_configs = fetch_server_configs(server_url) or {}
    all_names      = list(server_configs.keys())
    avail_names    = [n for n in all_names if server_configs[n].get("weight_available")]

    if not server_configs:
        st.warning("无法获取服务器配置", icon="⚠️")
    else:
        # 过滤缓存的选择项，确保只保留当前服务器提供的配置
        if "sb_selected_configs" in st.session_state:
            st.session_state["sb_selected_configs"] = [
                c for c in st.session_state["sb_selected_configs"] if c in all_names
            ]
        _cfg_default = st.session_state.get("sb_selected_configs", avail_names[:1] if avail_names else [])
        selected_configs = st.multiselect(
            "Network configurations",
            all_names,
            default=_cfg_default,
            key="sb_selected_configs",
            help="可同时选多个模型推理并对比结果",
        )
        for n in selected_configs:
            ok  = server_configs[n].get("weight_available", False)
            ico = "✅" if ok else "⚠️"
            st.caption(f"{ico} {n}")

    st.divider()

    conf_threshold = st.slider(
        "Confidence threshold",
        0.0, 1.0, 0.5, 0.01,
        key="sb_conf_threshold",
        help="像素判定为竹林的概率下限，调节不重新推理（本地实时计算）",
    )
    heatmap_alpha = st.slider("Heatmap blend α", 0.1, 0.9, 0.5,  0.05,
                              key="sb_heatmap_alpha",
                              help="热力图与原图混合比例")
    overlay_alpha = st.slider("Mask overlay α",  0.1, 0.9, 0.45, 0.05,
                              key="sb_overlay_alpha",
                              help="掩码叠加层不透明度")

    st.divider()
    _all_panel_names = PANEL_NAMES_RESEARCH if mode == "Research" else PANEL_NAMES_DEFAULT
    if "sb_download_types" in st.session_state:
        st.session_state["sb_download_types"] = [
            t for t in st.session_state["sb_download_types"] if t in _all_panel_names
        ]
    download_types = st.multiselect(
        "Download image types",
        _all_panel_names,
        default=st.session_state.get("sb_download_types", _all_panel_names),
        key="sb_download_types",
        help="选择 ZIP 下载时包含的图片类型，结构为 模型/类型/图片.png",
    )
    if not download_types:
        download_types = _all_panel_names[:]

    if mode == "Research":
        st.divider()
        st.markdown("**Research settings**")
        gt_dir_input = st.text_input(
            "Ground Truth folder (server path)",
            placeholder="/absolute/path/on/server/labels",
            key="sb_gt_dir_input",
            help="运行 server.py 的服务端机器上的标签文件夹绝对路径。",
        )
        if "sb_selected_metrics" in st.session_state:
            st.session_state["sb_selected_metrics"] = [
                m for m in st.session_state["sb_selected_metrics"] if m in ALL_METRICS
            ]
        selected_metrics = st.multiselect(
            "Metrics to display",
            ALL_METRICS,
            default=st.session_state.get("sb_selected_metrics", ALL_METRICS),
            key="sb_selected_metrics",
            help="选择在表格和图表中展示的指标",
        )
        if not selected_metrics:
            selected_metrics = ALL_METRICS[:]

    # 每次渲染后将当前 sidebar 状态写入磁盘，供刷新后恢复
    _save_sidebar({
        "sb_mode":             mode,
        "sb_server_url":       server_url,
        "sb_selected_configs": selected_configs,
        "sb_conf_threshold":   conf_threshold,
        "sb_heatmap_alpha":    heatmap_alpha,
        "sb_overlay_alpha":    overlay_alpha,
        "sb_download_types":   download_types,
        "sb_gt_dir_input":     gt_dir_input if mode == "Research" else "",
        "sb_selected_metrics": selected_metrics if mode == "Research" else list(ALL_METRICS),
    })


# ─── 主内容区 ─────────────────────────────────────────────────────────────────
st.title("FusNet · Bamboo Segmentation")
mode_label = (
    "Default — Multi-model Prediction"
    if mode == "Default"
    else "Research — GT Comparison & Metrics"
)
st.caption(
    f"Upload images (`.tif`, `.tiff`, `.png`, `.jpg`) to run inference.  "
    f"Mode: **{mode_label}**"
)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "expanders_expanded" not in st.session_state:
    st.session_state.expanders_expanded = True
if "_panel_cache" not in st.session_state:
    st.session_state["_panel_cache"] = {}

# 读取磁盘缓存元数据（早于 uploader，用于判断是否可以免上传恢复）
_disk_stems = _load_stems_meta()

uploaded_files = st.file_uploader(
    "Upload images",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key=f"up_{st.session_state.uploader_key}",
)

_use_disk_cache = not uploaded_files and bool(_disk_stems)

if not uploaded_files and not _use_disk_cache:
    st.info("👆 Upload at least one image to get started.", icon="ℹ️")
    st.stop()

# ─── 推理 / 磁盘缓存恢复 ──────────────────────────────────────────────────────
all_results: dict[str, dict]                 = {}
all_gt:      dict[str, Optional[np.ndarray]] = {}

if _use_disk_cache:
    # 从磁盘缓存恢复所有结果，无需服务器在线
    _cache_key = ("disk", tuple(sm["stem"] for sm in _disk_stems))
    if st.session_state.get("_infer_key") == _cache_key:
        all_results = st.session_state["_all_results"]
        all_gt      = st.session_state["_all_gt"]
    else:
        for sm in _disk_stems:
            img = _load_image_cache(sm["name"], sm["size"])
            if img is None:
                continue
            stem = sm["stem"]
            all_results[stem] = {"_raw": img}
            for cfg in sm.get("configs", []):
                r = _load_result_cache(sm["name"], sm["size"], cfg)
                if r:
                    all_results[stem][cfg] = r
            gt = _load_gt_cache(sm["name"], sm["size"])
            if gt is not None:
                all_gt[stem] = gt
        st.session_state["_infer_key"]   = _cache_key
        st.session_state["_all_results"] = all_results
        st.session_state["_all_gt"]      = all_gt
        st.session_state["_panel_cache"] = {}

    # 若 sidebar 没选模型，回退到缓存中存在的模型列表
    if not selected_configs:
        selected_configs = list(dict.fromkeys(
            cfg for sm in _disk_stems for cfg in sm.get("configs", [])
        ))

    st.info("💾 显示缓存结果（上次推理）。上传新图片可重新推理。", icon="💾")

else:
    # 新上传文件：检查服务器与模型选择
    if not health:
        st.error("推理服务器不可达，请先启动 `server.py` 并确认端口转发。", icon="🚨")
        st.stop()

    if not selected_configs:
        st.error("请在侧边栏选择至少一个模型配置。", icon="🚨")
        st.stop()

    _infer_key = (
        st.session_state.uploader_key,
        tuple((f.name, f.size) for f in uploaded_files),
        tuple(sorted(selected_configs)),
        mode,
        gt_dir_input.strip() if mode == "Research" else "",
    )

    if st.session_state.get("_infer_key") == _infer_key:
        all_results = st.session_state["_all_results"]
        all_gt      = st.session_state["_all_gt"]
    else:
        n_total  = len(uploaded_files) * len(selected_configs)
        progress = st.progress(0, text="Sending to inference server…")
        task_idx = 0

        for uf in uploaded_files:
            raw_bytes = uf.read()
            image     = read_image_bytes(raw_bytes, uf.name)
            stem      = Path(uf.name).stem

            all_results[stem] = {"_raw": image}

            if mode == "Research" and gt_dir_input.strip():
                all_gt[stem] = call_gt_api(server_url, gt_dir_input.strip(), stem)

            for cfg_name in selected_configs:
                task_idx += 1
                progress.progress(task_idx / n_total, text=f"Inferring {uf.name} [{cfg_name}]…")

                if not server_configs.get(cfg_name, {}).get("weight_available", False):
                    st.warning(f"`{cfg_name}` — weight not found on server, skipped.", icon="⚠️")
                    continue

                ikey = f"{cfg_name}::{uf.name}::{uf.size}"
                try:
                    probs, cam = call_infer_api(server_url, cfg_name, ikey, raw_bytes, uf.name)
                    all_results[stem][cfg_name] = {"probs": probs, "cam": cam}
                except requests.HTTPError as e:
                    st.error(f"**{uf.name}** [{cfg_name}] 服务器错误: {e.response.text}", icon="🚨")
                except Exception as e:
                    st.error(f"**{uf.name}** [{cfg_name}] 请求失败: {e}", icon="🚨")

        progress.empty()

        # 推理完成后写入磁盘缓存（与已有缓存合并）
        _existing_meta = {sm["stem"]: sm for sm in _load_stems_meta()}
        for uf in uploaded_files:
            stem = Path(uf.name).stem
            if stem not in all_results:
                continue
            _save_image_cache(uf.name, uf.size, all_results[stem]["_raw"])
            cfg_list = []
            for cfg in selected_configs:
                if cfg in all_results[stem]:
                    _save_result_cache(uf.name, uf.size, cfg,
                                       all_results[stem][cfg]["probs"],
                                       all_results[stem][cfg]["cam"])
                    cfg_list.append(cfg)
            gt = all_gt.get(stem)
            if gt is not None:
                _save_gt_cache(uf.name, uf.size, gt)
            _existing_meta[stem] = {"stem": stem, "name": uf.name,
                                     "size": uf.size, "configs": cfg_list}
        _save_stems_meta(list(_existing_meta.values()))

        st.session_state["_infer_key"]   = _infer_key
        st.session_state["_all_results"] = all_results
        st.session_state["_all_gt"]      = all_gt
        st.session_state["_panel_cache"] = {}

valid_stems = [
    s for s in all_results
    if any(c in all_results[s] for c in selected_configs)
]
if not valid_stems:
    st.stop()

n_done = sum(1 for s in valid_stems for c in selected_configs if c in all_results[s])
st.success(f"Done — {len(valid_stems)} image(s), {n_done} inference(s) completed.", icon="✅")

# ─── 搜索 + 清除 + 折叠控制 + 全局下载 ──────────────────────────────────────
col_search, col_clear, col_toggle, col_dl = st.columns([3, 1, 1, 1])
with col_search:
    search_query = st.text_input(
        "search",
        placeholder="🔍  Filter images by filename…",
        label_visibility="collapsed",
    )
with col_clear:
    if uploaded_files:
        if st.button("🗑️ Clear All", use_container_width=True):
            _clear_disk_cache()
            st.session_state.uploader_key += 1
            st.session_state.pop("_infer_key", None)
            st.rerun()
    elif _use_disk_cache:
        if st.button("🗑️ 清除缓存", use_container_width=True,
                     help="清除刷新保留的缓存，回到初始状态"):
            _clear_disk_cache()
            st.session_state.pop("_infer_key", None)
            st.rerun()
with col_toggle:
    toggle_label = "⊖ Collapse All" if st.session_state.expanders_expanded else "⊕ Expand All"
    if st.button(toggle_label, use_container_width=True):
        st.session_state.expanders_expanded = not st.session_state.expanders_expanded
        st.rerun()
with col_dl:
    if download_types:
        _zip_cache_key = (
            st.session_state.get("_infer_key"),
            conf_threshold, overlay_alpha, heatmap_alpha,
            tuple(sorted(download_types)),
        )
        if st.session_state.get("_zip_cache_key") != _zip_cache_key:
            zip_bytes = build_results_zip(
                all_results, all_gt, selected_configs,
                conf_threshold, overlay_alpha, heatmap_alpha,
                download_types,
            )
            st.session_state["_zip_cache_key"] = _zip_cache_key
            st.session_state["_zip_bytes"]     = zip_bytes
        else:
            zip_bytes = st.session_state["_zip_bytes"]
        st.download_button(
            "⬇ Download ZIP",
            zip_bytes,
            "fusnet_results.zip",
            "application/zip",
            use_container_width=True,
        )

filtered_stems = [
    s for s in valid_stems
    if not search_query or search_query.lower() in s.lower()
]
if search_query and not filtered_stems:
    st.info(f"No images match `{search_query}`.", icon="🔍")
    st.stop()

# ─── 布局：Research 模式用左右两标签页 ───────────────────────────────────────
summary_metrics: dict[str, list[dict]] = {c: [] for c in selected_configs}

if mode == "Research":
    results_tab, summary_tab = st.tabs(["🖼  Prediction Results", "📊  Summary"])
else:
    results_tab = st.container()
    summary_tab = None


# ─── 下载按钮 SVG 图标（与 Streamlit 全屏按钮风格一致） ────────────────────
_ICON_DL = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2.2"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="7 10 12 15 17 10"/>'
    '<line x1="12" y1="15" x2="12" y2="3"/>'
    '</svg>'
)


def _dl_float_html(img_bytes: bytes, filename: str) -> str:
    """Zero-height div with download link, pulled up into the image via negative margin."""
    b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"
    return (
        f'<div class="img-dl-float">'
        f'<a href="{data_uri}" download="{filename}" class="img-dl-btn" title="Download">'
        f'{_ICON_DL}</a>'
        f'</div>'
    )


# ─── 辅助：单 panel 展示（原生全屏 + 悬停下载按钮） ──────────────────────────
def _panel(col, title: str, arr: np.ndarray, mode_l: str = "RGB", stem: str = "", cfg: str = "", img_bytes: bytes = None):
    parts = [x for x in [stem, cfg.replace("/", "_").replace(" ", "_")] if x]
    fname = "_".join(parts + [title.replace(" ", "_")]) + ".png"
    with col:
        st.markdown(f'<div class="img-card-title">{title}</div>', unsafe_allow_html=True)
        st.image(arr, use_container_width=True)
        if img_bytes is None:
            img_bytes = pil_png_bytes(arr, mode_l)
        st.markdown(_dl_float_html(img_bytes, fname), unsafe_allow_html=True)


# ─── 展示阶段 ─────────────────────────────────────────────────────────────────
with results_tab:
    for img_idx, stem in enumerate(filtered_stems):
        with st.expander(f"📷  {stem}", expanded=st.session_state.expanders_expanded):
            raw_image = all_results[stem]["_raw"]
            h, w      = raw_image.shape[:2]
            gt_bool   = all_gt.get(stem)

            if mode == "Research" and gt_dir_input.strip() and gt_bool is None:
                st.warning(f"未找到 `{stem}` 对应的 GT 标签文件，仅展示预测结果。", icon="⚠️")

            if mode == "Research":
                view_tab, metric_tab = st.tabs(["🖼  Prediction Views", "📊  Image Metrics"])
            else:
                view_tab   = st.container()
                metric_tab = None

            with view_tab:
                _pcache = st.session_state["_panel_cache"]

                _raw_key = ("_raw", stem)
                if _raw_key not in _pcache:
                    _pcache[_raw_key] = pil_png_bytes(raw_image)
                raw_png = _pcache[_raw_key]

                for ci, cfg_name in enumerate(selected_configs):
                    if cfg_name not in all_results[stem]:
                        continue

                    probs = all_results[stem][cfg_name]["probs"]
                    cam   = all_results[stem][cfg_name]["cam"]

                    _pred_key = (stem, cfg_name, conf_threshold, overlay_alpha, heatmap_alpha)
                    if _pred_key not in _pcache:
                        mb  = resize_mask_bool(probs >= conf_threshold, h, w)
                        mu8 = mb.astype(np.uint8) * 255
                        pb  = overlay_mask(raw_image, mb, overlay_alpha, COLOR_PRED)
                        hm  = apply_heatmap(raw_image, cam, heatmap_alpha)
                        _pcache[_pred_key] = {
                            "mask_bool":      mb,
                            "mask_u8":        mu8,
                            "mask_u8_png":    pil_png_bytes(mu8, "L"),
                            "pred_blend":     pb,
                            "pred_blend_png": pil_png_bytes(pb),
                            "heatmap":        hm,
                            "heatmap_png":    pil_png_bytes(hm),
                        }
                    pc        = _pcache[_pred_key]
                    mask_bool = pc["mask_bool"]
                    mask_u8   = pc["mask_u8"]
                    pred_blend = pc["pred_blend"]
                    heatmap   = pc["heatmap"]

                    if ci > 0:
                        st.markdown("---")
                    st.markdown(
                        f'<div class="model-header">▸ {cfg_name}</div>',
                        unsafe_allow_html=True,
                    )

                    use_6 = mode == "Research" and gt_bool is not None
                    if use_6:
                        _gt_key = (stem, overlay_alpha)
                        if _gt_key not in _pcache:
                            gr  = resize_mask_bool(gt_bool, h, w)
                            gu8 = gr.astype(np.uint8) * 255
                            gb  = overlay_mask(raw_image, gr, overlay_alpha, COLOR_GT)
                            _pcache[_gt_key] = {
                                "gt_r":         gr,
                                "gt_u8":        gu8,
                                "gt_u8_png":    pil_png_bytes(gu8, "L"),
                                "gt_blend":     gb,
                                "gt_blend_png": pil_png_bytes(gb),
                            }
                        gc       = _pcache[_gt_key]
                        gt_r     = gc["gt_r"]
                        gt_u8    = gc["gt_u8"]
                        gt_blend = gc["gt_blend"]

                        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
                        _panel(c1, "Original",     raw_image,  stem=stem, cfg=cfg_name, img_bytes=raw_png)
                        _panel(c2, "GT Mask",      gt_u8,    "L", stem=stem, cfg=cfg_name, img_bytes=gc["gt_u8_png"])
                        _panel(c3, "Pred Mask",    mask_u8,  "L", stem=stem, cfg=cfg_name, img_bytes=pc["mask_u8_png"])
                        _panel(c4, "GT Overlay",   gt_blend,     stem=stem, cfg=cfg_name, img_bytes=gc["gt_blend_png"])
                        _panel(c5, "Pred Overlay", pred_blend,   stem=stem, cfg=cfg_name, img_bytes=pc["pred_blend_png"])
                        _panel(c6, "GradCAM",      heatmap,      stem=stem, cfg=cfg_name, img_bytes=pc["heatmap_png"])
                    else:
                        c1, c2, c3, c4 = st.columns(4, gap="medium")
                        _panel(c1, "Original",     raw_image,  stem=stem, cfg=cfg_name, img_bytes=raw_png)
                        _panel(c2, "Pred Mask",    mask_u8,  "L", stem=stem, cfg=cfg_name, img_bytes=pc["mask_u8_png"])
                        _panel(c3, "Pred Overlay", pred_blend,   stem=stem, cfg=cfg_name, img_bytes=pc["pred_blend_png"])
                        _panel(c4, "GradCAM",      heatmap,      stem=stem, cfg=cfg_name, img_bytes=pc["heatmap_png"])

            if mode == "Research" and metric_tab is not None:
                with metric_tab:
                    if gt_bool is None:
                        if not gt_dir_input.strip():
                            st.info("请在侧边栏指定服务端 Ground Truth 文件夹路径。", icon="ℹ️")
                        else:
                            st.warning("未找到对应 GT 文件，无法计算该图指标。", icon="⚠️")
                    else:
                        gt_r = resize_mask_bool(gt_bool, h, w)
                        img_model_metrics: dict[str, dict] = {}

                        _pcache = st.session_state["_panel_cache"]
                        for cfg_name in selected_configs:
                            if cfg_name not in all_results[stem]:
                                continue
                            _pred_key = (stem, cfg_name, conf_threshold, overlay_alpha, heatmap_alpha)
                            if _pred_key in _pcache:
                                mask_bool = _pcache[_pred_key]["mask_bool"]
                            else:
                                probs     = all_results[stem][cfg_name]["probs"]
                                mask_bool = resize_mask_bool(probs >= conf_threshold, h, w)
                            m         = compute_metrics(mask_bool, gt_r)
                            img_model_metrics[cfg_name] = m
                            summary_metrics[cfg_name].append(m)

                        if img_model_metrics and selected_metrics:
                            st.markdown(
                                f"**`{stem}` — per-model metrics** "
                                f"(confidence threshold = {conf_threshold:.2f})"
                            )
                            st.markdown(
                                paper_table_html(img_model_metrics, selected_metrics),
                                unsafe_allow_html=True,
                            )


# ─── 汇总标签页（研究模式） ───────────────────────────────────────────────────
if mode == "Research" and summary_tab is not None:
    with summary_tab:
        has_summary = any(len(v) > 0 for v in summary_metrics.values())
        if not has_summary:
            if not gt_dir_input.strip():
                st.info("请在侧边栏指定服务端 GT 文件夹路径以启用汇总指标。", icon="ℹ️")
            else:
                st.info("完成至少一张带 GT 标签图片的推理后，汇总结果将显示在此处。", icon="ℹ️")
        else:
            avg_metrics: dict[str, dict] = {}
            for cfg_name, mlist in summary_metrics.items():
                if mlist:
                    avg_metrics[cfg_name] = {
                        m: float(np.mean([ml[m] for ml in mlist]))
                        for m in ALL_METRICS
                    }

            if avg_metrics and selected_metrics:
                n_imgs = max(len(v) for v in summary_metrics.values())
                st.caption(
                    f"Averaged over **{n_imgs}** image(s) with matched GT labels  ·  "
                    f"confidence threshold = {conf_threshold:.2f}"
                )
                sum_tab1, sum_tab2 = st.tabs(["📋  Metrics Table", "📈  Bar Chart"])

                with sum_tab1:
                    st.markdown(
                        paper_table_html(avg_metrics, selected_metrics),
                        unsafe_allow_html=True,
                    )

                with sum_tab2:
                    chart_bytes = metrics_barchart_png(avg_metrics, selected_metrics)
                    st.image(chart_bytes, use_container_width=True)
                    st.markdown(
                        _dl_float_html(chart_bytes, "fusnet_metrics_chart.png"),
                        unsafe_allow_html=True,
                    )

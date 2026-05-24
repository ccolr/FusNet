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
st.markdown(
    """
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

  /* ── fusnet HTML panel grid (single st.markdown call per stem) ── */
  .fusnet-panels-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-bottom: 0.8rem; }
  .fusnet-panels-6 { display: grid; grid-template-columns: repeat(6, 1fr); gap: 0.5rem; margin-bottom: 0.8rem; }
  .fusnet-panel:hover .img-dl-btn { opacity: 1 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
COLOR_PRED = (220, 30, 30)  # red overlay for predictions
COLOR_GT = (220, 30, 30)  # red overlay for GT (same as prediction)
ALL_METRICS = ["Accuracy", "Precision", "Recall", "F1", "IoU", "mIoU"]

PANEL_NAMES_DEFAULT = ["Original", "Pred_Mask", "Pred_Overlay", "GradCAM"]
PANEL_NAMES_RESEARCH = ["Original", "GT_Mask", "Pred_Mask", "GT_Overlay", "Pred_Overlay", "GradCAM"]

_ICON_DL = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"'
    ' fill="none" stroke="currentColor" stroke-width="2.2"'
    ' stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
    '<polyline points="7 10 12 15 17 10"/>'
    '<line x1="12" y1="15" x2="12" y2="3"/>'
    "</svg>"
)

# ─── 磁盘缓存工具 ─────────────────────────────────────────────────────────────
_DISK_CACHE_DIR = Path(".fusnet_session")


def _fhash(name: str, size: int) -> str:
    return hashlib.md5(f"{name}:{size}".encode()).hexdigest()[:12]


def _chash(cfg: str) -> str:
    return hashlib.md5(cfg.encode()).hexdigest()[:8]


def _json_save(fname: str, data) -> None:
    try:
        _DISK_CACHE_DIR.mkdir(exist_ok=True)
        (_DISK_CACHE_DIR / fname).write_text(json.dumps(data))
    except Exception:
        pass


def _json_load(fname: str, default):
    p = _DISK_CACHE_DIR / fname
    try:
        return json.loads(p.read_text()) if p.exists() else default
    except Exception:
        return default


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


@st.cache_resource
def _register_cleanup_handler() -> bool:
    import atexit
    atexit.register(_clear_disk_cache)
    return True

_register_cleanup_handler()


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
    c = np.array(color, dtype=np.float32)
    for i in range(3):
        out[:, :, i] = np.where(
            mask,
            out[:, :, i] * (1 - alpha) + c[i] * alpha,
            out[:, :, i],
        )
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    h, w = image.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    hmap = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    blend = image.astype(np.float32) * (1 - alpha) + hmap.astype(np.float32) * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def resize_mask_bool(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((w, h), Image.NEAREST)) > 127


# ─── 指标计算 ─────────────────────────────────────────────────────────────────
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool).ravel(), gt.astype(bool).ravel()
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    eps = 1e-8
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fn + fp + eps)
    return {
        "Accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "Precision": tp / (tp + fp + eps),
        "Recall": tp / (tp + fn + eps),
        "F1": 2 * tp / (2 * tp + fp + fn + eps),
        "IoU": iou_fg,
        "mIoU": (iou_fg + iou_bg) / 2,
    }


# ─── 论文级 HTML 表格（booktabs 风格，居中） ──────────────────────────────────
def paper_table_html(model_metrics: dict, metrics: list) -> str:
    if not model_metrics or not metrics:
        return ""
    best = {m: max(model_metrics, key=lambda mn: model_metrics[mn].get(m, 0)) for m in metrics}
    header = "<thead><tr><th>Model</th>" + "".join(f"<th>{m}</th>" for m in metrics) + "</tr></thead>"
    rows = ""
    for mn, mv in model_metrics.items():
        cells = ""
        for m in metrics:
            v = mv.get(m, float("nan"))
            cls = ' class="best"' if mn == best[m] else ""
            cells += f"<td{cls}>{v:.4f}</td>"
        rows += f"<tr><td>{mn}</td>{cells}</tr>"
    return (
        f'<div class="paper-table-wrap">' f'<table class="paper-table">{header}<tbody>{rows}</tbody></table>' f"</div>"
    )


# ─── 论文级柱状图 ─────────────────────────────────────────────────────────────
def metrics_barchart_png(model_metrics: dict, metrics: list) -> bytes:
    models = list(model_metrics.keys())
    n_m, nc = len(metrics), len(models)
    y = np.arange(n_m)
    bw = min(0.65 / max(nc, 1), 0.25)
    offsets = (np.arange(nc) - (nc - 1) / 2) * bw
    cmap = plt.get_cmap("tab10").colors  # type: ignore

    fig_h = max(5.5, n_m * 0.8 + nc * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    all_vals = [model_metrics[mn].get(m, 0) for mn in models for m in metrics]
    xmax = min(1.15, max(all_vals, default=0) + 0.10)

    for i, (mn, off) in enumerate(zip(models, offsets)):
        vals = [model_metrics[mn].get(m, 0) for m in metrics]
        bars = ax.barh(y + off, vals, bw * 0.90, label=mn, color=cmap[i % 10], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(
                min(b.get_width() + 0.005, xmax - 0.002),
                b.get_y() + b.get_height() / 2,
                f"{v:.3f}",
                ha="left",
                va="center",
                fontsize=8,
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
    server_url: str,
    config_name: str,
    image_key: str,
    raw_bytes: bytes,
    filename: str,
) -> tuple[np.ndarray, np.ndarray]:
    resp = requests.post(
        f"{server_url}/infer",
        files={"file": (filename, raw_bytes)},
        data={"config_name": config_name},
        timeout=180,
    )
    resp.raise_for_status()
    d = resp.json()
    probs = np.frombuffer(base64.b64decode(d["probs"]), dtype=np.float32).reshape(tuple(d["probs_shape"]))
    cam = np.frombuffer(base64.b64decode(d["cam"]), dtype=np.float32).reshape(tuple(d["cam_shape"]))
    return probs, cam


@st.cache_data(show_spinner=False)
def call_gt_api(server_url: str, gt_dir: str, stem: str) -> Optional[np.ndarray]:
    try:
        r = requests.get(
            f"{server_url}/gt",
            params={"gt_dir": gt_dir, "stem": stem},
            timeout=30,
        )
        if r.status_code == 200:
            d = r.json()
            mask = np.frombuffer(base64.b64decode(d["mask"]), dtype=np.uint8).reshape(tuple(d["shape"]))
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
    all_results: dict,
    all_gt: dict,
    selected_configs: list,
    conf_threshold: float,
    overlay_alpha: float,
    heatmap_alpha: float,
    download_types: list,
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in all_results:
            raw_image = all_results[stem]["_raw"]
            h, w = raw_image.shape[:2]
            gt_bool = all_gt.get(stem)

            for cfg_name in selected_configs:
                if cfg_name not in all_results[stem]:
                    continue
                probs = all_results[stem][cfg_name]["probs"]
                cam = all_results[stem][cfg_name]["cam"]
                mask_bool = resize_mask_bool(probs >= conf_threshold, h, w)
                mask_u8 = mask_bool.astype(np.uint8) * 255
                pred_blend = overlay_mask(raw_image, mask_bool, overlay_alpha, COLOR_PRED)
                heatmap = apply_heatmap(raw_image, cam, heatmap_alpha)

                safe_cfg = cfg_name.replace("/", "_").replace(" ", "_")

                panels: dict[str, tuple[np.ndarray, str]] = {
                    "Original": (raw_image, "RGB"),
                    "Pred_Mask": (mask_u8, "L"),
                    "Pred_Overlay": (pred_blend, "RGB"),
                    "GradCAM": (heatmap, "RGB"),
                }
                if gt_bool is not None:
                    gt_r = resize_mask_bool(gt_bool, h, w)
                    gt_u8 = gt_r.astype(np.uint8) * 255
                    gt_blend = overlay_mask(raw_image, gt_r, overlay_alpha, COLOR_GT)
                    panels["GT_Mask"] = (gt_u8, "L")
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
    for _k, _v in _json_load("sidebar.json", {}).items():
        st.session_state[_k] = _v
    st.session_state["_sb_initialized"] = True

# 保证 widget key 在 session_state 中有默认值，避免同时传 value= 引发冲突警告
st.session_state.setdefault("sb_server_url", "http://localhost:8000")
st.session_state.setdefault("sb_conf_threshold", 0.5)
st.session_state.setdefault("sb_heatmap_alpha", 0.5)
st.session_state.setdefault("sb_overlay_alpha", 0.45)

# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
selected_metrics: list = ALL_METRICS[:]
gt_dir_input: str = ""
selected_configs: list = []
download_types: list = []

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
    all_names = list(server_configs.keys())
    avail_names = [n for n in all_names if server_configs[n].get("weight_available")]

    if not server_configs:
        st.warning("无法获取服务器配置", icon="⚠️")
    else:
        _cfg_saved = st.session_state.get("sb_selected_configs", avail_names[:1] if avail_names else [])
        st.markdown("**Network configurations**")
        selected_configs = []
        for n in all_names:
            ok = server_configs[n].get("weight_available", False)
            ico = "✅" if ok else "⚠️"
            _ck = f"sb_cfg__{n}"
            if _ck not in st.session_state:
                st.session_state[_ck] = n in _cfg_saved
            if st.checkbox(f"{ico} {n}", key=_ck):
                selected_configs.append(n)

    st.divider()

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key="sb_conf_threshold",
        help="像素判定为竹林的概率下限，调节不重新推理（本地实时计算）",
    )
    heatmap_alpha = st.slider(
        "Heatmap blend α", min_value=0.1, max_value=0.9, step=0.05, key="sb_heatmap_alpha", help="热力图与原图混合比例"
    )
    overlay_alpha = st.slider("Mask overlay α", min_value=0.1, max_value=0.9, step=0.05, key="sb_overlay_alpha", help="掩码叠加层不透明度")

    st.divider()
    _all_panel_names = PANEL_NAMES_RESEARCH if mode == "Research" else PANEL_NAMES_DEFAULT
    _dl_saved = st.session_state.get("sb_download_types", _all_panel_names)
    st.markdown("**Download image types**")
    download_types = []
    for t in _all_panel_names:
        _dk = f"sb_dl__{t}"
        if _dk not in st.session_state:
            st.session_state[_dk] = t in _dl_saved
        if st.checkbox(t, key=_dk):
            download_types.append(t)
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
        _metrics_saved = st.session_state.get("sb_selected_metrics", ALL_METRICS)
        st.markdown("**Metrics to display**")
        selected_metrics = []
        for m in ALL_METRICS:
            _mk = f"sb_metric__{m}"
            if _mk not in st.session_state:
                st.session_state[_mk] = m in _metrics_saved
            if st.checkbox(m, key=_mk):
                selected_metrics.append(m)
        if not selected_metrics:
            selected_metrics = ALL_METRICS[:]

    # 每次渲染后将当前 sidebar 状态写入磁盘，供刷新后恢复
    _json_save("sidebar.json", {
        "sb_mode": mode,
        "sb_server_url": server_url,
        "sb_selected_configs": selected_configs,
        "sb_conf_threshold": conf_threshold,
        "sb_heatmap_alpha": heatmap_alpha,
        "sb_overlay_alpha": overlay_alpha,
        "sb_download_types": download_types,
        "sb_gt_dir_input": gt_dir_input if mode == "Research" else "",
        "sb_selected_metrics": selected_metrics if mode == "Research" else list(ALL_METRICS),
    })


# ─── 主内容区 ─────────────────────────────────────────────────────────────────
st.title("FusNet · Bamboo Segmentation")
mode_label = "Default — Multi-model Prediction" if mode == "Default" else "Research — GT Comparison & Metrics"
st.caption(f"Upload images (`.tif`, `.tiff`, `.png`, `.jpg`) to run inference.  " f"Mode: **{mode_label}**")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "expanders_expanded" not in st.session_state:
    st.session_state.expanders_expanded = True
if "_panel_cache" not in st.session_state:
    st.session_state["_panel_cache"] = {}

# 读取磁盘缓存元数据（早于 uploader，用于判断是否可以免上传恢复）
_disk_stems = _json_load("stems.json", [])

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
all_results: dict[str, dict] = {}
all_gt: dict[str, Optional[np.ndarray]] = {}

if _use_disk_cache:
    # 从磁盘缓存恢复所有结果，无需服务器在线
    _cache_key = ("disk", tuple(sm["stem"] for sm in _disk_stems))
    if st.session_state.get("_infer_key") == _cache_key:
        all_results = st.session_state["_all_results"]
        all_gt = st.session_state["_all_gt"]
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
        st.session_state["_infer_key"] = _cache_key
        st.session_state["_all_results"] = all_results
        st.session_state["_all_gt"] = all_gt
        st.session_state["_panel_cache"] = {}

    # 若 sidebar 没选模型，回退到缓存中存在的模型列表
    if not selected_configs:
        selected_configs = list(dict.fromkeys(cfg for sm in _disk_stems for cfg in sm.get("configs", [])))

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
        all_gt = st.session_state["_all_gt"]
    else:
        n_total = len(uploaded_files) * len(selected_configs)
        progress = st.progress(0, text="Sending to inference server…")
        task_idx = 0

        for uf in uploaded_files:
            raw_bytes = uf.read()
            image = read_image_bytes(raw_bytes, uf.name)
            stem = Path(uf.name).stem

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
        _existing_meta = {sm["stem"]: sm for sm in _json_load("stems.json", [])}
        for uf in uploaded_files:
            stem = Path(uf.name).stem
            if stem not in all_results:
                continue
            _save_image_cache(uf.name, uf.size, all_results[stem]["_raw"])
            cfg_list = []
            for cfg in selected_configs:
                if cfg in all_results[stem]:
                    _save_result_cache(
                        uf.name, uf.size, cfg, all_results[stem][cfg]["probs"], all_results[stem][cfg]["cam"]
                    )
                    cfg_list.append(cfg)
            gt = all_gt.get(stem)
            if gt is not None:
                _save_gt_cache(uf.name, uf.size, gt)
            _existing_meta[stem] = {"stem": stem, "name": uf.name, "size": uf.size, "configs": cfg_list}
        _json_save("stems.json", list(_existing_meta.values()))

        st.session_state["_infer_key"] = _infer_key
        st.session_state["_all_results"] = all_results
        st.session_state["_all_gt"] = all_gt
        st.session_state["_panel_cache"] = {}

valid_stems = [s for s in all_results if any(c in all_results[s] for c in selected_configs)]
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
        if st.button("🗑️ 清除缓存", use_container_width=True, help="清除刷新保留的缓存，回到初始状态"):
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
            conf_threshold,
            overlay_alpha,
            heatmap_alpha,
            tuple(sorted(download_types)),
        )
        if st.session_state.get("_zip_cache_key") != _zip_cache_key:
            zip_bytes = build_results_zip(
                all_results,
                all_gt,
                selected_configs,
                conf_threshold,
                overlay_alpha,
                heatmap_alpha,
                download_types,
            )
            st.session_state["_zip_cache_key"] = _zip_cache_key
            st.session_state["_zip_bytes"] = zip_bytes
        else:
            zip_bytes = st.session_state["_zip_bytes"]
        st.download_button(
            "⬇ Download ZIP",
            zip_bytes,
            "fusnet_results.zip",
            "application/zip",
            use_container_width=True,
        )

filtered_stems = [s for s in valid_stems if not search_query or search_query.lower() in s.lower()]
if search_query and not filtered_stems:
    st.info(f"No images match `{search_query}`.", icon="🔍")
    st.stop()

# ─── 分页（每页最多 10 张，防止大量 widget 导致 WebSocket 超时） ───────────────
_PAGE_SIZE = 10
_n_pages = max(1, (len(filtered_stems) + _PAGE_SIZE - 1) // _PAGE_SIZE)
if "page_idx" not in st.session_state:
    st.session_state.page_idx = 0
if st.session_state.get("_last_infer_key") != st.session_state.get("_infer_key"):
    st.session_state.page_idx = 0
    st.session_state["_last_infer_key"] = st.session_state.get("_infer_key")
st.session_state.page_idx = max(0, min(st.session_state.page_idx, _n_pages - 1))
_p0 = st.session_state.page_idx * _PAGE_SIZE
page_stems = filtered_stems[_p0 : _p0 + _PAGE_SIZE]

# ─── Research 模式：对全部图片预计算汇总指标（避免翻页后 summary 数据缺失） ────
summary_metrics: dict[str, list[dict]] = {c: [] for c in selected_configs}
if mode == "Research":
    _sm_key = (st.session_state.get("_infer_key"), conf_threshold, tuple(sorted(selected_configs)))
    if st.session_state.get("_sm_cache_key") != _sm_key:
        _sm_data: dict[str, list[dict]] = {c: [] for c in selected_configs}
        for _s in valid_stems:
            _gt = all_gt.get(_s)
            if _gt is None:
                continue
            _h, _w = all_results[_s]["_raw"].shape[:2]
            _gt_r = resize_mask_bool(_gt, _h, _w)
            for _cfg in selected_configs:
                if _cfg not in all_results[_s]:
                    continue
                _mb = resize_mask_bool(all_results[_s][_cfg]["probs"] >= conf_threshold, _h, _w)
                _sm_data[_cfg].append(compute_metrics(_mb, _gt_r))
        st.session_state["_sm_cache_key"] = _sm_key
        st.session_state["_sm_data"] = _sm_data
    summary_metrics = st.session_state["_sm_data"]


def _dl_float_html(img_bytes: bytes, filename: str) -> str:
    b64 = base64.b64encode(img_bytes).decode()
    data_uri = f"data:image/png;base64,{b64}"
    return (
        f'<div class="img-dl-float">'
        f'<a href="{data_uri}" download="{filename}" class="img-dl-btn" title="Download">'
        f"{_ICON_DL}</a>"
        f"</div>"
    )


def _panel_html(title: str, b64: str, fname: str) -> str:
    uri = f"data:image/png;base64,{b64}"
    return (
        f'<div class="fusnet-panel">'
        f'<div class="img-card-title">{title}</div>'
        f'<div style="position:relative">'
        f'<img src="{uri}" style="width:100%;display:block;border-radius:4px">'
        f'<div style="position:absolute;top:0.3rem;right:0.3rem">'
        f'<a href="{uri}" download="{fname}" class="img-dl-btn" title="Download">{_ICON_DL}</a>'
        f'</div></div></div>'
    )


# ─── 渲染缓存（PNG + base64 + HTML 预计算，仅在关键参数改变时重建） ────────────
_render_key = (conf_threshold, overlay_alpha, heatmap_alpha)
_panel_render_key = (
    st.session_state.get("_infer_key"),
    _render_key,
    mode,
    tuple(sorted(selected_configs)),
    tuple(selected_metrics) if mode == "Research" else (),
)

if st.session_state.get("_panel_render_key") != _panel_render_key:
    _new_pcache: dict = {}
    _safe_name = lambda n: n.replace("/", "_").replace(" ", "_")

    for _stem in valid_stems:
        _raw = all_results[_stem]["_raw"]
        _h, _w = _raw.shape[:2]

        _raw_b64 = base64.b64encode(pil_png_bytes(_raw)).decode()

        # ── GT ──
        _gt = all_gt.get(_stem)
        _gc_entry: Optional[dict] = None
        if _gt is not None:
            _gr = resize_mask_bool(_gt, _h, _w)
            _gt_u8 = pil_png_bytes(_gr.astype(np.uint8) * 255, "L")
            _gt_bl = pil_png_bytes(overlay_mask(_raw, _gr, overlay_alpha, COLOR_GT))
            _gc_entry = {
                "gt_r": _gr,
                "gt_u8_b64": base64.b64encode(_gt_u8).decode(),
                "gt_blend_b64": base64.b64encode(_gt_bl).decode(),
            }

        # ── per-config predictions ──
        _cfg_entries: dict = {}
        for _cfg in selected_configs:
            if _cfg not in all_results[_stem]:
                continue
            _probs = all_results[_stem][_cfg]["probs"]
            _cam = all_results[_stem][_cfg]["cam"]
            _mb = resize_mask_bool(_probs >= conf_threshold, _h, _w)
            _mu8 = pil_png_bytes(_mb.astype(np.uint8) * 255, "L")
            _pb = pil_png_bytes(overlay_mask(_raw, _mb, overlay_alpha, COLOR_PRED))
            _hm = pil_png_bytes(apply_heatmap(_raw, _cam, heatmap_alpha))
            _cfg_entries[_cfg] = {
                "mask_bool": _mb,
                "mask_u8_b64": base64.b64encode(_mu8).decode(),
                "pred_blend_b64": base64.b64encode(_pb).decode(),
                "heatmap_b64": base64.b64encode(_hm).decode(),
            }

        # ── pre-build view HTML (single st.markdown call per stem) ──
        _vparts: list = []
        for _ci, _cfg in enumerate(selected_configs):
            if _cfg not in _cfg_entries:
                continue
            _ce = _cfg_entries[_cfg]
            _sc = _safe_name(_cfg)
            _use6 = mode == "Research" and _gc_entry is not None
            _ncols = 6 if _use6 else 4
            if _ci > 0:
                _vparts.append('<hr class="sample-divider" style="margin:0.8rem 0">')
            _vparts.append(f'<div class="model-header">▸ {_cfg}</div>')
            _vparts.append(f'<div class="fusnet-panels-{_ncols}">')
            _pd = [("Original", _raw_b64, f"{_stem}_{_sc}_Original.png")]
            if _use6:
                _pd += [
                    ("GT Mask",      _gc_entry["gt_u8_b64"],    f"{_stem}_{_sc}_GT_Mask.png"),
                    ("Pred Mask",    _ce["mask_u8_b64"],         f"{_stem}_{_sc}_Pred_Mask.png"),
                    ("GT Overlay",   _gc_entry["gt_blend_b64"],  f"{_stem}_{_sc}_GT_Overlay.png"),
                    ("Pred Overlay", _ce["pred_blend_b64"],      f"{_stem}_{_sc}_Pred_Overlay.png"),
                    ("GradCAM",      _ce["heatmap_b64"],         f"{_stem}_{_sc}_GradCAM.png"),
                ]
            else:
                _pd += [
                    ("Pred Mask",    _ce["mask_u8_b64"],    f"{_stem}_{_sc}_Pred_Mask.png"),
                    ("Pred Overlay", _ce["pred_blend_b64"], f"{_stem}_{_sc}_Pred_Overlay.png"),
                    ("GradCAM",      _ce["heatmap_b64"],    f"{_stem}_{_sc}_GradCAM.png"),
                ]
            for _t, _b, _fn in _pd:
                _vparts.append(_panel_html(_t, _b, _fn))
            _vparts.append("</div>")
        _new_pcache[("_view_html", _stem)] = "".join(_vparts)

        # ── pre-build metrics HTML (Research + GT only) ──
        if mode == "Research" and _gc_entry is not None and _cfg_entries and selected_metrics:
            _im: dict = {
                _cfg: compute_metrics(_ce["mask_bool"], _gc_entry["gt_r"])
                for _cfg, _ce in _cfg_entries.items()
            }
            _new_pcache[("_metrics_html", _stem)] = (
                f"<p><strong><code>{_stem}</code> — per-model metrics"
                f" (confidence threshold = {conf_threshold:.2f})</strong></p>"
                + paper_table_html(_im, selected_metrics)
            )
        else:
            _new_pcache[("_metrics_html", _stem)] = None

    st.session_state["_panel_cache"] = _new_pcache
    st.session_state["_panel_render_key"] = _panel_render_key

# ─── 布局：Research 模式用左右两标签页 ───────────────────────────────────────
if mode == "Research":
    results_tab, summary_tab = st.tabs(["🖼  Prediction Results", "📊  Summary"])
else:
    results_tab = st.container()
    summary_tab = None


# ─── 展示阶段 ─────────────────────────────────────────────────────────────────
with results_tab:
    if _n_pages > 1:
        _pc1, _pc2, _pc3 = st.columns([1, 4, 1])
        with _pc1:
            if st.button("◀ Prev", disabled=st.session_state.page_idx == 0, use_container_width=True, key="pg_prev_t"):
                st.session_state.page_idx -= 1
                st.rerun()
        with _pc2:
            st.markdown(
                f"<div style='text-align:center;padding:6px 0;"
                f"color:#6b7280;font-size:0.9rem;'>"
                f"Page {st.session_state.page_idx + 1} / {_n_pages}"
                f" &nbsp;·&nbsp; {len(filtered_stems)} images total</div>",
                unsafe_allow_html=True,
            )
        with _pc3:
            if st.button(
                "Next ▶", disabled=st.session_state.page_idx == _n_pages - 1, use_container_width=True, key="pg_next_t"
            ):
                st.session_state.page_idx += 1
                st.rerun()

    for stem in page_stems:
        with st.expander(f"📷  {stem}", expanded=st.session_state.expanders_expanded):
            gt_bool = all_gt.get(stem)

            if mode == "Research" and gt_dir_input.strip() and gt_bool is None:
                st.warning(f"未找到 `{stem}` 对应的 GT 标签文件，仅展示预测结果。", icon="⚠️")

            if mode == "Research":
                view_tab, metric_tab = st.tabs(["🖼  Prediction Views", "📊  Image Metrics"])
            else:
                view_tab = st.container()
                metric_tab = None

            with view_tab:
                _view_html = st.session_state["_panel_cache"].get(("_view_html", stem), "")
                if _view_html:
                    st.markdown(_view_html, unsafe_allow_html=True)

            if mode == "Research" and metric_tab is not None:
                with metric_tab:
                    if gt_bool is None:
                        if not gt_dir_input.strip():
                            st.info("请在侧边栏指定服务端 Ground Truth 文件夹路径。", icon="ℹ️")
                        else:
                            st.warning("未找到对应 GT 文件，无法计算该图指标。", icon="⚠️")
                    else:
                        _mhtml = st.session_state["_panel_cache"].get(("_metrics_html", stem))
                        if _mhtml:
                            st.markdown(_mhtml, unsafe_allow_html=True)
                        else:
                            st.info("暂无指标数据。", icon="ℹ️")

    if _n_pages > 1:
        _bc1, _bc2, _bc3 = st.columns([1, 4, 1])
        with _bc1:
            if st.button("◀ Prev", disabled=st.session_state.page_idx == 0, use_container_width=True, key="pg_prev_b"):
                st.session_state.page_idx -= 1
                st.rerun()
        with _bc2:
            st.markdown(
                f"<div style='text-align:center;padding:6px 0;"
                f"color:#6b7280;font-size:0.9rem;'>"
                f"Page {st.session_state.page_idx + 1} / {_n_pages}"
                f" &nbsp;·&nbsp; {len(filtered_stems)} images total</div>",
                unsafe_allow_html=True,
            )
        with _bc3:
            if st.button(
                "Next ▶", disabled=st.session_state.page_idx == _n_pages - 1, use_container_width=True, key="pg_next_b"
            ):
                st.session_state.page_idx += 1
                st.rerun()

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
                    avg_metrics[cfg_name] = {m: float(np.mean([ml[m] for ml in mlist])) for m in ALL_METRICS}

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

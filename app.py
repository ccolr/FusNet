"""
app.py  ──  FusNet Web Demo v2（远程推理客户端）
================================================
模式：
  • Default  — 多模型预测，每张图 N 组 4-panel（原图 / 预测掩码 / 叠加 / GradCAM）
  • Research — 指定 GT 文件夹，6-panel + 逐图指标表 + 总体对比表与柱状图

启动：
    streamlit run app.py

本地需先做 SSH 端口转发：
    ssh -L 8000:localhost:8000 user@remote-host
"""

import base64
import io
import os
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

  /* ── paper-style table (booktabs) ── */
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

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── 常量 ─────────────────────────────────────────────────────────────────────
COLOR_PRED  = (220, 30,  30)   # red overlay for predictions
COLOR_GT    = (30,  160, 30)   # green overlay for GT
IMAGE_EXTS  = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
ALL_METRICS = ["Precision", "Recall", "F1", "IoU", "mIoU"]


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


def read_gt_mask(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix.lower() in (".tif", ".tiff"):
            import rasterio
            with rasterio.open(str(path)) as src:
                arr = src.read(1)
        else:
            arr = np.array(Image.open(str(path)).convert("L"))
        return arr > 0 if arr.max() <= 1 else arr > 127
    except Exception:
        return None


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


# ─── 模糊 GT 文件匹配 ────────────────────────────────────────────────────────
def _lcs_len(a: str, b: str) -> int:
    """最长公共子串长度（大小写不敏感）。"""
    a, b = a.lower(), b.lower()
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        curr = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def find_gt_file(stem: str, gt_dir: Path) -> Optional[Path]:
    """
    在 gt_dir 中找与 stem 最相似的文件。
    最长公共子串 / min(len_a, len_b) >= 0.6 才认为匹配成功。
    """
    best_f, best_r = None, 0.0
    for f in gt_dir.iterdir():
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        lcs   = _lcs_len(stem, f.stem)
        ratio = lcs / max(min(len(stem), len(f.stem)), 1)
        if ratio > best_r:
            best_r, best_f = ratio, f
    return best_f if best_r >= 0.6 else None


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
    }


# ─── 论文级 HTML 表格（booktabs 风格） ───────────────────────────────────────
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
    return f'<table class="paper-table">{header}<tbody>{rows}</tbody></table>'


# ─── 论文级柱状图 ─────────────────────────────────────────────────────────────
def metrics_barchart_png(model_metrics: dict, metrics: list) -> bytes:
    models  = list(model_metrics.keys())
    n_m, nc = len(metrics), len(models)
    x       = np.arange(n_m)
    bw      = min(0.65 / max(nc, 1), 0.28)
    offsets = (np.arange(nc) - (nc - 1) / 2) * bw
    cmap    = plt.get_cmap("tab10").colors  # type: ignore

    fig, ax = plt.subplots(figsize=(max(7, n_m * 2.0), 4.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, (mn, off) in enumerate(zip(models, offsets)):
        vals = [model_metrics[mn].get(m, 0) for m in metrics]
        bars = ax.bar(x + off, vals, bw * 0.90,
                      label=mn, color=cmap[i % 10],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.007,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)

    all_vals = [model_metrics[mn].get(m, 0) for mn in models for m in metrics]
    ymax = min(1.22, max(all_vals, default=0) + 0.18)
    ax.set_ylim(0, ymax)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.55, color="#aaa")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(fontsize=8.5, framealpha=0.95, loc="upper right")
    plt.tight_layout(pad=1.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ─── API 调用（缓存：同图+同模型不重复请求；滑块变化只触发本地计算） ──────────
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


# ─── 侧边栏 ───────────────────────────────────────────────────────────────────
selected_metrics: list = ALL_METRICS[:]
gt_dir_input: str = ""
selected_configs: list = []

with st.sidebar:
    st.markdown("## 🌿 FusNet")
    st.caption("Bamboo Segmentation Demo")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Default", "Research"],
        horizontal=True,
        help="Default: 多模型预测  ·  Research: 含 GT 对比与指标",
    )
    st.divider()

    server_url = st.text_input(
        "Inference server URL",
        value="http://localhost:8000",
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
        selected_configs = st.multiselect(
            "Network configurations",
            all_names,
            default=avail_names[:1] if avail_names else [],
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
        help="像素判定为竹林的概率下限，调节不重新推理（本地实时计算）",
    )
    heatmap_alpha = st.slider("Heatmap blend α", 0.1, 0.9, 0.5,  0.05,
                              help="热力图与原图混合比例")
    overlay_alpha = st.slider("Mask overlay α",  0.1, 0.9, 0.45, 0.05,
                              help="掩码叠加层不透明度")

    if mode == "Research":
        st.divider()
        st.markdown("**Research settings**")
        gt_dir_input = st.text_input(
            "Ground Truth folder",
            placeholder="/path/to/labels",
            help="包含真实二值掩码的文件夹。文件名与原图进行模糊匹配（最长公共子串比≥0.6）。",
        )
        selected_metrics = st.multiselect(
            "Metrics to display",
            ALL_METRICS,
            default=ALL_METRICS,
            help="选择在表格和图表中展示的指标",
        )
        if not selected_metrics:
            selected_metrics = ALL_METRICS[:]


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

uploaded_files = st.file_uploader(
    "Upload images",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed",
    key=f"up_{st.session_state.uploader_key}",
)

if uploaded_files and st.button("Clear all", icon="🗑️"):
    st.session_state.uploader_key += 1
    st.rerun()

if not uploaded_files:
    st.info("👆 Upload at least one image to get started.", icon="ℹ️")
    st.stop()

if not health:
    st.error("推理服务器不可达，请先启动 `server.py` 并确认端口转发。", icon="🚨")
    st.stop()

if not selected_configs:
    st.error("请在侧边栏选择至少一个模型配置。", icon="🚨")
    st.stop()

# ── GT 目录验证（研究模式） ──────────────────────────────────────────────────
gt_dir: Optional[Path] = None
if mode == "Research" and gt_dir_input.strip():
    p = Path(gt_dir_input.strip())
    if p.exists() and p.is_dir():
        gt_dir = p
        st.success(f"GT folder: `{gt_dir}`", icon="📂")
    else:
        st.error(f"GT 文件夹不存在：`{p}`", icon="🚨")

# ─── 推理阶段 ─────────────────────────────────────────────────────────────────
# all_results[stem][config_name] = {"probs": ndarray, "cam": ndarray}
# all_results[stem]["_raw"] = uint8 RGB ndarray
all_results: dict[str, dict]                  = {}
all_gt:      dict[str, Optional[np.ndarray]]  = {}

n_total = len(uploaded_files) * len(selected_configs)
progress = st.progress(0, text="Sending to inference server…")
task_idx = 0

for uf in uploaded_files:
    raw_bytes = uf.read()
    image     = read_image_bytes(raw_bytes, uf.name)
    stem      = Path(uf.name).stem

    all_results[stem] = {"_raw": image}

    if mode == "Research" and gt_dir is not None:
        gp = find_gt_file(stem, gt_dir)
        all_gt[stem] = read_gt_mask(gp) if gp else None

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

valid_stems = [
    s for s in all_results
    if any(c in all_results[s] for c in selected_configs)
]
if not valid_stems:
    st.stop()

n_done = sum(1 for s in valid_stems for c in selected_configs if c in all_results[s])
st.success(f"Done — {len(valid_stems)} image(s), {n_done} inference(s) completed.", icon="✅")


# ─── 辅助：单 panel 展示 + 下载 ──────────────────────────────────────────────
def _panel(col, title: str, arr: np.ndarray, dl_key: str, mode_l: str = "RGB"):
    with col:
        st.markdown(f'<div class="img-card-title">{title}</div>', unsafe_allow_html=True)
        st.image(arr, use_container_width=True)
        st.download_button(
            "⬇ Download",
            pil_png_bytes(arr, mode=mode_l),
            f"{dl_key}.png",
            "image/png",
            key=f"dl_{dl_key}",
            use_container_width=True,
        )


# ─── 展示阶段 ─────────────────────────────────────────────────────────────────
summary_metrics: dict[str, list[dict]] = {c: [] for c in selected_configs}

for img_idx, stem in enumerate(valid_stems):
    if img_idx > 0:
        st.markdown('<hr class="sample-divider">', unsafe_allow_html=True)

    st.markdown(f"### {stem}")
    raw_image = all_results[stem]["_raw"]
    h, w      = raw_image.shape[:2]
    gt_bool   = all_gt.get(stem)  # None in Default mode or when GT not found

    if mode == "Research" and gt_dir is not None and gt_bool is None:
        st.warning(f"未找到 `{stem}` 对应的 GT 标签文件，仅展示预测结果。", icon="⚠️")

    # ── Tab 布局（研究模式每张图用 Tabs） ─────────────────────────────────────
    if mode == "Research":
        view_tab, metric_tab = st.tabs(["🖼  Prediction Views", "📊  Image Metrics"])
    else:
        view_tab   = st.container()
        metric_tab = None

    # ── 预测视图 ──────────────────────────────────────────────────────────────
    with view_tab:
        for ci, cfg_name in enumerate(selected_configs):
            if cfg_name not in all_results[stem]:
                continue

            probs     = all_results[stem][cfg_name]["probs"]
            cam       = all_results[stem][cfg_name]["cam"]
            mask_bool = resize_mask_bool(probs >= conf_threshold, h, w)
            mask_u8   = mask_bool.astype(np.uint8) * 255

            pred_blend = overlay_mask(raw_image, mask_bool, overlay_alpha, COLOR_PRED)
            heatmap    = apply_heatmap(raw_image, cam, heatmap_alpha)

            pfx = f"i{img_idx}c{ci}"
            if ci > 0:
                st.markdown("---")
            st.markdown(
                f'<div class="model-header">▸ {cfg_name}</div>',
                unsafe_allow_html=True,
            )

            use_6 = mode == "Research" and gt_bool is not None
            if use_6:
                gt_r     = resize_mask_bool(gt_bool, h, w)
                gt_u8    = gt_r.astype(np.uint8) * 255
                gt_blend = overlay_mask(raw_image, gt_r, overlay_alpha, COLOR_GT)

                c1, c2, c3, c4, c5, c6 = st.columns(6, gap="small")
                _panel(c1, "Original",     raw_image,  f"{pfx}_orig")
                _panel(c2, "GT Mask",      gt_u8,      f"{pfx}_gtmask",  "L")
                _panel(c3, "Pred Mask",    mask_u8,    f"{pfx}_pmask",   "L")
                _panel(c4, "GT Overlay",   gt_blend,   f"{pfx}_gtblend")
                _panel(c5, "Pred Overlay", pred_blend, f"{pfx}_pblend")
                _panel(c6, "GradCAM",      heatmap,    f"{pfx}_cam")
            else:
                c1, c2, c3, c4 = st.columns(4, gap="medium")
                _panel(c1, "Original",     raw_image,  f"{pfx}_orig")
                _panel(c2, "Pred Mask",    mask_u8,    f"{pfx}_pmask",   "L")
                _panel(c3, "Pred Overlay", pred_blend, f"{pfx}_pblend")
                _panel(c4, "GradCAM",      heatmap,    f"{pfx}_cam")

    # ── 逐图指标 Tab（研究模式） ──────────────────────────────────────────────
    if mode == "Research" and metric_tab is not None:
        with metric_tab:
            if gt_bool is None:
                if gt_dir is None:
                    st.info("请在侧边栏指定 Ground Truth 文件夹路径。", icon="ℹ️")
                else:
                    st.warning("未找到对应 GT 文件，无法计算该图指标。", icon="⚠️")
            else:
                gt_r = resize_mask_bool(gt_bool, h, w)
                img_model_metrics: dict[str, dict] = {}

                for cfg_name in selected_configs:
                    if cfg_name not in all_results[stem]:
                        continue
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


# ─── 研究模式：总体汇总 ───────────────────────────────────────────────────────
if mode == "Research" and any(len(v) > 0 for v in summary_metrics.values()):
    st.markdown('<hr class="sample-divider">', unsafe_allow_html=True)
    st.markdown("## 📊  Overall Model Comparison")

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
            st.image(chart_bytes, use_container_width=False)

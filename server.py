"""
server.py  ──  FusNet 推理后端
================================
在远端服务器上运行：
    python server.py [--host 0.0.0.0] [--port 8000]

本地通过 SSH 端口转发访问：
    ssh -L 8000:localhost:8000 user@remote-host

依赖：
    pip install fastapi uvicorn[standard] torch torchvision
                rasterio pillow numpy einops mamba_ssm
"""

import argparse
import base64
import io
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

# ─── 模型配置 ─────────────────────────────────────────────────────────────────
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

MEAN = [55.7578 / 255, 67.4502 / 255, 58.6568 / 255]
STD  = [37.5201 / 255, 34.2345 / 255, 30.3007 / 255]

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="FusNet Inference Server", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model_cache: dict[str, object] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(config_name: str):
    if config_name in _model_cache:
        return _model_cache[config_name]

    from model.FusNet import FusNet

    cfg   = MODEL_CONFIGS[config_name]
    model = FusNet(active_branches=cfg["active_branches"])
    state = torch.load(cfg["weight_path"], map_location=_device, weights_only=True)
    model.load_state_dict(state)
    model.to(_device).eval()
    _model_cache[config_name] = model
    print(f"[server] Loaded '{config_name}' on {_device}")
    return model


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def _read_image(data: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith((".tif", ".tiff")):
        import rasterio
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


def _preprocess(image: np.ndarray, size: int = 224) -> torch.Tensor:
    pil = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return tfm(arr).unsqueeze(0)


class _GradCAM:
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
        logits[:, 1, :, :].mean().backward()
        alpha = self.gradient.mean(dim=(2, 3), keepdim=True)
        cam   = F.relu((alpha * self.feature).sum(dim=1, keepdim=True))
        cam   = cam.squeeze().cpu().numpy()
        lo, hi = cam.min(), cam.max()
        return ((cam - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _run_inference(model, image: np.ndarray):
    """Returns (probs_np, cam) where probs_np is float32 [0,1] at original resolution."""
    orig_h, orig_w = image.shape[:2]
    tensor = _preprocess(image).to(_device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits   = F.interpolate(logits, size=(orig_h, orig_w),
                                 mode="bilinear", align_corners=False)
        probs_t  = torch.softmax(logits, dim=1)[:, 1, :, :]
        probs_np = probs_t.squeeze(0).cpu().numpy().astype(np.float32)

    target_layer = model.decoder[-1].conv
    gradcam      = _GradCAM(model, target_layer)
    tensor_grad  = _preprocess(image).to(_device)
    tensor_grad.requires_grad_(True)
    cam = gradcam(tensor_grad)
    gradcam.remove()

    return probs_np, cam


# ─── API 端点 ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(_device)}


@app.get("/configs")
def get_configs():
    return {
        name: {
            "active_branches":  cfg["active_branches"],
            "weight_available": os.path.exists(cfg["weight_path"]),
        }
        for name, cfg in MODEL_CONFIGS.items()
    }


@app.post("/infer")
async def infer(
    file:        UploadFile = File(...),
    config_name: str        = Form(...),
):
    if config_name not in MODEL_CONFIGS:
        raise HTTPException(400, f"Unknown config: {config_name!r}")

    cfg = MODEL_CONFIGS[config_name]
    if not os.path.exists(cfg["weight_path"]):
        raise HTTPException(404, f"Weight file not found: {cfg['weight_path']}")

    data  = await file.read()
    image = _read_image(data, file.filename or "image.png")

    try:
        model        = _load_model(config_name)
        probs_np, cam = _run_inference(model, image)
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    # probs → base64 raw float32 bytes + shape (client applies threshold locally)
    probs_b64 = base64.b64encode(probs_np.tobytes()).decode()
    cam_b64   = base64.b64encode(cam.tobytes()).decode()

    return {
        "probs":       probs_b64,
        "probs_shape": list(probs_np.shape),
        "cam":         cam_b64,
        "cam_shape":   list(cam.shape),
    }


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FusNet Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument(
        "--preload-all",
        action="store_true",
        help="启动时预加载所有可用权重（避免首次请求延迟）",
    )
    args = parser.parse_args()

    if args.preload_all:
        for name, cfg in MODEL_CONFIGS.items():
            if os.path.exists(cfg["weight_path"]):
                _load_model(name)

    uvicorn.run(app, host=args.host, port=args.port)

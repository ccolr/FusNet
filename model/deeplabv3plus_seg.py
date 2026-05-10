"""
Standalone DeepLabV3+ wrapper for binary / multi-class segmentation.

Wraps the DeepLabV3+ implementation in the top-level ``DeeplabV3`` package
without modifying it, so the model can be used as a drop-in replacement
for FusNet / SwinUnetSeg in comparison experiments.

Expected input : (B, 3, 224, 224)  float32, already normalised
Output         : (B, num_classes, 224, 224)  raw logits
"""

import os
import sys

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from DeeplabV3 import (
    deeplabv3plus_resnet50,
    deeplabv3plus_resnet101,
    deeplabv3plus_mobilenet,
)


_ARCH_FACTORY = {
    "deeplabv3plus_resnet50":   deeplabv3plus_resnet50,
    "deeplabv3plus_resnet101":  deeplabv3plus_resnet101,
    "deeplabv3plus_mobilenet":  deeplabv3plus_mobilenet,
}


class DeepLabV3PlusSeg(nn.Module):
    """DeepLabV3+ wrapper for segmentation.

    Args:
        arch                : architecture name; one of
                              ``deeplabv3plus_resnet50``, ``deeplabv3plus_resnet101``,
                              ``deeplabv3plus_mobilenet``.
        num_classes         : number of output channels (2 for binary segmentation,
                              consistent with FusNet default).
        output_stride       : 8 or 16. 16 is faster / less VRAM.
        pretrained_backbone : load ImageNet-pretrained backbone weights via
                              torchvision's URL hub.
        pretrained_path     : optional path to a full DeepLabV3+ checkpoint.
                              Loaded with ``strict=False`` so the final classifier
                              head is re-initialised when ``num_classes`` differs.
    """

    def __init__(
        self,
        arch: str = "deeplabv3plus_resnet50",
        num_classes: int = 2,
        output_stride: int = 16,
        pretrained_backbone: bool = True,
        pretrained_path: str = None,
    ):
        super().__init__()
        if arch not in _ARCH_FACTORY:
            raise ValueError(
                f"Unknown DeepLabV3+ arch '{arch}'. Choices: {list(_ARCH_FACTORY)}"
            )
        factory = _ARCH_FACTORY[arch]
        self.model = factory(
            num_classes=num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
        self.arch = arch

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _load_pretrained(self, pretrained_path: str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model", ckpt)))

        own = self.model.state_dict()
        filtered = {
            k: v
            for k, v in state.items()
            if k in own and own[k].shape == v.shape
        }
        msg = self.model.load_state_dict(filtered, strict=False)
        print(f"[DeepLabV3PlusSeg] loaded pretrained weights from '{pretrained_path}'")
        print(
            f"[DeepLabV3PlusSeg] loaded={len(filtered)}  "
            f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}"
        )

    def encoder_parameters(self):
        yield from self.model.backbone.parameters()

    def decoder_parameters(self):
        yield from self.model.classifier.parameters()

"""
Standalone Swin-Unet wrapper for binary / multi-class segmentation.

No YACS config required — all hyper-parameters are explicit keyword
arguments so the module can be imported directly in comparison scripts.

Expected input : (B, 3, 224, 224)  float32, already normalised
Output         : (B, num_classes, 224, 224)  raw logits
"""

import copy

import torch
import torch.nn as nn

# SwinTransformerSys lives one level up, in networks/
# This import works when the project root is on sys.path (normal CWD usage).
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUnetSeg(nn.Module):
    """Swin-Unet for segmentation, with optional ImageNet pretrained encoder init.

    Default hyper-parameters match the 'swin_tiny_patch4_window7_224_lite' config
    (depths=[2,2,2,2], decoder=[2,2,2,1], embed_dim=96).

    Args:
        img_size        : input spatial size (must be divisible by patch_size*32).
        num_classes     : number of output channels (2 for binary segmentation,
                          consistent with FusNet default).
        embed_dim       : base embedding dimension.
        depths          : encoder stage depths.
        depths_decoder  : decoder stage depths (reversed order).
        num_heads       : attention heads per encoder stage.
        window_size     : local attention window size.
        drop_path_rate  : stochastic depth rate.
        pretrained_path : path to an ImageNet Swin-T checkpoint (.pth).
                          When supplied the encoder (and mirrored decoder)
                          weights are initialised from it.
    """

    def __init__(
        self,
        img_size: int = 224,
        num_classes: int = 2,
        embed_dim: int = 96,
        depths=(2, 2, 2, 2),
        depths_decoder=(2, 2, 2, 1),
        num_heads=(3, 6, 12, 24),
        window_size: int = 7,
        drop_path_rate: float = 0.2,
        pretrained_path: str = None,
    ):
        super().__init__()

        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            num_heads=list(num_heads),
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=drop_path_rate,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first",
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swin_unet(x)

    # ------------------------------------------------------------------
    def _load_pretrained(self, pretrained_path: str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(pretrained_path, map_location=device)
        pretrained_dict = ckpt.get("model", ckpt)

        model_dict = self.swin_unet.state_dict()

        # Mirror encoder weights into the corresponding decoder stages
        # layers.0 → layers_up.3,  layers.1 → layers_up.2, etc.
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k and k[7].isdigit():
                mirrored = "layers_up." + str(3 - int(k[7])) + k[8:]
                full_dict[mirrored] = v

        # Keep only keys that exist in the model with matching shapes
        filtered = {
            k: v
            for k, v in full_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }

        msg = self.swin_unet.load_state_dict(filtered, strict=False)
        print(f"[SwinUnetSeg] loaded pretrained weights from '{pretrained_path}'")
        print(f"[SwinUnetSeg] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")

    # ------------------------------------------------------------------
    # Convenience: expose encoder / decoder parameter groups for
    # differential learning-rate optimisers.
    def encoder_parameters(self):
        # patch_embed + (optional) absolute_pos_embed + encoder layers + encoder norm
        yield from self.swin_unet.patch_embed.parameters()
        if hasattr(self.swin_unet, "absolute_pos_embed"):
            yield self.swin_unet.absolute_pos_embed
        yield from self.swin_unet.layers.parameters()
        if hasattr(self.swin_unet, "norm") and self.swin_unet.norm is not None:
            yield from self.swin_unet.norm.parameters()

    def decoder_parameters(self):
        dec_modules = [
            self.swin_unet.layers_up,
            self.swin_unet.concat_back_dim,
            self.swin_unet.norm_up,
            self.swin_unet.up,
            self.swin_unet.output,
        ]
        for m in dec_modules:
            yield from m.parameters()

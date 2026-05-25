# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
MMDetection-compatible DINOv3 ViT backbone.

This module provides a backbone wrapper that extracts multi-level features from
DINOv3 Vision Transformer intermediate layers, producing outputs suitable for
MMDetection's FPN neck or other detection heads.

Usage (standalone):
    backbone = DinoVisionTransformerBackbone(model_name="dinov3_vitb16")
    feats = backbone(torch.randn(2, 3, 512, 512))
    # feats: tuple of 4 tensors, each (2, 768, 32, 32)

Usage (MMDetection config):
    backbone=dict(
        type='DinoVisionTransformerBackbone',
        model_name='dinov3_vitb16',
        frozen_stages=-1,
        out_indices=(0, 1, 2, 3),
    )
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from dinov3.eval.detection.models.utils import LayerNorm2D


# ------------------------------------------------------------------------
# Model registry — maps model_name -> (embed_dim, n_blocks, has_variable_dims)
# ------------------------------------------------------------------------
_MODEL_SPECS: Dict[str, Dict] = {
    "dinov3_vits16":       {"embed_dim": 384,  "n_blocks": 12, "patch_size": 16},
    "dinov3_vits16plus":   {"embed_dim": 384,  "n_blocks": 12, "patch_size": 16},
    "dinov3_vitb16":       {"embed_dim": 768,  "n_blocks": 12, "patch_size": 16},
    "dinov3_vitl16":       {"embed_dim": 1024, "n_blocks": 24, "patch_size": 16},
    "dinov3_vitl16plus":   {"embed_dim": 1024, "n_blocks": 24, "patch_size": 16},
    "dinov3_vith16plus":   {"embed_dim": 1280, "n_blocks": 32, "patch_size": 16},
    "dinov3_vit7b16":      {"embed_dim": 4096, "n_blocks": 40, "patch_size": 16},
}

# Hub function name -> model_name (all use same naming convention)
_HUB_FUNCTIONS = [
    "dinov3_vits16", "dinov3_vits16plus",
    "dinov3_vitb16",
    "dinov3_vitl16", "dinov3_vitl16plus",
    "dinov3_vith16plus",
    "dinov3_vit7b16",
]


def _get_default_layers_to_use(n_blocks: int, num_levels: int = 4) -> List[int]:
    """Return default layer indices evenly spaced across the network.

    Follows the existing convention: last block of each quarter.
    """
    return [m * n_blocks // num_levels - 1 for m in range(1, num_levels + 1)]


class DinoVisionTransformerBackbone(nn.Module):
    """MMDetection-compatible DINOv3 ViT backbone.

    Extracts patch-token features from selected intermediate transformer blocks,
    returning multi-level feature maps at the same spatial resolution (stride=patch_size).
    MMDetection's FPN neck is expected to build the multi-scale pyramid from these.

    Key design:
    - Input:  (B, 3, H, W) tensor  (H, W must be divisible by patch_size)
    - Output: tuple of (B, embed_dim, H//patch, W//patch) per selected layer
    - Pretrained weights loaded via DINOv3 hub API in ``init_weights()``.

    Args:
        model_name: Hub backbone name (e.g. ``"dinov3_vitb16"``).
        pretrained: Whether to load pretrained weights.
        layers_to_use: Indices of ViT blocks whose outputs to return.
            Default: evenly-spaced across all blocks (4 levels).
        out_indices: Which of the extracted layers to actually output.
            Default: all layers.
        use_layernorm: Whether to apply LayerNorm2D to each extracted output.
        out_channels: If not None, project concatenated features to this dim via 1x1 conv.
            When set, output becomes a single-level feature map (not tuple of multi-level).
            This is useful for DETR-style heads that expect a single feature level.
        frozen_stages: -1 = freeze all backbone params; 0 = train all; (no partial freeze for ViT).
        init_cfg: MMDetection-style init config (stored for compatibility, handled in init_weights).
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained: bool = True,
        layers_to_use: Optional[List[int]] = None,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        use_layernorm: bool = True,
        out_channels: Optional[int] = None,
        frozen_stages: int = -1,
        init_cfg: Optional[dict] = None,
    ):
        super().__init__()

        if model_name not in _MODEL_SPECS:
            raise ValueError(
                f"Unknown model_name: {model_name}. Available: {list(_MODEL_SPECS.keys())}"
            )

        spec = _MODEL_SPECS[model_name]
        self.model_name = model_name
        self.embed_dim: int = spec["embed_dim"]
        self.n_blocks: int = spec["n_blocks"]
        self.patch_size: int = spec["patch_size"]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # Layer selection
        if layers_to_use is None:
            layers_to_use = _get_default_layers_to_use(self.n_blocks)
        self.layers_to_use = layers_to_use

        # Build the ViT backbone via hub API
        self._build_vit()

        # Optional per-level normalization
        if use_layernorm:
            n_selected = len(layers_to_use)
            embed_dims = getattr(
                self.backbone, "embed_dims",
                [self.embed_dim] * self.n_blocks,
            )
            selected_dims = [embed_dims[i] for i in layers_to_use]
            self.layer_norms = nn.ModuleList(
                [LayerNorm2D(dim) for dim in selected_dims]
            )
        else:
            self.layer_norms = None

        # Optional output projection (for DETR-style single-level output)
        self.out_proj = None
        if out_channels is not None:
            concat_dim = self.embed_dim * len(layers_to_use)
            self.out_proj = nn.Conv2d(concat_dim, out_channels, kernel_size=1)
            self._out_channels_val = [out_channels]
        else:
            self._out_channels_val = [self.embed_dim] * len(out_indices)

        # Apply freezing
        if frozen_stages == -1:
            self._freeze_backbone()

    def _build_vit(self):
        """Build the DINOv3 ViT via hub API."""
        import dinov3.hub.backbones as hub_backbones

        backbone_fn = getattr(hub_backbones, self.model_name, None)
        if backbone_fn is None:
            raise ValueError(
                f"Hub function '{self.model_name}' not found in dinov3.hub.backbones"
            )
        self.backbone = backbone_fn(pretrained=self.pretrained)

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def init_weights(self):
        """Initialize weights (load pretrained if requested).

        Called by MMDetection's model initialization flow.
        If pretrained weights were already loaded in _build_vit, this is a no-op.
        If the backbone was built without pretrained=True, this reloads weights.
        """
        if self.pretrained and self.init_cfg is None:
            return  # already loaded in _build_vit

        if self.init_cfg is not None:
            # MMDetection init_cfg flow: load from checkpoint path
            checkpoint = self.init_cfg.get("checkpoint", None)
            if checkpoint is not None and isinstance(checkpoint, str):
                state_dict = torch.load(checkpoint, map_location="cpu")
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # Handle MMDetection checkpoint prefix
                state_dict = {
                    k.replace("backbone.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("backbone.")
                } or state_dict
                missing, unexpected = self.backbone.load_state_dict(
                    state_dict, strict=False
                )
                if missing:
                    import logging
                    logger = logging.getLogger("dinov3")
                    logger.warning(f"Missing keys when loading backbone: {missing[:5]}...")

    @property
    def out_channels(self) -> List[int]:
        """Return output channel dimensions (for MMDetection FPN compatibility)."""
        return self._out_channels_val

    @property
    def strides(self) -> List[int]:
        """Return strides of output feature maps."""
        if self.out_proj is not None:
            return [self.patch_size]
        return [self.patch_size] * len(self.out_indices)

    def train(self, mode: bool = True):
        """Override to handle frozen backbone."""
        super().train(mode)
        if self.frozen_stages == -1:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 3, H, W). H and W must be divisible by patch_size.

        Returns:
            Tuple of feature maps. If out_proj is set, returns single-element tuple
            of (B, out_channels, H//patch, W//patch).
            Otherwise returns tuple of (B, embed_dim, H//patch, W//patch), one per
            selected layer filtered by out_indices.
        """
        # get_intermediate_layers handles:
        #   patch embedding → prepend CLS + storage tokens →
        #   run transformer blocks → strip CLS/register tokens →
        #   apply norm → reshape to (B, C, H_patch, W_patch)
        features = self.backbone.get_intermediate_layers(
            x,
            n=self.layers_to_use,
            reshape=True,
            norm=True,
            return_class_token=False,
            return_extra_tokens=False,
        )

        # features is tuple of (B, embed_dim, H_patch, W_patch), one per selected layer

        if self.layer_norms is not None:
            features = tuple(
                ln(f).contiguous() for ln, f in zip(self.layer_norms, features)
            )

        if self.out_proj is not None:
            concat = torch.cat(features, dim=1)
            return (self.out_proj(concat),)

        return tuple(features[i] for i in self.out_indices)


# ------------------------------------------------------------------------
# MMDetection registry registration
# ------------------------------------------------------------------------
def register_mm_backbone() -> bool:
    """Register DinoVisionTransformerBackbone with MMDetection's model registry.

    Call this before creating the MMDetection Runner so that the config parser
    can resolve ``type='DinoVisionTransformerBackbone'``.

    Returns True if registration succeeded, False if MMDetection is not installed.
    """
    try:
        from mmengine.registry import MODELS
        MODELS.register_module(
            name="DinoVisionTransformerBackbone",
            module=DinoVisionTransformerBackbone,
            force=True,
        )
        return True
    except ImportError:
        return False

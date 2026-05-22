"""DINO V3 with OLMoEarth integration for stage 2 training.

Composes SSLMetaArch (DINO V3 self-supervised training) with a frozen
OLMoEarth model (inference-only).

Pipeline per training step:
    1. SSLMetaArch.forward_backward runs the normal DINO/iBOT step (with its
       own backward).
    2. The frozen OLMoEarth encoder is loaded for evaluation / logging.
"""

import logging

import torch
from torch import nn

from dinov3.train.olmoearth_inference import load_olmoearth_model
from dinov3.train.ssl_meta_arch import SSLMetaArch

logger = logging.getLogger("dinov3")


class DINOv3WithOLMoEarth(nn.Module):
    """DINO V3 stage 2 training with OLMoEarth inference.

    Wraps SSLMetaArch for DINO V3 training and adds a frozen OLMoEarth
    model for multi-modal embedding extraction.

    Config (cfg.olmoearth):
        enabled (bool): Master switch.
        model_id_or_path (str): ModelID string or local path.
        patch_size (int): Patch size for OLMoEarth encoder.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dino_model = SSLMetaArch(cfg)

        # OLMoEarth config
        olmoe_cfg = cfg.olmoearth
        self.olmoearth_model_id_or_path = olmoe_cfg.model_id_or_path
        self.olmoearth_patch_size = olmoe_cfg.patch_size
        self.olmoearth_missing_value = getattr(olmoe_cfg, "missing_value", -99999)
        self.olmoearth_model = None  # Loaded in init_weights after FSDP setup

        logger.info(
            f"OLMoEarth integration enabled: model={self.olmoearth_model_id_or_path}, "
            f"patch_size={self.olmoearth_patch_size}"
        )

    def init_weights(self):
        self.dino_model.init_weights()

        # Load OLMoEarth model after FSDP has moved DINO models to CUDA
        logger.info(f"Loading OLMoEarth model: {self.olmoearth_model_id_or_path}")
        self.olmoearth_model = load_olmoearth_model(
            self.olmoearth_model_id_or_path,
            device=torch.device("cuda"),
        )

    def prepare_for_distributed_training(self):
        # Only DINO models go through FSDP
        self.dino_model.prepare_for_distributed_training()

    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **kwargs
    ) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        # Pop OLMoEarth + HR-target fields that SSLMetaArch does not expect
        data.pop("olmoearth_modalities", None)
        data.pop("olmoearth_metadata", None)
        data.pop("hr_target_images", None)
        data.pop("hr_target_masks", None)
        data.pop("hr_target_start_times", None)
        data.pop("hr_data_start_time", None)

        return self.dino_model.forward_backward(
            data, teacher_temp=teacher_temp, iteration=iteration
        )

    # --- Delegated methods ---

    def train(self, mode=True):
        self.training = mode
        self.dino_model.train()
        if self.olmoearth_model is not None:
            self.olmoearth_model.eval()
        return self

    def update_ema(self, m):
        self.dino_model.update_ema(m)

    def get_params_groups(self):
        return self.dino_model.get_params_groups()

    def build_data_augmentation_dino(self, cfg):
        return self.dino_model.build_data_augmentation_dino(cfg)

    def build_data_augmentation_dino_h5(self, cfg):
        """Build multi-channel DINO augmentation for H5 pipeline."""
        from dinov3.data.augmentations import DataAugmentationDINOMultiChannel

        n_channels = getattr(cfg.crops, "n_channels", 3)
        nir_mean = getattr(cfg.crops, "nir_mean", 0.5)
        nir_std = getattr(cfg.crops, "nir_std", 0.25)
        debug_crop_dims = getattr(cfg.olmoearth, "debug_crop_dims", False)
        spatial_align = getattr(cfg.olmoearth, "spatial_align", 4)
        hr_h5_resolution_ratio = getattr(cfg.olmoearth, "hr_h5_resolution_ratio", 16)

        return DataAugmentationDINOMultiChannel(
            global_crops_scale=cfg.crops.global_crops_scale,
            local_crops_scale=cfg.crops.local_crops_scale,
            local_crops_number=cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            n_channels=n_channels,
            rgb_mean=cfg.crops.rgb_mean,
            rgb_std=cfg.crops.rgb_std,
            nir_mean=nir_mean,
            nir_std=nir_std,
            horizontal_flips=cfg.crops.horizontal_flips,
            share_color_jitter=cfg.crops.share_color_jitter,
            debug_crop_dims=debug_crop_dims,
            spatial_align=spatial_align,
            hr_h5_resolution_ratio=hr_h5_resolution_ratio,
        )

    @property
    def model_ema(self):
        return self.dino_model.model_ema

    @property
    def student(self):
        return self.dino_model.student

    @property
    def has_gram_teacher(self):
        return self.dino_model.has_gram_teacher

    def gram_load_ema_teacher(self):
        self.dino_model.gram_load_ema_teacher()

    def update_gram(self, m=0):
        self.dino_model.update_gram(m)

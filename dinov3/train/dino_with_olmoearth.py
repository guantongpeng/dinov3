"""DINO V3 with OLMoEarth integration for stage 2 training.

Composes SSLMetaArch (DINO V3 self-supervised training) with a frozen
OLMoEarth model (inference-only) and a fusion + multi-branch decoder stack
trained on top of both towers.

Pipeline per training step:
    1. SSLMetaArch.forward_backward runs the normal DINO/iBOT step (with its
       own backward). A forward hook on the student backbone captures the
       patch tokens produced during that forward.
    2. The frozen OLMoEarth encoder is run on each global crop's H5 data.
    3. Per crop: pool OLMoEarth tokens, cross-attend with the captured DINO
       tokens, refine via a small self-attention stack -> fused tokens.
    4. Two decoder branches consume the fused tokens:
       - H5 feature-reconstruction (predict per-modality OLMoEarth tokens,
         loss vs. detached OE tokens, optional cosine + MSE).
       - HR pixel-reconstruction conditioned on a time embedding (predicts
         the input HR crop AND every available hr_target image, MSE in
         ImageNet-normalized space, masked by hr_target_masks).
    5. The combined fusion loss is backward'd into the fusion module +
       decoders. Calling the FSDP-wrapped backbone a second time after SSL's
       backward produces non-finite activations, so the captured tokens are
       detached and the DINO backbone receives no fusion gradient in this
       revision.
"""

import logging
from typing import Any

import torch
from torch import Tensor, nn

from dinov3.train.olmoearth_inference import load_olmoearth_model, run_olmoearth_inference
from dinov3.train.ssl_meta_arch import SSLMetaArch

logger = logging.getLogger("dinov3")

# Default modality set, mirrored from H5OlmoEarthDataset's OLMOEARTH_MODALITIES.
# Kept here to avoid a circular import on the dataset module.
_DEFAULT_MODALITIES = ("sentinel2_l2a", "sentinel1", "landsat")


class DINOv3WithOLMoEarth(nn.Module):
    """DINO V3 stage 2 training with OLMoEarth inference.

    Wraps SSLMetaArch for DINO V3 training and adds a frozen OLMoEarth
    model for multi-modal embedding extraction. OLMoEarth embeddings are
    computed each training step and made available for future fusion loss.

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

        # --- Fusion config ---
        fusion_cfg = getattr(cfg, "fusion", None)
        self.fusion_enabled = bool(fusion_cfg is not None and getattr(fusion_cfg, "enabled", False))
        if self.fusion_enabled:
            self.fusion_n_cross_heads = int(getattr(fusion_cfg, "n_cross_heads", 6))
            self.fusion_n_self_blocks = int(getattr(fusion_cfg, "n_self_blocks", 2))
            self.hr_decoder_blocks = int(getattr(fusion_cfg, "hr_decoder_blocks", 2))
            self.feature_recon_loss_weight = float(getattr(fusion_cfg, "feature_recon_loss_weight", 1.0))
            self.hr_recon_loss_weight = float(getattr(fusion_cfg, "hr_recon_loss_weight", 1.0))
            self.feature_recon_use_cosine = bool(getattr(fusion_cfg, "feature_recon_use_cosine", True))
        else:
            self.fusion_n_cross_heads = 0
            self.fusion_n_self_blocks = 0
            self.hr_decoder_blocks = 0
            self.feature_recon_loss_weight = 0.0
            self.hr_recon_loss_weight = 0.0
            self.feature_recon_use_cosine = False

        # Pixel normalization stats for the HR recon loss. Stored as plain
        # attributes (not buffers) because FSDP/_apply on the wrapped model
        # appears to corrupt them in mixed precision. They are moved to CUDA
        # in init_weights().
        mean_list = list(cfg.crops.rgb_mean)
        std_list = list(cfg.crops.rgb_std)
        if int(getattr(cfg.crops, "n_channels", 3)) == 4:
            mean_list.append(float(getattr(cfg.crops, "nir_mean", 0.5)))
            std_list.append(float(getattr(cfg.crops, "nir_std", 0.25)))
        self._hr_pixel_mean_list = mean_list
        self._hr_pixel_std_list = std_list
        self.hr_pixel_mean = torch.tensor(mean_list, dtype=torch.float32)
        self.hr_pixel_std = torch.tensor(std_list, dtype=torch.float32)

        # Fusion submodules. Built eagerly so they exist BEFORE the optimizer
        # is constructed from get_params_groups().
        self.fusion_module: FusionModule | None = None
        self.olmoearth_pooler: OlmoEarthTokenPooler | None = None
        self.h5_recon_head: H5ReconHead | None = None
        self.hr_recon_head: HRReconHead | None = None
        self.time_embedding: TimeEmbedding | None = None
        self._fusion_built = False
        self._captured_global_patch_tokens: Tensor | None = None

        if self.fusion_enabled:
            mods = list(getattr(olmoe_cfg, "olmoearth_modalities", None) or _DEFAULT_MODALITIES)
            oe_embed_dim = int(getattr(olmoe_cfg, "embed_dim", 128))
            self._build_fusion_modules(cfg, modalities=mods, oe_embed_dim=oe_embed_dim)

        logger.info(
            f"OLMoEarth integration enabled: model={self.olmoearth_model_id_or_path}, "
            f"patch_size={self.olmoearth_patch_size}; fusion_enabled={self.fusion_enabled}"
        )

    def _apply(self, fn, recurse=True):
        # Move fusion submodules registered directly on self, then delegate
        # dino_model and olmoearth_model to their own bookkeeping. (Pixel-norm
        # mean/std are plain attributes, not buffers, so they are not touched
        # here — init_weights re-creates them on CUDA.)
        for name, child in self.named_children():
            if name in ("dino_model", "olmoearth_model"):
                continue
            if child is not None:
                child._apply(fn, recurse)
        self.dino_model._apply(fn, recurse)
        if self.olmoearth_model is not None:
            self.olmoearth_model._apply(fn)
        return self

    def init_weights(self):
        self.dino_model.init_weights()

        # Load OLMoEarth model after FSDP has moved DINO models to CUDA
        logger.info(f"Loading OLMoEarth model: {self.olmoearth_model_id_or_path}")
        self.olmoearth_model = load_olmoearth_model(
            self.olmoearth_model_id_or_path,
            device=torch.device("cuda"),
        )

        # Move fusion submodules to CUDA. FSDP's setup only moved dino_model;
        # fusion + decoders + pixel-norm tensors live on `self` and need to be
        # moved explicitly. We re-create mean/std tensors because going through
        # register_buffer + _apply ends up corrupting them under bf16 compute
        # precision.
        cuda = torch.device("cuda")
        self.hr_pixel_mean = torch.tensor(self._hr_pixel_mean_list, dtype=torch.float32, device=cuda)
        self.hr_pixel_std = torch.tensor(self._hr_pixel_std_list, dtype=torch.float32, device=cuda)
        for name, child in self.named_children():
            if name in ("dino_model", "olmoearth_model"):
                continue
            if child is not None:
                child.to(cuda)
        logger.info(
            f"Fusion modules moved to CUDA. "
            f"hr_pixel_mean={self.hr_pixel_mean.tolist()}, "
            f"hr_pixel_std={self.hr_pixel_std.tolist()}"
        )

        if self.fusion_enabled:
            self._register_backbone_capture_hook()

    def _register_backbone_capture_hook(self):
        """Capture DINO global-crop patch tokens from the FIRST student-backbone
        forward (the one inside SSLMetaArch) via a forward hook.

        Running the FSDP-wrapped backbone a second time after SSL has already
        backward'd produces non-finite activations, so we cannot do an
        independent forward for fusion. The hook stores a detached, cloned copy
        of the patch tokens after norm, which the fusion stack consumes (with
        no gradient back into the backbone).
        """
        def hook(module, args, kwargs, output):
            # SSLMetaArch calls backbone([global, local], masks=..., is_training=True)
            # → returns a list of dicts. The first dict is for global crops.
            if isinstance(output, list) and len(output) >= 1 and isinstance(output[0], dict):
                tokens = output[0].get("x_norm_patchtokens", None)
                if tokens is not None:
                    self._captured_global_patch_tokens = tokens.detach().clone()
            return output

        self.dino_model.student.backbone.register_forward_hook(hook, with_kwargs=True)

    def prepare_for_distributed_training(self):
        # Only DINO models go through FSDP
        self.dino_model.prepare_for_distributed_training()

    def forward_backward(
        self, data, *, teacher_temp, iteration=0, **kwargs
    ) -> tuple[Tensor, dict[str, float | Tensor]]:
        metrics_dict: dict[str, Any] = {}

        # Extract OLMoEarth + HR-target fields that SSLMetaArch does not expect
        olmoearth_modalities_list = data.pop("olmoearth_modalities", None)
        olmoearth_metadata_list = data.pop("olmoearth_metadata", None)
        hr_target_images = data.pop("hr_target_images", None)
        hr_target_masks = data.pop("hr_target_masks", None)
        hr_target_start_times = data.pop("hr_target_start_times", None)
        hr_data_start_time = data.pop("hr_data_start_time", None)

        # Keep a handle on global crops BEFORE delegating, because SSLMetaArch
        # consumes / mutates the data dict. Move to CUDA here (SSL itself moves
        # its own copy in forward_backward).
        collated_global_crops = data["collated_global_crops"].cuda(non_blocking=True)

        # ---- 1. Standard DINO V3 forward-backward (unchanged) ----
        total_loss, dino_metrics = self.dino_model.forward_backward(
            data, teacher_temp=teacher_temp, iteration=iteration
        )
        metrics_dict.update(dino_metrics)

        if not self.fusion_enabled or self.olmoearth_model is None:
            return total_loss, metrics_dict
        if olmoearth_modalities_list is None or olmoearth_metadata_list is None:
            return total_loss, metrics_dict

        # ---- 2. Fusion + decoder forward + backward ----
        fusion_loss = self._fusion_forward_backward(
            collated_global_crops=collated_global_crops,
            olmoearth_modalities_list=olmoearth_modalities_list,
            olmoearth_metadata_list=olmoearth_metadata_list,
            hr_target_images=hr_target_images,
            hr_target_masks=hr_target_masks,
            hr_target_start_times=hr_target_start_times,
            hr_data_start_time=hr_data_start_time,
            metrics_dict=metrics_dict,
        )

        # The fusion path already called .backward(); we only add its value
        # to total_loss so the trainer's logging shows the combined scalar.
        # SSLMetaArch returns a Python float that becomes a tensor when DINO
        # losses are accumulated; convert to a 0-dim tensor to satisfy the
        # trainer's total_loss.new_empty(...) call.
        if isinstance(total_loss, torch.Tensor):
            total_loss = total_loss + fusion_loss.detach().to(total_loss.device)
        else:
            total_loss = torch.as_tensor(
                float(total_loss) + float(fusion_loss.detach()),
                device=fusion_loss.device,
            )
        return total_loss, metrics_dict

    # ----- Fusion helpers -----

    def _build_fusion_modules(self, cfg, *, modalities: list[str], oe_embed_dim: int) -> None:
        """Build the fusion stack eagerly so submodules exist before the
        optimizer is constructed from get_params_groups()."""
        backbone = self.dino_model.student.backbone
        embed_dim = int(backbone.embed_dim)
        patch_size = int(backbone.patch_size)
        n_channels = int(self.hr_pixel_mean.shape[0])

        # Heuristic: reuse the student backbone's attention head count
        try:
            num_heads = int(backbone.blocks[0].attn.num_heads)
        except Exception:
            num_heads = max(1, embed_dim // 64)

        gcs = int(cfg.crops.global_crops_size)
        ratio = int(getattr(cfg.olmoearth, "hr_h5_resolution_ratio", 16))
        oe_patch = int(getattr(cfg.olmoearth, "patch_size", 2))
        # H5 spatial pixels = gcs/ratio, then OE patchifies again with oe_patch
        h5_side = max(1, gcs // ratio // oe_patch)

        self.fusion_module = FusionModule(
            dim=embed_dim, num_heads=num_heads, n_self_blocks=self.fusion_n_self_blocks,
        )
        self.olmoearth_pooler = OlmoEarthTokenPooler(
            dim=embed_dim, modalities=modalities, oe_embed_dim=oe_embed_dim,
        )
        self.h5_recon_head = H5ReconHead(
            dim=embed_dim, h5_grid=(h5_side, h5_side),
            modalities=modalities, oe_embed_dim=oe_embed_dim,
        )
        self.hr_recon_head = HRReconHead(
            dim=embed_dim, num_heads=num_heads, patch_size=patch_size,
            out_channels=n_channels, n_blocks=self.hr_decoder_blocks,
        )
        self.time_embedding = TimeEmbedding(dim=embed_dim)

        self._fusion_built = True
        logger.info(
            f"Fusion modules built: dim={embed_dim}, heads={num_heads}, "
            f"h5_grid=({h5_side},{h5_side}), patch={patch_size}, "
            f"channels={n_channels}, oe_dim={oe_embed_dim}, modalities={modalities}"
        )

    def _fusion_forward_backward(
        self,
        *,
        collated_global_crops: Tensor,
        olmoearth_modalities_list: list[dict[str, Tensor]],
        olmoearth_metadata_list: list[dict[str, Tensor]],
        hr_target_images: Tensor | None,
        hr_target_masks: Tensor | None,
        hr_target_start_times: list[list[str | None]] | None,
        hr_data_start_time: list[str | None] | None,
        metrics_dict: dict[str, Any],
    ) -> Tensor:
        device = collated_global_crops.device
        n_global_crops = len(olmoearth_modalities_list)
        assert collated_global_crops.shape[0] % n_global_crops == 0, (
            f"global crops total {collated_global_crops.shape[0]} not divisible by "
            f"n_global_crops={n_global_crops}"
        )
        B = collated_global_crops.shape[0] // n_global_crops

        # ---- 2.1 Use patch tokens captured by the forward hook during SSL.
        # We do NOT do a second backbone forward because under FSDP this
        # produces non-finite activations after the first backward.
        if self._captured_global_patch_tokens is None:
            logger.warning("Fusion: no captured patch tokens (hook missed) — skipping fusion step")
            return torch.zeros((), device=device)
        patch_tokens_all = self._captured_global_patch_tokens.to(torch.float32)
        self._captured_global_patch_tokens = None  # consume
        gc_fp32 = collated_global_crops.to(torch.float32)
        # shape: (n_global_crops * B, N_d, D_d)
        N_d = patch_tokens_all.shape[1]
        D_d = patch_tokens_all.shape[2]
        side = int(round(N_d ** 0.5))
        assert side * side == N_d, f"non-square DINO token grid: {N_d}"
        dino_grid = (side, side)

        # ---- 2.2 Run frozen OLMoEarth per crop, detach + sanitize outputs ----
        oe_embeddings_per_crop: list[dict[str, Tensor]] = []
        oe_valid_per_crop: list[dict[str, Tensor]] = []
        for modalities, metadata in zip(olmoearth_modalities_list, olmoearth_metadata_list):
            embeddings, per_sample_valid = run_olmoearth_inference(
                self.olmoearth_model,
                modalities,
                metadata,
                patch_size=self.olmoearth_patch_size,
                missing_value=self.olmoearth_missing_value,
            )
            # Zero out per-sample outputs of modalities whose inputs were
            # entirely missing — otherwise the encoder may have produced
            # very large activations from the -99999 sentinel.
            sanitized = {}
            for name, emb in embeddings.items():
                emb = emb.detach()
                if name in per_sample_valid:
                    v = per_sample_valid[name]
                    expand_shape = [-1] + [1] * (emb.ndim - 1)
                    keep = v.view(expand_shape).to(emb.dtype)
                    emb = emb * keep
                sanitized[name] = emb
            oe_embeddings_per_crop.append(sanitized)
            oe_valid_per_crop.append(per_sample_valid)

        # ---- 2.3 Per-crop fusion + losses ----
        feat_recon_terms = []
        hr_recon_input_terms = []
        hr_recon_target_terms = []

        patch_tokens_by_crop = patch_tokens_all.view(n_global_crops, B, N_d, D_d)

        for crop_idx in range(n_global_crops):
            dino_tokens = patch_tokens_by_crop[crop_idx].to(torch.float32)
            kv = self.olmoearth_pooler(
                {k: v.to(torch.float32) for k, v in oe_embeddings_per_crop[crop_idx].items()}
            )
            fused = self.fusion_module(dino_tokens, kv.detach())

            # --- H5 feature reconstruction ---
            preds = self.h5_recon_head(fused, dino_grid, oe_embeddings_per_crop[crop_idx])
            feat_loss = h5_feature_recon_loss(
                preds,
                {k: v.to(torch.float32) for k, v in oe_embeddings_per_crop[crop_idx].items()},
                valid_masks=oe_valid_per_crop[crop_idx],
                use_cosine=self.feature_recon_use_cosine,
            )
            feat_recon_terms.append(feat_loss)

            # --- HR reconstruction ---
            crop_slice = slice(crop_idx * B, (crop_idx + 1) * B)
            input_hr_normalized = gc_fp32[crop_slice]

            # The input HR crops are ImageNet-normalized; convert back to raw
            # [0,1] so hr_recon_loss can apply a single normalization path.
            mean_b = self.hr_pixel_mean.to(input_hr_normalized.dtype)
            std_b = self.hr_pixel_std.to(input_hr_normalized.dtype)
            input_hr_raw = input_hr_normalized * std_b[None, :, None, None] + mean_b[None, :, None, None]

            input_times = hr_data_start_time if hr_data_start_time is not None else [None] * B
            time_emb_input = self.time_embedding(list(input_times))
            pred_input = self.hr_recon_head(fused, time_emb_input, dino_grid)
            hr_recon_input_terms.append(hr_recon_loss(
                pred_input, input_hr_raw, mean_b, std_b,
            ))

            # Per-target reconstruction
            if (
                hr_target_images is not None
                and hr_target_masks is not None
                and hr_target_start_times is not None
                and hr_target_images.numel() > 0
            ):
                hr_target_images_dev = hr_target_images.to(device=device, dtype=torch.float32)
                hr_target_masks_dev = hr_target_masks.to(device=device)
                max_targets = hr_target_images_dev.shape[1]
                for k in range(max_targets):
                    valid_k = hr_target_masks_dev[:, k]
                    if not bool(valid_k.any()):
                        continue
                    times_k = [
                        (hr_target_start_times[b][k] if k < len(hr_target_start_times[b]) else None)
                        for b in range(B)
                    ]
                    time_emb_k = self.time_embedding(times_k)
                    pred_k = self.hr_recon_head(fused, time_emb_k, dino_grid)
                    tgt_k = hr_target_images_dev[:, k]
                    hr_recon_target_terms.append(hr_recon_loss(
                        pred_k, tgt_k, mean_b, std_b, valid=valid_k,
                    ))

        feat_recon_loss = torch.stack(feat_recon_terms).mean() if feat_recon_terms else torch.zeros((), device=device)
        hr_recon_input_loss = torch.stack(hr_recon_input_terms).mean() if hr_recon_input_terms else torch.zeros((), device=device)
        hr_recon_target_loss = torch.stack(hr_recon_target_terms).mean() if hr_recon_target_terms else torch.zeros((), device=device)

        fusion_loss = (
            self.feature_recon_loss_weight * feat_recon_loss
            + self.hr_recon_loss_weight * (hr_recon_input_loss + hr_recon_target_loss)
        )

        if fusion_loss.requires_grad:
            fusion_loss.backward()

        metrics_dict["fusion/feature_recon_loss"] = feat_recon_loss.detach()
        metrics_dict["fusion/hr_recon_loss_input"] = hr_recon_input_loss.detach()
        metrics_dict["fusion/hr_recon_loss_targets"] = hr_recon_target_loss.detach()
        metrics_dict["fusion/total"] = fusion_loss.detach()
        return fusion_loss

    # --- Delegated methods ---

    def train(self, mode=True):
        # Avoid super().train(mode) because SSLMetaArch.train() does not accept a mode argument.
        self.training = mode
        self.dino_model.train()
        if self.olmoearth_model is not None:
            self.olmoearth_model.eval()
        return self

    def update_ema(self, m):
        self.dino_model.update_ema(m)

    def get_params_groups(self):
        groups = list(self.dino_model.get_params_groups())
        if not self.fusion_enabled:
            return groups
        fusion_params = []
        for mod in (
            self.fusion_module, self.olmoearth_pooler, self.h5_recon_head,
            self.hr_recon_head, self.time_embedding,
        ):
            if mod is None:
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    fusion_params.append(p)
        if fusion_params:
            group = {
                "params": fusion_params,
                "is_last_layer": False,
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
                "name": "fusion",
            }
            if getattr(self.dino_model.cfg.optim, "multi_tensor_optim", False):
                group["foreach"] = True
                group["fused"] = True
            groups.append(group)
        return groups

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

"""OLMoEarth inference utilities for DINO V3 stage 2 training.

Provides functions to load a frozen OLMoEarth model and run inference
on batched modality data from H5OlmoEarthDataset.
"""

import logging

import torch
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

logger = logging.getLogger("dinov3")

MISSING_VALUE = -99999


def load_olmoearth_model(
    model_id_or_path: str,
    device: torch.device = torch.device("cuda"),
) -> torch.nn.Module:
    """Load a frozen OLMoEarth model for inference.

    Args:
        model_id_or_path: Either a ModelID string (e.g. "OlmoEarth-v1-Nano")
                          or a local path containing config.json + weights.pth.
        device: Target device.

    Returns:
        Frozen OLMoEarth model in eval mode on the specified device.
    """
    try:
        model_id = ModelID(model_id_or_path)
        logger.info(f"Loading OLMoEarth model from HuggingFace Hub: {model_id.value}")
        model = load_model_from_id(model_id)
    except ValueError:
        logger.info(f"Loading OLMoEarth model from local path: {model_id_or_path}")
        model = load_model_from_path(model_id_or_path)

    model.eval()
    model.requires_grad_(False)
    model.to(device)
    logger.info(f"OLMoEarth model loaded on {device}, params frozen")
    return model


def run_olmoearth_inference(
    model: torch.nn.Module,
    olmoearth_modalities: dict[str, torch.Tensor],
    olmoearth_metadata: dict[str, torch.Tensor],
    patch_size: int = 4,
    device: torch.device = torch.device("cuda"),
) -> dict[str, torch.Tensor]:
    """Run OLMoEarth encoder inference on batched modality data.

    Constructs a MaskedOlmoEarthSample from the batch tensors, runs the
    encoder, and returns per-modality embedding tensors.

    Args:
        model: Frozen OLMoEarth model.
        olmoearth_modalities: Dict of modality name -> [B, ...] tensor.
        olmoearth_metadata: Dict with 'timestamps' [B, T, 3] and 'latlon' [B, 2].
        patch_size: Patch size for the OLMoEarth encoder.
        device: Compute device.

    Returns:
        Dict mapping modality name to embedding tensor,
        e.g. {"sentinel2_l2a": [B, P_H, P_W, T, Band_Sets, D], ...}
    """

    batch_dict = {}
    mask_dict = {}

    # Timestamps (required by MaskedOlmoEarthSample)
    timestamps = olmoearth_metadata["timestamps"].to(device=device, dtype=torch.long)
    batch_dict["timestamps"] = timestamps

    # Latlon
    if "latlon" in olmoearth_metadata:
        latlon = olmoearth_metadata["latlon"].to(device=device, dtype=torch.float32)
        batch_dict["latlon"] = latlon
        latlon_mask_shape = list(latlon.shape) + [1]  # [B, 2] -> [B, 2, 1] for num_band_sets=1
        mask_dict["latlon_mask"] = torch.ones(
            latlon_mask_shape, dtype=torch.float32, device=device
        ) * MaskValue.ONLINE_ENCODER.value

    # Modality data and masks
    for modality_name, modality_tensor in olmoearth_modalities.items():
        mod_data = modality_tensor.to(device=device, dtype=torch.float32)
        batch_dict[modality_name] = mod_data

        mod_spec = Modality.get(modality_name)
        mask_shape = list(mod_data.shape)
        mask_shape[-1] = mod_spec.num_band_sets

        # Check per-batch-item: if entirely MISSING_VALUE, mark as MISSING
        mask = torch.full(
            mask_shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.float32, device=device
        )

        # Reshape to [B, -1] for efficient per-sample check
        flat = mod_data.reshape(mod_data.shape[0], -1)
        is_missing = (flat == MISSING_VALUE).all(dim=-1)  # [B]

        if is_missing.any():
            # Expand [B] -> [B, 1, 1, 1, 1] to broadcast over mask shape
            expand_shape = [-1] + [1] * (len(mask_shape) - 1)
            is_missing_expanded = is_missing.view(expand_shape)
            mask = mask.where(
                ~is_missing_expanded,
                torch.tensor(MaskValue.MISSING.value, dtype=torch.float32, device=device),
            )

        mask_dict[modality_name + "_mask"] = mask

    masked_sample = MaskedOlmoEarthSample(**batch_dict, **mask_dict)

    with torch.no_grad():
        output = model.encoder(masked_sample, fast_pass=False, patch_size=patch_size)
        tokens_and_masks = output["tokens_and_masks"]

    # Extract per-modality embeddings
    embeddings = {}
    for modality in tokens_and_masks.modalities:
        features = getattr(tokens_and_masks, modality)
        if features is not None:
            embeddings[modality] = features

    return embeddings

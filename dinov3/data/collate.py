# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import random

import torch

from .datasets.h5_olmoearth import _DEFAULT_SPATIAL_ALIGN


def collate_h5_olmoearth_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
    spatial_align=_DEFAULT_SPATIAL_ALIGN,
):
    """Collate function for H5OlmoEarthDataset with DINO global/local crops.

    samples_list[i] is a dict with:
        global_crops: list of (C, gH, gW) normalized float tensors
        local_crops: list of (C, lH, lW) normalized float tensors
        h5_olmoearth_crops: list of 2 dicts with "modalities" and "metadata"
        hr_data_start_time: str or None
        hr_target_images: list of (H, W, C) uint8 numpy arrays
        hr_target_start_times: list of str or None

    Returns a batch dict compatible with SSLMetaArch.forward_backward,
    plus per-global-crop OLMoEarth data.
    """
    import numpy as np

    # ---- 1. Collate DINO crops (global + local) ----
    n_global_crops = len(samples_list[0]["global_crops"])
    n_local_crops = len(samples_list[0]["local_crops"])

    collated_global_crops = torch.stack(
        [s["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops * B, C, gH, gW]
    collated_local_crops = torch.stack(
        [s["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
    )  # [n_local_crops * B, C, lH, lW]

    # ---- 2. Generate iBOT masks ----
    if local_batch_size is not None:
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    # ---- 3. Collate OLMoEarth modalities per global crop ----
    n_h5_crops = len(samples_list[0]["h5_olmoearth_crops"])

    olmoearth_modalities_by_crop = []
    olmoearth_metadata_by_crop = []

    for crop_idx in range(n_h5_crops):
        # Collate modalities for this crop
        modality_keys = list(samples_list[0]["h5_olmoearth_crops"][crop_idx]["modalities"].keys())
        crop_modalities = {}
        for key in modality_keys:
            tensors = [s["h5_olmoearth_crops"][crop_idx]["modalities"][key] for s in samples_list]
            shapes = [t.shape for t in tensors]
            if len(set(shapes)) == 1:
                crop_modalities[key] = torch.stack(tensors)
            else:
                ndim = len(shapes[0])
                if ndim == 4:
                    max_h = max(t.shape[0] for t in tensors)
                    max_w = max(t.shape[1] for t in tensors)
                    max_h = (max_h // spatial_align) * spatial_align
                    max_w = (max_w // spatial_align) * spatial_align
                    resized = []
                    for t in tensors:
                        if t.shape[0] != max_h or t.shape[1] != max_w:
                            h, w, tc_dim = t.shape[0], t.shape[1], t.shape[2] * t.shape[3]
                            t_flat = t.permute(2, 3, 0, 1).reshape(tc_dim, h, w).unsqueeze(0)
                            t_flat = torch.nn.functional.interpolate(
                                t_flat, size=(max_h, max_w), mode="bilinear", align_corners=False
                            ).squeeze(0)
                            t = t_flat.reshape(t.shape[2], t.shape[3], max_h, max_w).permute(2, 3, 0, 1)
                        resized.append(t)
                    crop_modalities[key] = torch.stack(resized)
                elif ndim == 3:
                    max_h = max(t.shape[0] for t in tensors)
                    max_w = max(t.shape[1] for t in tensors)
                    max_h = (max_h // spatial_align) * spatial_align
                    max_w = (max_w // spatial_align) * spatial_align
                    resized = []
                    for t in tensors:
                        if t.shape[0] != max_h or t.shape[1] != max_w:
                            t = t.permute(2, 0, 1).unsqueeze(0)
                            t = torch.nn.functional.interpolate(
                                t, size=(max_h, max_w), mode="bilinear", align_corners=False
                            ).squeeze(0).permute(1, 2, 0)
                        resized.append(t)
                    crop_modalities[key] = torch.stack(resized)
                else:
                    crop_modalities[key] = torch.stack(tensors)

        olmoearth_modalities_by_crop.append(crop_modalities)

        # Collate metadata for this crop
        crop_metadata = {}
        for key in samples_list[0]["h5_olmoearth_crops"][crop_idx]["metadata"]:
            crop_metadata[key] = torch.stack(
                [s["h5_olmoearth_crops"][crop_idx]["metadata"][key] for s in samples_list]
            )
        olmoearth_metadata_by_crop.append(crop_metadata)

    # ---- 4. Collate HR target images and start_times ----
    has_targets = any(s.get("hr_target_images") for s in samples_list)
    hr_target_images_batch = torch.empty(0)
    hr_target_masks_batch = torch.empty(0)
    hr_target_start_times = []
    hr_data_start_time = []

    if has_targets:
        target_size = collated_global_crops.shape[-1]
        all_target_tensors = []
        all_target_masks = []
        all_target_start_times = []

        for s in samples_list:
            target_arrays = s.get("hr_target_images", [])
            target_times = s.get("hr_target_start_times", [])
            hr_data_start_time.append(s.get("hr_data_start_time"))

            tensors = []
            for arr in target_arrays:
                tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
                if tensor.shape[1] != target_size or tensor.shape[2] != target_size:
                    tensor = torch.nn.functional.interpolate(
                        tensor.unsqueeze(0),
                        size=(target_size, target_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                tensors.append(tensor)

            all_target_tensors.append(tensors)
            all_target_start_times.append(target_times)

        max_targets = max(len(t) for t in all_target_tensors) if all_target_tensors else 0
        if max_targets > 0:
            n_channels = all_target_tensors[0][0].shape[0]
            padded_targets = []
            target_masks = []
            for tensors in all_target_tensors:
                n = len(tensors)
                if n < max_targets:
                    pad = torch.zeros(n_channels, target_size, target_size)
                    tensors = tensors + [pad] * (max_targets - n)
                padded_targets.append(torch.stack(tensors))
                mask = torch.zeros(max_targets, dtype=torch.bool)
                mask[:n] = True
                target_masks.append(mask)

            hr_target_images_batch = torch.stack(padded_targets)
            hr_target_masks_batch = torch.stack(target_masks)

        hr_target_start_times = all_target_start_times
    else:
        hr_data_start_time = [s.get("hr_data_start_time") for s in samples_list]

    batch = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "olmoearth_modalities": olmoearth_modalities_by_crop,
        "olmoearth_metadata": olmoearth_metadata_by_crop,
        "hr_target_images": hr_target_images_batch,
        "hr_target_masks": hr_target_masks_batch,
        "hr_target_start_times": hr_target_start_times,
        "hr_data_start_time": hr_data_start_time,
    }

    return batch


def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
):
    n_global_crops = len(samples_list[0][0]["global_crops"])
    n_local_crops = len(samples_list[0][0]["local_crops"])

    collated_global_crops = torch.stack(
        [s[0]["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
    )  # [n_global_crops, B, ...]
    collated_local_crops = torch.stack([s[0]["local_crops"][i] for i in range(n_local_crops) for s in samples_list])
    if "gram_teacher_crops" in samples_list[0][0]:
        collated_gram_teacher_crops = torch.stack(
            [s[0]["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )  # [n_global_crops, B, ...]
    else:
        collated_gram_teacher_crops = None

    if local_batch_size is not None:
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift:
            shift_x, shift_y = (
                random.randint(0, mask.shape[0] - 1),
                random.randint(0, mask.shape[1] - 1),
            )
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out


# def get_batch_subset(collated_data_batch, target_bs):
def get_batch_subset(collated_data_batch, divide_by):
    old_bs = collated_data_batch["collated_global_crops"].shape[0] // 2
    target_bs = (old_bs + divide_by - 1) // divide_by
    collated_global_crops = (
        collated_data_batch["collated_global_crops"].unflatten(0, (2, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )
    collated_local_crops = (
        collated_data_batch["collated_local_crops"].unflatten(0, (-1, old_bs)).narrow(1, 0, target_bs).flatten(0, 1)
    )

    masks_old_bs = collated_data_batch["collated_masks"].shape[0] // 2
    masks_target_bs = masks_old_bs // divide_by
    collated_masks = (
        collated_data_batch["collated_masks"]
        .unflatten(0, (2, masks_old_bs))
        .narrow(1, 0, masks_target_bs)
        .flatten(0, 1)
    )
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    while mask_indices_list.shape[0] == 0:
        _unbind = list(collated_data_batch["collated_masks"].unbind(0))
        random.shuffle(_unbind)
        _bind = torch.stack(_unbind, dim=0)
        collated_masks = _bind.unflatten(0, (2, masks_old_bs)).narrow(1, 0, masks_target_bs).flatten(0, 1)
        mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    upperbound = collated_data_batch["upperbound"]

    new_batch = {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

    if "global_batch_size" in collated_data_batch.keys():
        new_batch["global_batch_size"] = collated_data_batch["global_batch_size"] // divide_by

    return new_batch

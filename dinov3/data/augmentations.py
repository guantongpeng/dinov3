# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output


class DataAugmentationDINOMultiChannel(object):
    """DINO-style augmentation that supports N-channel input (3 for RGB, 4 for RGBNIR).

    Unlike DataAugmentationDINO, this class:
      - Operates on (C, H, W) float tensors directly (no PIL conversion)
      - Extracts random crop coordinates so that H5 modalities can be
        proportionally cropped to the same spatial region
      - Applies GaussianBlur on all channels, but color augmentation
        (jitter, grayscale, solarize) only on the first 3 (RGB) channels
      - Normalizes RGB and NIR channels separately
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        n_channels=3,
        rgb_mean=IMAGENET_DEFAULT_MEAN,
        rgb_std=IMAGENET_DEFAULT_STD,
        nir_mean=0.5,
        nir_std=0.25,
        horizontal_flips=True,
        share_color_jitter=False,
        debug_crop_dims=False,
        spatial_align=4,
        hr_h5_resolution_ratio=16,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.n_channels = n_channels
        self.horizontal_flips = horizontal_flips
        self.debug_crop_dims = debug_crop_dims
        self.spatial_align = spatial_align
        self.hr_h5_resolution_ratio = hr_h5_resolution_ratio

        # Build full mean/std vectors for Normalize
        mean_list = list(rgb_mean)
        std_list = list(rgb_std)
        if n_channels == 4:
            mean_list.append(nir_mean)
            std_list.append(nir_std)
        self.mean = mean_list
        self.std = std_list

        logger.info("###################################")
        logger.info("Using multi-channel DINO augmentation:")
        logger.info(f"  global_crops_scale: {global_crops_scale}")
        logger.info(f"  local_crops_scale: {local_crops_scale}")
        logger.info(f"  local_crops_number: {local_crops_number}")
        logger.info(f"  global_crops_size: {global_crops_size}")
        logger.info(f"  local_crops_size: {local_crops_size}")
        logger.info(f"  n_channels: {n_channels}")
        logger.info(f"  mean: {mean_list}")
        logger.info(f"  std: {std_list}")
        logger.info(f"  horizontal_flips: {horizontal_flips}")
        logger.info("###################################")

        # Color distortions (applied to RGB channels only)
        self.color_jittering = v2.Compose([
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8,
            ),
            v2.RandomGrayscale(p=0.2),
        ])

        # NIR jitter: brightness/contrast only (no saturation/hue/grayscale)
        self.nir_jittering = v2.Compose([
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4)],
                p=0.8,
            ),
        ])

        # GaussianBlur (applied to all channels)
        self.global_blur1 = GaussianBlur(p=1.0)
        self.global_blur2_blur = GaussianBlur(p=0.1)
        self.global_blur2_solarize = v2.RandomSolarize(threshold=0.5, p=0.2)
        self.local_blur = GaussianBlur(p=0.5)

        # Normalize (all channels)
        self.normalize = v2.Compose([
            v2.ToDtype(torch.float32, scale=False),
            make_normalize_transform(mean=mean_list, std=std_list),
        ])

    def _apply_geometric_crop(self, tensor, crop_size, scale):
        """Apply RandomResizedCrop, return (cropped_tensor, top_frac, left_frac, h_frac, w_frac).

        The fractional coordinates (top_frac, left_frac, h_frac, w_frac) are
        returned so that H5 modalities can be cropped to the same physical
        sub-region.  Since HR and H5 share the same physical extent (after
        Level 1 crop), applying the same fractions to each H5 modality
        selects the spatially corresponding region.
        """
        _, h, w = tensor.shape
        top, left, crop_h, crop_w = v2.RandomResizedCrop.get_params(
            tensor, scale=scale, ratio=[3.0 / 4.0, 4.0 / 3.0]
        )
        top_frac = top / h
        left_frac = left / w
        h_frac = crop_h / h
        w_frac = crop_w / w

        cropped = v2.functional.resized_crop(
            tensor, top, left, crop_h, crop_w,
            [crop_size, crop_size],
            interpolation=v2.InterpolationMode.BICUBIC,
        )
        return cropped, top_frac, left_frac, h_frac, w_frac

    def _apply_augmentation(self, crop, is_global, aug_idx):
        """Apply post-geometric augmentation: color jitter, blur, normalize.

        Order matches DataAugmentationDINO: color jitter -> blur -> normalize.
        Color jitter + grayscale on RGB only; GaussianBlur on all channels.
        """
        # Color jitter + grayscale on RGB channels only (before blur)
        # NIR gets brightness/contrast jitter only (no saturation/hue/grayscale)
        if self.n_channels == 4:
            rgb = crop[:3]
            nir = crop[3:4]
            rgb = self.color_jittering(rgb)
            nir = self.nir_jittering(nir)
            crop = torch.cat([rgb, nir], dim=0)
        else:
            crop = self.color_jittering(crop)

        # GaussianBlur on all channels; solarize on all channels
        if is_global:
            if aug_idx == 0:
                crop = self.global_blur1(crop)
            else:
                crop = self.global_blur2_blur(crop)
                crop = self.global_blur2_solarize(crop)
        else:
            crop = self.local_blur(crop)

        # Normalize all channels
        crop = self.normalize(crop)
        return crop

    def _apply_h5_aligned_geometric_crop(self, hr_tensor, crop_size, scale):
        """Generate crop params in H5 reference space, apply to HR with exact integer mapping.

        Uses ``self.hr_h5_resolution_ratio`` to map between H5 and HR pixel
        coordinates with exact integer multiplication (no float division),
        guaranteeing pixel-perfect alignment across all modalities.

        Returns:
            cropped_hr: (C, crop_size, crop_size) tensor
            h5_top, h5_left, h5_crop_h, h5_crop_w: crop params in H5 ref space
        """
        align = self.spatial_align
        ratio = self.hr_h5_resolution_ratio
        _, hr_h, hr_w = hr_tensor.shape

        # Derive H5 reference dims from HR dims and the known ratio
        h5_ref_h = hr_h // ratio
        h5_ref_w = hr_w // ratio

        # Sample crop params in H5 reference (coarsest) space
        dummy = torch.empty(1, h5_ref_h, h5_ref_w)
        h5_top, h5_left, h5_crop_h, h5_crop_w = v2.RandomResizedCrop.get_params(
            dummy, scale=scale, ratio=[3.0 / 4.0, 4.0 / 3.0]
        )

        # Align crop dimensions and start positions to spatial_align
        h5_crop_h = max(align, int(h5_crop_h) // align * align)
        h5_crop_w = max(align, int(h5_crop_w) // align * align)
        h5_top = int(h5_top) // align * align
        h5_left = int(h5_left) // align * align

        # Clamp to valid range
        h5_top = min(h5_top, h5_ref_h - h5_crop_h)
        h5_left = min(h5_left, h5_ref_w - h5_crop_w)
        h5_top = max(h5_top, 0)
        h5_left = max(h5_left, 0)

        # Map to HR pixel coordinates using exact integer multiplication
        hr_top = h5_top * ratio
        hr_left = h5_left * ratio
        hr_crop_h = h5_crop_h * ratio
        hr_crop_w = h5_crop_w * ratio

        # Apply resized crop on HR tensor
        cropped = v2.functional.resized_crop(
            hr_tensor, hr_top, hr_left, hr_crop_h, hr_crop_w,
            [crop_size, crop_size],
            interpolation=v2.InterpolationMode.BICUBIC,
        )
        return cropped, h5_top, h5_left, h5_crop_h, h5_crop_w

    def __call__(self, image_tensor):
        """
        Args:
            image_tensor: (C, H, W) float tensor in [0, 1] range.
                          This is the Level-1-cropped HR image — after Level 1,
                          HR and H5 cover the same physical region.

        Returns:
            dict with:
                "global_crops": list of (C, global_crops_size, global_crops_size)
                "local_crops": list of (C, local_crops_size, local_crops_size)
                "global_crop_flips": list of bool
                "h5_crop_params": list of (top, left, crop_h, crop_w) in H5 ref space
                "hr_h5_resolution_ratio": int — the configured ratio
        """
        _, hr_h, hr_w = image_tensor.shape
        ratio = self.hr_h5_resolution_ratio

        if self.debug_crop_dims:
            logger.info(f"[L2 input] HR=({hr_h},{hr_w}), ratio={ratio}")

        # Global crops (2)
        global_crops = []
        h5_crop_params = []
        global_crop_flips = []

        for aug_idx in range(2):
            cropped, h5_t, h5_l, h5_ch, h5_cw = self._apply_h5_aligned_geometric_crop(
                image_tensor, self.global_crops_size, self.global_crops_scale,
            )
            h5_crop_params.append((h5_t, h5_l, h5_ch, h5_cw))

            # Horizontal flip
            do_flip = self.horizontal_flips and (torch.rand(1).item() < 0.5)
            if do_flip:
                cropped = v2.functional.hflip(cropped)
            global_crop_flips.append(do_flip)

            # Post-geometric augmentation
            cropped = self._apply_augmentation(cropped, is_global=True, aug_idx=aug_idx)
            global_crops.append(cropped)

            if self.debug_crop_dims:
                hr_t, hr_l = h5_t * ratio, h5_l * ratio
                hr_ch, hr_cw = h5_ch * ratio, h5_cw * ratio
                logger.info(
                    f"  global[{aug_idx}] crop HR=({hr_t},{hr_l},{hr_ch},{hr_cw}) "
                    f"H5=({h5_t},{h5_l},{h5_ch},{h5_cw}) flip={do_flip} -> ({cropped.shape[1]},{cropped.shape[2]})"
                )

        output = {
            "global_crops": global_crops,
            "global_crop_flips": global_crop_flips,
            "h5_crop_params": h5_crop_params,
            "hr_h5_resolution_ratio": ratio,
        }

        # Local crops
        local_crops = []
        for li in range(self.local_crops_number):
            cropped, top_f, left_f, h_f, w_f = self._apply_geometric_crop(
                image_tensor, self.local_crops_size, self.local_crops_scale
            )
            # Horizontal flip
            do_flip = self.horizontal_flips and (torch.rand(1).item() < 0.5)
            if do_flip:
                cropped = v2.functional.hflip(cropped)
            # Post-geometric augmentation
            cropped = self._apply_augmentation(cropped, is_global=False, aug_idx=0)
            local_crops.append(cropped)

            if self.debug_crop_dims:
                hr_t = int(top_f * hr_h)
                hr_l = int(left_f * hr_w)
                hr_ch = int(h_f * hr_h)
                hr_cw = int(w_f * hr_w)
                logger.info(
                    f"  local[{li}]  crop HR=({hr_t},{hr_l},{hr_ch},{hr_cw}) "
                    f"flip={do_flip} -> ({cropped.shape[1]},{cropped.shape[2]})"
                )

        output["local_crops"] = local_crops
        return output

"""Tests for H5OlmoEarthDataset with DataAugmentationDINOMultiChannel.

Validates the full pipeline: synthetic data -> dataset -> augmentation ->
collation -> batching, with all new parameters (hr_h5_resolution_ratio,
spatial_align, n_channels, hr_crop_scale).

Usage:
    source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate
    python tests/test_h5_olmoearth_dataloader.py
"""

import os
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_olmoearth_pretrain = os.environ.get("OLMOEARTH_PRETRAIN_ROOT")
if _olmoearth_pretrain:
    sys.path.insert(0, _olmoearth_pretrain)

from h5_data_maker import build_test_dataset
from dinov3.data.datasets.h5_olmoearth import H5OlmoEarthDataset
from dinov3.data.augmentations import DataAugmentationDINOMultiChannel
from dinov3.data.collate import collate_h5_olmoearth_and_cast
from dinov3.data.masking import MaskingGenerator

# --- Test parameters (matching YAML config) ---
RATIO = 20
SPATIAL_ALIGN = 2
N_CHANNELS = 4
GLOBAL_CROPS_SIZE = 224
LOCAL_CROPS_SIZE = 96
HR_CROP_SCALE = (0.1, 0.2)


def test_dataset_basic():
    """Test basic dataset loading without augmentation."""

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = build_test_dataset(tmpdir, n_samples=2)

        dataset = H5OlmoEarthDataset(
            root=index_path,
            transform=DataAugmentationDINOMultiChannel(
                global_crops_scale=(0.4, 1.0),
                local_crops_scale=(0.05, 0.4),
                local_crops_number=8,
                global_crops_size=GLOBAL_CROPS_SIZE,
                local_crops_size=LOCAL_CROPS_SIZE,
                n_channels=N_CHANNELS,
                hr_h5_resolution_ratio=RATIO,
                spatial_align=SPATIAL_ALIGN,
            ),
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
            hr_crop_scale=(0.5, 1.0),
        )

        assert len(dataset) > 0, "Dataset should have at least 1 sample"

        sample = dataset[0]
        assert "global_crops" in sample
        assert "local_crops" in sample
        assert "h5_olmoearth_crops" in sample
        assert len(sample["global_crops"]) == 2
        assert len(sample["local_crops"]) == 8
        assert len(sample["h5_olmoearth_crops"]) == 2

        # Check HR crop shapes (N_CHANNELS for RGBNIR)
        for gc in sample["global_crops"]:
            assert gc.shape == (N_CHANNELS, GLOBAL_CROPS_SIZE, GLOBAL_CROPS_SIZE), \
                f"Expected ({N_CHANNELS}, {GLOBAL_CROPS_SIZE}, {GLOBAL_CROPS_SIZE}), got {gc.shape}"
        for lc in sample["local_crops"]:
            assert lc.shape == (N_CHANNELS, LOCAL_CROPS_SIZE, LOCAL_CROPS_SIZE), \
                f"Expected ({N_CHANNELS}, {LOCAL_CROPS_SIZE}, {LOCAL_CROPS_SIZE}), got {lc.shape}"

        # Check H5 modality shapes
        for crop_h5 in sample["h5_olmoearth_crops"]:
            assert "modalities" in crop_h5
            assert "metadata" in crop_h5
            # All spatial modalities should have the same H5 target size
            target_h5_h = max(SPATIAL_ALIGN, (GLOBAL_CROPS_SIZE // RATIO) // SPATIAL_ALIGN * SPATIAL_ALIGN)
            for mod_name, mod_tensor in crop_h5["modalities"].items():
                assert mod_tensor.shape[0] == target_h5_h, \
                    f"{mod_name}: expected H={target_h5_h}, got {mod_tensor.shape[0]}"
                assert mod_tensor.shape[1] == target_h5_h, \
                    f"{mod_name}: expected W={target_h5_h}, got {mod_tensor.shape[1]}"

        print("  Basic dataset loading test PASSED")


def test_dataset_with_full_config():
    """Test dataset with full YAML-like config parameters."""

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = build_test_dataset(tmpdir, n_samples=2)

        transform = DataAugmentationDINOMultiChannel(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=GLOBAL_CROPS_SIZE,
            local_crops_size=LOCAL_CROPS_SIZE,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
            debug_crop_dims=False,
        )

        dataset = H5OlmoEarthDataset(
            root=index_path,
            transform=transform,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
            hr_crop_scale=HR_CROP_SCALE,
            debug_crop_dims=False,
        )

        sample = dataset[0]

        # Verify H5 spatial dims are all the same (uniform resize)
        for crop_i, crop_h5 in enumerate(sample["h5_olmoearth_crops"]):
            h5_shapes = set()
            for mod_name, mod_tensor in crop_h5["modalities"].items():
                h5_shapes.add((mod_tensor.shape[0], mod_tensor.shape[1]))
            assert len(h5_shapes) == 1, \
                f"Crop {crop_i}: H5 modalities have inconsistent spatial dims: {h5_shapes}"

        print("  Full config dataset test PASSED")


def test_collate():
    """Test collation with collate_h5_olmoearth_and_cast."""

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = build_test_dataset(tmpdir, n_samples=4)

        transform = DataAugmentationDINOMultiChannel(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=GLOBAL_CROPS_SIZE,
            local_crops_size=LOCAL_CROPS_SIZE,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
        )

        dataset = H5OlmoEarthDataset(
            root=index_path,
            transform=transform,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
            hr_crop_scale=HR_CROP_SCALE,
        )

        samples = [dataset[i % len(dataset)] for i in range(4)]

        patch_size = 16
        n_tokens = (GLOBAL_CROPS_SIZE // patch_size) ** 2
        mask_generator = MaskingGenerator(
            input_size=(GLOBAL_CROPS_SIZE // patch_size, GLOBAL_CROPS_SIZE // patch_size),
            max_num_patches=int(0.5 * n_tokens),
        )

        batch = collate_h5_olmoearth_and_cast(
            samples,
            mask_ratio_tuple=(0.1, 0.5),
            mask_probability=0.5,
            dtype=torch.float32,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            spatial_align=SPATIAL_ALIGN,
        )

        B = 4  # batch size
        assert "collated_global_crops" in batch
        assert "collated_local_crops" in batch
        assert "collated_masks" in batch
        assert "olmoearth_modalities" in batch
        assert "olmoearth_metadata" in batch

        assert batch["collated_global_crops"].shape == (2 * B, N_CHANNELS, GLOBAL_CROPS_SIZE, GLOBAL_CROPS_SIZE)
        assert batch["collated_local_crops"].shape == (8 * B, N_CHANNELS, LOCAL_CROPS_SIZE, LOCAL_CROPS_SIZE)

        # olmoearth_modalities is a list per crop
        assert len(batch["olmoearth_modalities"]) == 2  # 2 global crops
        for crop_modalities in batch["olmoearth_modalities"]:
            assert "sentinel2_l2a" in crop_modalities
            # Batched: [B, ...]
            assert crop_modalities["sentinel2_l2a"].shape[0] == B

        print("  Collation test PASSED")
        print(f"    Global crops: {batch['collated_global_crops'].shape}")
        print(f"    Local crops: {batch['collated_local_crops'].shape}")
        print(f"    Modalities per crop: {list(batch['olmoearth_modalities'][0].keys())}")


def test_dataloader():
    """Test with PyTorch DataLoader."""

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = build_test_dataset(tmpdir, n_samples=8)

        transform = DataAugmentationDINOMultiChannel(
            global_crops_scale=(0.32, 1.0),
            local_crops_scale=(0.05, 0.32),
            local_crops_number=8,
            global_crops_size=GLOBAL_CROPS_SIZE,
            local_crops_size=LOCAL_CROPS_SIZE,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
        )

        dataset = H5OlmoEarthDataset(
            root=index_path,
            transform=transform,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
            hr_crop_scale=HR_CROP_SCALE,
        )

        patch_size = 16
        n_tokens = (GLOBAL_CROPS_SIZE // patch_size) ** 2
        mask_generator = MaskingGenerator(
            input_size=(GLOBAL_CROPS_SIZE // patch_size, GLOBAL_CROPS_SIZE // patch_size),
            max_num_patches=int(0.5 * n_tokens),
        )

        from torch.utils.data import DataLoader

        collate_fn = partial(
            collate_h5_olmoearth_and_cast,
            mask_ratio_tuple=(0.1, 0.5),
            mask_probability=0.5,
            dtype=torch.float32,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            spatial_align=SPATIAL_ALIGN,
        )

        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )

        batch = next(iter(loader))
        assert "collated_global_crops" in batch
        assert "olmoearth_modalities" in batch

        print("  DataLoader test PASSED")
        print(f"    Global crops: {batch['collated_global_crops'].shape}")
        print(f"    Local crops: {batch['collated_local_crops'].shape}")


def test_h5_alignment():
    """Verify that H5 modalities stay spatially aligned through Level 1 + Level 2 crops."""

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = build_test_dataset(tmpdir, n_samples=3)

        transform = DataAugmentationDINOMultiChannel(
            global_crops_scale=(0.4, 1.0),
            local_crops_scale=(0.05, 0.4),
            local_crops_number=4,
            global_crops_size=GLOBAL_CROPS_SIZE,
            local_crops_size=LOCAL_CROPS_SIZE,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
        )

        dataset = H5OlmoEarthDataset(
            root=index_path,
            transform=transform,
            n_channels=N_CHANNELS,
            hr_h5_resolution_ratio=RATIO,
            spatial_align=SPATIAL_ALIGN,
            hr_crop_scale=HR_CROP_SCALE,
        )

        # Check multiple samples to catch random alignment issues
        for i in range(min(3, len(dataset))):
            sample = dataset[i]

            for crop_i, crop_h5 in enumerate(sample["h5_olmoearth_crops"]):
                spatial_dims = {}
                for mod_name, mod_tensor in crop_h5["modalities"].items():
                    spatial_dims[mod_name] = (mod_tensor.shape[0], mod_tensor.shape[1])

                # All modalities must have identical spatial dimensions
                unique_dims = set(spatial_dims.values())
                assert len(unique_dims) == 1, \
                    f"Sample {i}, crop {crop_i}: misaligned H5 dims: {spatial_dims}"

                # Verify target size matches expected
                target_h = max(SPATIAL_ALIGN, (GLOBAL_CROPS_SIZE // RATIO) // SPATIAL_ALIGN * SPATIAL_ALIGN)
                actual_h, actual_w = list(unique_dims)[0]
                assert actual_h == target_h, f"Expected H5 H={target_h}, got {actual_h}"
                assert actual_w == target_h, f"Expected H5 W={target_h}, got {actual_w}"

        print("  H5 alignment test PASSED")


if __name__ == "__main__":
    print("Testing H5OlmoEarthDataset with new API...")
    test_dataset_basic()
    test_dataset_with_full_config()
    test_collate()
    test_dataloader()
    test_h5_alignment()
    print("\nAll tests PASSED!")

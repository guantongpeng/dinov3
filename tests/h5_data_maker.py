"""Create synthetic test data for H5OlmoEarthDataset tests.

Creates a directory structure with:
  - H5 files containing OLMoEarth modalities (sentinel2_l2a, sentinel1, etc.)
  - TIFF files containing HR_4V (4-channel RGBNIR) data
  - CSV meta files mapping TIFF image indices to start times
  - A dataset_index.json pointing to all files

Usage:
    source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate
    python tests/h5_data_maker.py
"""

import csv
import json
import os
import sys

import h5py
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path


# Configurable dimensions
HR_H, HR_W = 640, 640  # HR image size (must be divisible by hr_h5_resolution_ratio)
H5_H, H5_W = 32, 32    # H5 spatial dims (HR_H // ratio, HR_W // ratio)
T = 4                   # Temporal sequence length
N_CHANNELS = 4          # RGBNIR


def create_tif(path: str, height: int, width: int, n_bands: int = 4):
    """Create a synthetic GeoTIFF with random data."""
    transform = from_bounds(0, 0, 1, 1, width, height)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_bands,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        for b in range(n_bands):
            data = np.random.randint(0, 255, (height, width), dtype=np.uint8)
            dst.write(data, b + 1)


def create_h5(path: str, h: int = H5_H, w: int = H5_W, t: int = T):
    """Create a synthetic H5 file with OLMoEarth modalities."""
    with h5py.File(path, "w") as f:
        # sentinel2_l2a: [H, W, T, 12]
        f.create_dataset("sentinel2_l2a", data=np.random.randint(0, 10000, (h, w, t, 12), dtype=np.uint16))
        # sentinel1: [H, W, T, 2]
        f.create_dataset("sentinel1", data=np.random.randn(h, w, t, 2).astype(np.float32))
        # landsat: [H, W, T, 11]
        f.create_dataset("landsat", data=np.random.randint(0, 10000, (h, w, t, 11), dtype=np.uint16))
        # timestamps: [T, 3]
        f.create_dataset("timestamps", data=np.random.randint(0, 1000, (t, 3), dtype=np.int32))
        # latlon: [2]
        f.create_dataset("latlon", data=np.array([39.9, 116.4], dtype=np.float64))

        # Missing timesteps masks
        mask_grp = f.create_group("missing_timesteps_masks")
        mask_grp.create_dataset("sentinel2_l2a", data=np.array([True] * t))
        mask_grp.create_dataset("sentinel1", data=np.array([True] * (t - 1) + [False]))
        mask_grp.create_dataset("landsat", data=np.array([True] * (t - 2) + [False] * 2))


def create_hr_meta_csv(path: str, tif_names: list[str]):
    """Create a CSV mapping TIFF image indices to start times."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["crs", "col", "row", "tile_time", "imageidx", "start_time", "end_time"])
        writer.writeheader()
        for i, name in enumerate(tif_names):
            month = i + 1
            writer.writerow({
                "crs": "EPSG:4326",
                "col": 0,
                "row": 0,
                "tile_time": f"2024-{month:02d}",
                "imageidx": name,
                "start_time": f"2024-{month:02d}-01T00:00:00",
                "end_time": f"2024-{month:02d}-28T23:59:59",
            })


def build_test_dataset(output_dir: str, n_samples: int = 3):
    """Build a complete test dataset with H5, TIFF, CSV, and JSON index."""
    os.makedirs(output_dir, exist_ok=True)

    year = "2024"
    h5_dir = Path(output_dir) / year / "H5"
    hr_dir = Path(output_dir) / year / "HR"
    meta_dir = hr_dir / "meta"
    h5_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    samples_index = []

    for i in range(n_samples):
        sample_id = f"sample_{i}"

        # H5 file
        h5_path = h5_dir / f"{sample_id}.h5"
        create_h5(str(h5_path))

        # TIFF files (2-3 temporal images per sample)
        tif_dir = hr_dir / sample_id
        tif_dir.mkdir(parents=True, exist_ok=True)

        n_tifs = 2 + (i % 2)  # 2 or 3 TIFFs per sample
        tif_names = [f"T{j}" for j in range(n_tifs)]
        tif_info = []

        for tif_name in tif_names:
            tif_path = tif_dir / f"{tif_name}.tif"
            create_tif(str(tif_path), HR_H, HR_W, N_CHANNELS)
            tif_info.append({
                "imageidx": tif_name,
                "tif_path": str(Path(year) / "HR" / sample_id / f"{tif_name}.tif"),
                "start_time": f"2024-{(tif_names.index(tif_name) + 1):02d}",
            })

        # Meta CSV
        csv_path = meta_dir / f"{sample_id}.csv"
        create_hr_meta_csv(str(csv_path), tif_names)

        samples_index.append({
            "sample_id": sample_id,
            "years": {
                year: {
                    "h5_path": str(Path(year) / "H5" / f"{sample_id}.h5"),
                    "hr_dims": [HR_H, HR_W, N_CHANNELS],
                    "tif_info": tif_info,
                }
            },
        })

    # Write JSON index
    index = {
        "dataset_root": output_dir,
        "samples": samples_index,
    }
    index_path = Path(output_dir) / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Created test dataset in {output_dir}")
    print(f"  Samples: {n_samples}")
    print(f"  HR size: ({HR_H}, {HR_W}), {N_CHANNELS} channels")
    print(f"  H5 size: ({H5_H}, {H5_W})")
    print(f"  Index: {index_path}")

    return str(index_path)


if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(__file__), "test_h5_olmoearth_v2")
    build_test_dataset(test_dir, n_samples=4)

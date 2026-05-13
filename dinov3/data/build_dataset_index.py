#!/usr/bin/env python3
"""Build a JSON index file for year-organized H5+HR datasets.

Scans a directory with the following structure:

    <dataset_root>/
      2024/
        H5/sample_0.h5, sample_1.h5, ...
        HR/sample_0/T0.tif, T1.tif, ...
        HR/meta/sample_0.csv, ...
      2025/
        H5/sample_0.h5, ...
        HR/sample_0/T0.tif, ...
        HR/meta/sample_0.csv, ...

And produces a JSON index that groups each geographic sampling point
(sample_id) across all available years.  Each (year, sample_id) pair
is an independent training sample.

Usage:
    python build_dataset_index.py <dataset_root> [-o index.json]
"""

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

YEAR_PATTERN = re.compile(r"^\d{4}$")


def _parse_start_time(time_str: str) -> str:
    try:
        dt = datetime.fromisoformat(time_str)
        return dt.strftime("%Y-%m")
    except (ValueError, TypeError):
        return time_str[:7] if len(time_str) >= 7 else time_str


def _parse_hr_meta_csv(csv_path: str) -> dict[str, str]:
    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imageidx = row["imageidx"]
            start_time = _parse_start_time(row["start_time"])
            result[imageidx] = start_time
    return result


def _get_tif_dimensions(tif_path: str) -> list[int]:
    import rasterio
    with rasterio.open(tif_path) as src:
        return [src.height, src.width, src.count]


def discover_year_dirs(root: Path) -> list[Path]:
    """Find year subdirectories (names matching YYYY)."""
    year_dirs = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and YEAR_PATTERN.match(child.name):
            year_dirs.append(child)
    return year_dirs


def scan_year(year_dir: Path) -> dict[str, dict]:
    """Scan one year directory and return {sample_id: year_data_dict}."""
    year = year_dir.name
    h5_dir = year_dir / "H5"
    hr_dir = year_dir / "HR"
    hr_meta_dir = hr_dir / "meta"

    if not h5_dir.is_dir():
        logger.warning(f"No H5 directory in {year_dir}")
        return {}

    results = {}

    for h5_path in sorted(h5_dir.glob("*.h5")) + sorted(h5_dir.glob("*.hdf5")):
        sample_id = h5_path.stem
        tif_folder = hr_dir / sample_id

        if not tif_folder.is_dir():
            logger.debug(f"Skipping {year}/{sample_id}: no TIFF folder at {tif_folder}")
            continue

        tif_files = sorted(
            list(tif_folder.glob("*.tif")) + list(tif_folder.glob("*.tiff"))
        )
        if not tif_files:
            logger.debug(f"Skipping {year}/{sample_id}: no TIFF files in {tif_folder}")
            continue

        # HR dimensions from first TIFF
        hr_dims = _get_tif_dimensions(str(tif_files[0]))

        # Parse meta CSV
        csv_path = hr_meta_dir / f"{sample_id}.csv"
        meta = _parse_hr_meta_csv(str(csv_path)) if csv_path.exists() else {}

        tif_info = []
        for tif_path in tif_files:
            imageidx = tif_path.stem
            start_time = meta.get(imageidx)
            tif_info.append({
                "imageidx": imageidx,
                "tif_path": str(tif_path.relative_to(year_dir.parent)),
                "start_time": start_time,
            })

        results[sample_id] = {
            "h5_path": str(h5_path.relative_to(year_dir.parent)),
            "hr_dir": str(tif_folder.relative_to(year_dir.parent)),
            "hr_meta_path": str(csv_path.relative_to(year_dir.parent)) if csv_path.exists() else None,
            "hr_dims": hr_dims,
            "tif_info": tif_info,
        }

    logger.info(f"Year {year}: found {len(results)} valid samples")
    return results


def build_index(dataset_root: str) -> dict:
    """Build the full JSON index from a year-organized dataset root."""
    root = Path(dataset_root).resolve()

    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    # Check for year subdirectories
    year_dirs = discover_year_dirs(root)
    if not year_dirs:
        raise ValueError(
            f"No year subdirectories (YYYY) found in {root}. "
            "Expected structure: <root>/<YYYY>/H5/ and <root>/<YYYY>/HR/"
        )

    # Group samples by sample_id across years
    sample_groups: dict[str, dict] = {}  # sample_id -> {year: year_data}

    for year_dir in year_dirs:
        year_samples = scan_year(year_dir)
        for sample_id, year_data in year_samples.items():
            if sample_id not in sample_groups:
                sample_groups[sample_id] = {"sample_id": sample_id, "years": {}}
            sample_groups[sample_id]["years"][year_dir.name] = year_data

    # Convert to sorted list
    samples = sorted(sample_groups.values(), key=lambda s: s["sample_id"])

    total_training_samples = sum(len(s["years"]) for s in samples)
    logger.info(
        f"Index built: {len(samples)} geographic points, "
        f"{total_training_samples} training samples (year x point)"
    )

    return {
        "dataset_root": str(root),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Build JSON index for H5+HR dataset")
    parser.add_argument("dataset_root", help="Root directory of the year-organized dataset")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path (default: <dataset_root>/dataset_index.json)")
    args = parser.parse_args()

    index = build_index(args.dataset_root)

    output_path = args.output or str(Path(args.dataset_root) / "dataset_index.json")
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    logger.info(f"Index written to {output_path}")


if __name__ == "__main__":
    main()

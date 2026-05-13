"""H5 + TIFF dataset for DINO V3 stage 2 training with OLMoEarth multi-modal data.

This dataset reads H5 files for OLMoEarth modalities and TIFF files for HR_4V
data. The HR_4V data is stored as multi-temporal TIFF files in a separate
directory, with meta information (start_time) in CSV files.

Key behaviors:
    - HR_4V data is read from TIFF files (using rasterio), not from H5.
    - ``hr_data_dir`` is required — must point to the TIFF root directory.
    - If a sample's TIFF folder has no files, the sample is skipped.
    - One TIFF is randomly selected as ``data`` (DINO V3 input), the rest as
      ``targets``. Both carry ``start_time`` (YYYY-MM) from the CSV meta.
    - A random spatial crop is applied to both HR and H5 modalities so they
      cover the same physical region.  Because HR has a much higher resolution,
      the pixel-level crop ranges differ (proportional to resolution ratio).
    - HR images may have 3 or 4 channels (RGB or RGB+NIR). The first 3
      channels are used for DINO V3 augmentation; all channels are preserved
      in the target tensors.
    - Other modalities are read from H5 and processed via OLMoEarth's
      normalization pipeline.
"""

import copy
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

logger = logging.getLogger("dinov3")

# Default values — overridden by cfg.olmoearth in YAML
_DEFAULT_MAX_SEQUENCE_LENGTH = 12
_DEFAULT_MISSING_VALUE = -99999
_DEFAULT_SPATIAL_ALIGN = 4

OLMOEARTH_MODALITIES = [
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    # "worldcover",
    # "srtm",
    # "openstreetmap_raster",
    # "wri_canopy_height_map",
    # "cdl",
    # "worldcereal",
]


# ---------------------------------------------------------------------------
# H5 helpers (unchanged)
# ---------------------------------------------------------------------------

def _read_h5_file(h5_path: str, modalities: list[str] | None = None) -> tuple[dict, dict]:
    sample_dict = {}
    missing_timesteps_masks = {}

    with h5py.File(h5_path, "r") as h5file:
        keys = list(h5file.keys())

        for k in keys:
            if k == "missing_timesteps_masks":
                for mk, mv in h5file[k].items():
                    if modalities is None or mk in modalities:
                        missing_timesteps_masks[mk] = mv[()]
                continue

            if modalities is None or k in modalities or k == "timestamps":
                sample_dict[k] = h5file[k][()]

    return sample_dict, missing_timesteps_masks


def _pad_timestamps(sample_dict: dict, max_sequence_length: int) -> tuple[dict, int]:
    timestamps = sample_dict["timestamps"]
    current_length = timestamps.shape[0]
    if current_length < max_sequence_length:
        pad_width = ((0, max_sequence_length - current_length), (0, 0))
        sample_dict["timestamps"] = np.pad(timestamps, pad_width=pad_width, mode="edge")
    return sample_dict, current_length


def _fill_missing_timesteps(
    modality_data: np.ndarray, mask: np.ndarray, max_sequence_length: int, dtype: np.dtype,
    missing_value: float = _DEFAULT_MISSING_VALUE,
) -> np.ndarray:
    modality_data = modality_data.astype(dtype)
    h, w, t, c = modality_data.shape
    full_data = np.full((h, w, max_sequence_length, c), missing_value, dtype=dtype)
    present_indices = np.where(mask)[0]
    num_to_copy = min(len(present_indices), t)
    if num_to_copy > 0:
        full_data[:, :, present_indices[:num_to_copy], :] = modality_data[:, :, :num_to_copy, :]
    return full_data


def _fill_sample_with_missing_values(
    sample_dict: dict,
    inference_modalities: list[str],
    missing_timesteps_masks: dict,
    max_sequence_length: int,
    dtype: np.dtype,
    missing_value: float = _DEFAULT_MISSING_VALUE,
) -> tuple[dict, list[str]]:

    missing_modalities = []

    time = sample_dict["timestamps"].shape[0]
    height, width = None, None
    for mod_name, mod_data in sample_dict.items():
        if mod_name in ("timestamps", "latlon"):
            continue
        try:
            mod_spec = Modality.get(mod_name)
        except Exception:
            continue
        if mod_spec.is_spatial and mod_data is not None:
            height = mod_data.shape[0] // mod_spec.image_tile_size_factor
            width = mod_data.shape[1] // mod_spec.image_tile_size_factor
            break

    for modality in inference_modalities:
        if modality not in sample_dict:
            mod_spec = Modality.get(modality)
            expected_shape = OlmoEarthSample.compute_expected_shape(
                modality, height, width, time
            )
            sample_dict[modality] = np.full(expected_shape, missing_value, dtype=dtype)
            missing_modalities.append(modality)
            continue

        if modality in missing_timesteps_masks:
            mask = missing_timesteps_masks[modality]
            modality_data = sample_dict[modality].astype(dtype)
            has_missing = not np.all(mask) or len(mask) < max_sequence_length
            if has_missing:
                sample_dict[modality] = _fill_missing_timesteps(
                    modality_data, mask, max_sequence_length, dtype, missing_value
                )

    return sample_dict, missing_modalities


def _normalize_sample(
    sample_dict: dict,
    missing_modalities: list[str],
    normalizer_computed=None,
    normalizer_predefined=None,
    missing_value: float = _DEFAULT_MISSING_VALUE,
) -> dict:

    if normalizer_computed is None:
        normalizer_computed = Normalizer(Strategy.COMPUTED)
    if normalizer_predefined is None:
        normalizer_predefined = Normalizer(Strategy.PREDEFINED)

    for modality_name in sample_dict:
        if modality_name in ("timestamps", "latlon"):
            continue
        if modality_name in missing_modalities:
            continue

        modality_data = sample_dict[modality_name]
        missing_mask = modality_data == missing_value

        mod_spec = Modality.get(modality_name)
        try:
            normalized = normalizer_computed.normalize(mod_spec, modality_data)
        except Exception as e:
            logger.warning(
                f"Modality {modality_name} COMPUTED normalization failed ({e}), "
                "falling back to PREDEFINED strategy"
            )
            normalized = normalizer_predefined.normalize(mod_spec, modality_data)

        sample_dict[modality_name] = np.where(missing_mask, modality_data, normalized).astype(
            np.float32
        )

    return sample_dict


# ---------------------------------------------------------------------------
# TIFF / HR helpers
# ---------------------------------------------------------------------------

def _parse_start_time(time_str: str) -> str:
    """Parse an ISO timestamp and return YYYY-MM (month precision)."""
    try:
        dt = datetime.fromisoformat(time_str)
        return dt.strftime("%Y-%m")
    except (ValueError, TypeError):
        return time_str[:7] if len(time_str) >= 7 else time_str


def _parse_hr_meta_csv(csv_path: str) -> dict[str, str]:
    """Parse HR meta CSV and return {imageidx: start_time_yyyy_mm}.

    CSV columns: crs, col, row, tile_time, imageidx, start_time, end_time
    """
    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            imageidx = row["imageidx"]
            start_time = _parse_start_time(row["start_time"])
            result[imageidx] = start_time
    return result


def _get_tif_dimensions(tif_path: str) -> tuple[int, int, int]:
    """Get (height, width, bands) from a TIFF file without reading pixels."""
    with rasterio.open(tif_path) as src:
        return src.height, src.width, src.count


def _random_crop_params(
    hr_h: int, hr_w: int, crop_scale: tuple[float, float]
) -> tuple[int, int, int, int]:
    """Generate random crop parameters in HR pixel coordinates.

    Returns (top, left, crop_h, crop_w).
    """
    scale = random.uniform(crop_scale[0], crop_scale[1])
    crop_h = max(1, int(hr_h * scale))
    crop_w = max(1, int(hr_w * scale))
    top = random.randint(0, hr_h - crop_h)
    left = random.randint(0, hr_w - crop_w)
    return top, left, crop_h, crop_w


_NON_SPATIAL_KEYS = frozenset({"timestamps", "latlon"})


def _random_crop_params_aligned(
    h: int, w: int, crop_scale: tuple[float, float], align: int = _DEFAULT_SPATIAL_ALIGN
) -> tuple[int, int, int, int]:
    """Generate random crop parameters in H5 reference space with alignment.

    Crop dimensions are rounded down to multiples of ``align`` so that H5
    spatial sizes are compatible with the OLMoEarth patchify step.  Start
    positions are also aligned so that the corresponding HR pixels (obtained
    by scaling up) land on exact integer boundaries.

    Returns (top, left, crop_h, crop_w), all in the reference pixel space.
    """
    scale = random.uniform(crop_scale[0], crop_scale[1])
    crop_h = max(align, int(h * scale) // align * align)
    crop_w = max(align, int(w * scale) // align * align)

    # Clamp crop size to not exceed image dimensions
    crop_h = min(crop_h, (h // align) * align)
    crop_w = min(crop_w, (w // align) * align)

    max_top = h - crop_h
    max_left = w - crop_w
    top = random.randint(0, max_top // align) * align if max_top > 0 else 0
    left = random.randint(0, max_left // align) * align if max_left > 0 else 0

    return top, left, crop_h, crop_w


def _read_tif_crop(
    tif_path: str, top: int, left: int, crop_h: int, crop_w: int
) -> np.ndarray:
    """Read a crop window from a TIFF file, return (H, W, C) uint8 array.

    Uses rasterio windowed reading so only the required region is loaded.
    """
    with rasterio.open(tif_path) as src:
        window = Window(col_off=left, row_off=top, width=crop_w, height=crop_h)
        data = src.read(window=window)  # (C, H, W)

    data = np.transpose(data, (1, 2, 0))  # (H, W, C)

    if data.dtype in (np.uint16, np.int16):
        data = (data / 256.0).clip(0, 255).astype(np.uint8)
    elif data.dtype in (np.float32, np.float64):
        if data.max() <= 1.0:
            data = (data * 255).clip(0, 255).astype(np.uint8)
        else:
            data = data.clip(0, 255).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = data.astype(np.float32).clip(0, 255).astype(np.uint8)

    return data


def _hr_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert (H, W, C) uint8 array to (C, H, W) float tensor in [0, 1].

    Preserves all channels (supports C=3 or C=4 for RGBNIR).
    """
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0


def _crop_h5_sample_dict_fractional(
    sample_dict: dict,
    top_frac: float, left_frac: float, h_frac: float, w_frac: float,
    align: int = _DEFAULT_SPATIAL_ALIGN,
) -> dict:
    """Crop all H5 spatial modalities with the same fractional coordinates.

    Since H5 modalities are stored at the same spatial resolution, applying
    the same fractions selects the same physical sub-region in every modality.
    Crop dimensions are rounded down to multiples of ``align``.
    """
    cropped = {}
    for key, val in sample_dict.items():
        if val is None:
            cropped[key] = None
            continue

        if key in _NON_SPATIAL_KEYS:
            cropped[key] = val
            continue

        is_spatial = val.ndim >= 2 and val.shape[0] > 1 and val.shape[1] > 1
        if is_spatial:
            mod_h, mod_w = val.shape[0], val.shape[1]
            m_top = int(top_frac * mod_h)
            m_left = int(left_frac * mod_w)
            m_crop_h = max(align, int(h_frac * mod_h) // align * align)
            m_crop_w = max(align, int(w_frac * mod_w) // align * align)

            # Clamp to valid range while preserving alignment
            m_top = min(m_top, mod_h - m_crop_h)
            m_left = min(m_left, mod_w - m_crop_w)
            if m_top < 0:
                m_top = 0
                m_crop_h = (mod_h // align) * align
            if m_left < 0:
                m_left = 0
                m_crop_w = (mod_w // align) * align

            cropped[key] = val[m_top:m_top + m_crop_h, m_left:m_left + m_crop_w]
        else:
            cropped[key] = val

    return cropped


def _flip_h5_sample_dict(sample_dict: dict) -> dict:
    """Horizontally flip spatial modalities in the sample dict."""
    flipped = {}
    for key, val in sample_dict.items():
        if val is None:
            flipped[key] = None
        elif key in ("timestamps", "latlon"):
            flipped[key] = val
        elif val.ndim >= 2:
            flipped[key] = np.flip(val, axis=1).copy()  # flip W axis
        else:
            flipped[key] = val
    return flipped


def _resize_h5_sample_dict_uniform(
    sample_dict: dict,
    hr_h5_resolution_ratio: int,
    global_crops_size: int,
    align: int = _DEFAULT_SPATIAL_ALIGN,
) -> dict:
    """Resize all H5 spatial modalities to the same uniform target size.

    All H5 modalities are stored at the same spatial resolution, so after
    cropping they should all be resized to the same target:
        target = aligned(global_crops_size / hr_h5_resolution_ratio)

    This preserves the configured resolution ratio between HR and H5,
    and keeps all H5 modalities at identical spatial size for token-level
    cross-modal alignment.

    Args:
        sample_dict: Level-2-cropped H5 data.
        hr_h5_resolution_ratio: Integer ratio of HR pixels per H5 pixel.
        global_crops_size: Target size HR global crops are resized to.
        align: Spatial dims rounded to multiples of this.
    """
    import torch.nn.functional as F

    target_h = max(align, (global_crops_size // hr_h5_resolution_ratio) // align * align)
    target_w = max(align, (global_crops_size // hr_h5_resolution_ratio) // align * align)

    resized = {}
    for key, val in sample_dict.items():
        if val is None:
            resized[key] = None
            continue

        if key in _NON_SPATIAL_KEYS:
            resized[key] = val
            continue

        is_spatial = val.ndim >= 2 and val.shape[0] > 1 and val.shape[1] > 1
        if is_spatial:
            mod_h, mod_w = val.shape[0], val.shape[1]

            if mod_h == target_h and mod_w == target_w:
                resized[key] = val
            elif val.ndim == 4:
                # (H, W, T, C) → resize H, W via bilinear interpolation
                h, w, t, c = val.shape
                t_flat = torch.tensor(val, dtype=torch.float32).permute(2, 3, 0, 1).reshape(t * c, h, w).unsqueeze(0)
                t_flat = F.interpolate(t_flat, size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
                val = t_flat.reshape(t, c, target_h, target_w).permute(2, 3, 0, 1).numpy()
                resized[key] = val
            elif val.ndim == 3:
                # (H, W, C) → resize H, W
                h, w, c = val.shape
                t_flat = torch.tensor(val, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                t_flat = F.interpolate(t_flat, size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
                resized[key] = t_flat.permute(1, 2, 0).numpy()
            else:
                resized[key] = val
        else:
            resized[key] = val

    return resized


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class H5OlmoEarthDataset(ExtendedVisionDataset):
    """Dataset for DINO V3 stage 2 training with OLMoEarth H5 + TIFF data.

    Each sample loads:
        - OLMoEarth modalities from an H5 file (normalized)
        - HR_4V data from TIFF files in ``hr_data_dir``

    A random spatial crop is applied so that HR and H5 modalities cover the
    same physical region.  The HR data image is resized to ``hr_image_size``
    and normalized (no multi-crop augmentation).

    Args:
        root: Path to directory containing H5 files, or a text file listing
              H5 file paths (one per line).
        transform: DataAugmentationDINOMultiChannel instance for DINO global/local
                   crop augmentation. Must be provided for proper training.
        target_transform: Not used.
        transforms: Not used.
        olmoearth_modalities: List of OLMoEarth modalities to load and process.
        max_sequence_length: Maximum sequence length for temporal padding.
        hr_data_dir: Directory containing HR TIFF folders (e.g. .../HR/).
                     Each subfolder (e.g. sample_0/) holds T*.tif files.
                     Required — HR_4V data is only read from TIFF files.
        hr_meta_dir: Directory containing HR meta CSV files (e.g. .../HR/meta/).
                     Each CSV (e.g. sample_0.csv) maps imageidx -> start_time.
                     Defaults to ``hr_data_dir/meta/`` if not specified.
        hr_crop_scale: (min_scale, max_scale) for the random spatial crop
                       applied to both HR and H5 modalities.  (1.0, 1.0)
                       disables cropping.
        n_channels: Number of image channels (3 for RGB, 4 for RGBNIR).
        spatial_align: Spatial dims rounded to multiples of this for patchify.
        missing_value: Fill value for missing timesteps / modalities.
        debug_crop_dims: If True, log HR/H5 dimensions at each crop stage.
        return_hr_targets: If True, include ``hr_target_images`` and
            ``hr_target_start_times`` in the returned sample dict. Set to
            False to skip reading non-selected TIFFs and reduce per-sample
            memory/IO when targets are not consumed downstream.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        olmoearth_modalities: list[str] | None = None,
        max_sequence_length: int = _DEFAULT_MAX_SEQUENCE_LENGTH,
        hr_data_dir: str | None = None,
        hr_meta_dir: str | None = None,
        hr_crop_scale: tuple[float, float] | None = None,
        n_channels: int = 4,
        spatial_align: int = _DEFAULT_SPATIAL_ALIGN,
        missing_value: float = _DEFAULT_MISSING_VALUE,
        debug_crop_dims: bool = False,
        hr_h5_resolution_ratio: int = 16,
        return_hr_targets: bool = True,
    ):
        super().__init__(
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
            root=root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.olmoearth_modalities = olmoearth_modalities or OLMOEARTH_MODALITIES
        self.max_sequence_length = max_sequence_length
        self.hr_data_dir = hr_data_dir
        self.hr_meta_dir = hr_meta_dir
        self.hr_crop_scale = hr_crop_scale or (0.5, 1.0)
        self.n_channels = n_channels
        self.spatial_align = spatial_align
        self.missing_value = missing_value
        self.debug_crop_dims = debug_crop_dims
        self.hr_h5_resolution_ratio = hr_h5_resolution_ratio
        self.return_hr_targets = return_hr_targets

        if self.transform is None:
            raise ValueError(
                "H5OlmoEarthDataset requires a 'transform' (DataAugmentationDINOMultiChannel). "
                "Pass it via the dataset constructor or the data loader builder."
            )

        self._samples = self._discover_samples()
        logger.info(f"H5OlmoEarth Dataset: found {len(self._samples)} valid samples")

        self._normalizer_computed = None
        self._normalizer_predefined = None

    def _discover_samples(self) -> list[dict]:
        """Build index of valid samples (H5 + matching TIFF folder).

        Supports three discovery modes:
          1. JSON index file (root is a .json path) — uses pre-built index
          2. Year-organized directory (root contains YYYY/ subdirs)
          3. Flat directory (legacy, root contains H5/ and HR/ subdirs)

        Each (year, sample_id) pair becomes an independent training sample.
        """
        root_path = Path(self.root)

        # --- Mode 1: JSON index ---
        if root_path.is_file() and root_path.suffix == ".json":
            return self._discover_from_json(root_path)

        # --- Mode 2 & 3: Directory ---
        if root_path.is_dir():
            year_dirs = self._find_year_dirs(root_path)
            if year_dirs:
                return self._discover_from_year_dirs(root_path, year_dirs)
            else:
                return self._discover_flat(root_path)

        # --- Mode: file list ---
        if root_path.is_file() and root_path.suffix in (".txt", ".lst", ".csv"):
            with open(root_path) as f:
                h5_candidates = [line.strip() for line in f if line.strip()]
            return self._discover_from_h5_list(h5_candidates)

        raise ValueError(f"Cannot discover samples from {root_path}")

    @staticmethod
    def _find_year_dirs(root: Path) -> list[Path]:
        """Find year subdirectories (names matching YYYY)."""
        import re
        year_pat = re.compile(r"^\d{4}$")
        return sorted(
            child for child in root.iterdir()
            if child.is_dir() and year_pat.match(child.name)
        )

    def _discover_from_json(self, json_path: Path) -> list[dict]:
        """Load samples from a pre-built JSON index file."""
        with open(json_path) as f:
            index = json.load(f)

        dataset_root = Path(index["dataset_root"])
        # If dataset_root in JSON is stale, resolve relative to JSON location
        if not dataset_root.is_dir():
            dataset_root = json_path.parent

        samples = []
        for sample_group in index["samples"]:
            sample_id = sample_group["sample_id"]
            for year, year_data in sample_group["years"].items():
                h5_path = dataset_root / year_data["h5_path"]
                hr_dims = year_data["hr_dims"]

                tif_info = []
                for tif_entry in year_data["tif_info"]:
                    tif_path = str(dataset_root / tif_entry["tif_path"])
                    tif_info.append((tif_path, tif_entry.get("start_time")))

                samples.append({
                    "h5_path": str(h5_path),
                    "tif_info": tif_info,
                    "hr_dims": hr_dims,
                    "year": year,
                    "sample_id": sample_id,
                })

        return samples

    def _discover_from_year_dirs(self, root: Path, year_dirs: list[Path]) -> list[dict]:
        """Discover samples from year-organized directory structure."""
        samples = []

        for year_dir in year_dirs:
            year = year_dir.name
            h5_dir = year_dir / "H5"
            hr_dir = year_dir / "HR"
            hr_meta_dir = hr_dir / "meta"

            if not h5_dir.is_dir():
                logger.debug(f"Skipping year {year}: no H5 directory")
                continue

            for h5_path in sorted(h5_dir.glob("*.h5")) + sorted(h5_dir.glob("*.hdf5")):
                sample_name = h5_path.stem
                tif_folder = hr_dir / sample_name

                if not tif_folder.is_dir():
                    logger.debug(f"Skipping {year}/{sample_name}: no TIFF folder")
                    continue

                tif_files = sorted(
                    list(tif_folder.glob("*.tif")) + list(tif_folder.glob("*.tiff"))
                )
                if not tif_files:
                    continue

                hr_dims = _get_tif_dimensions(str(tif_files[0]))

                csv_path = hr_meta_dir / f"{sample_name}.csv"
                meta = _parse_hr_meta_csv(str(csv_path)) if csv_path.exists() else {}

                tif_info = []
                for tif_path in tif_files:
                    imageidx = tif_path.stem
                    start_time = meta.get(imageidx)
                    tif_info.append((str(tif_path), start_time))

                samples.append({
                    "h5_path": str(h5_path),
                    "tif_info": tif_info,
                    "hr_dims": hr_dims,
                    "year": year,
                    "sample_id": sample_name,
                })

        return samples

    def _discover_flat(self, root: Path) -> list[dict]:
        """Discover samples from flat directory (legacy, no year subdirs)."""
        h5_candidates = sorted(str(p) for p in root.rglob("*.h5"))
        h5_candidates += sorted(str(p) for p in root.rglob("*.hdf5"))

        samples = []
        for h5_path in h5_candidates:
            sample_name = Path(h5_path).stem

            if self.hr_data_dir is None:
                logger.debug(f"Skipping {h5_path}: no hr_data_dir specified")
                continue

            tif_folder = Path(self.hr_data_dir) / sample_name
            if not tif_folder.is_dir():
                logger.debug(f"Skipping {h5_path}: no TIFF folder at {tif_folder}")
                continue

            tif_files = sorted(
                list(tif_folder.glob("*.tif")) + list(tif_folder.glob("*.tiff"))
            )
            if not tif_files:
                continue

            hr_dims = _get_tif_dimensions(str(tif_files[0]))

            if self.hr_meta_dir is not None:
                csv_path = Path(self.hr_meta_dir) / f"{sample_name}.csv"
            else:
                csv_path = Path(self.hr_data_dir) / "meta" / f"{sample_name}.csv"

            meta = _parse_hr_meta_csv(str(csv_path)) if csv_path.exists() else {}

            tif_info = []
            for tif_path in tif_files:
                imageidx = tif_path.stem
                start_time = meta.get(imageidx)
                tif_info.append((str(tif_path), start_time))

            samples.append({
                "h5_path": h5_path,
                "tif_info": tif_info,
                "hr_dims": hr_dims,
                "year": None,
                "sample_id": sample_name,
            })

        return samples

    def _discover_from_h5_list(self, h5_candidates: list[str]) -> list[dict]:
        """Discover samples from a list of H5 file paths."""
        samples = []
        for h5_path in h5_candidates:
            sample_name = Path(h5_path).stem

            if self.hr_data_dir is None:
                continue

            tif_folder = Path(self.hr_data_dir) / sample_name
            if not tif_folder.is_dir():
                continue

            tif_files = sorted(
                list(tif_folder.glob("*.tif")) + list(tif_folder.glob("*.tiff"))
            )
            if not tif_files:
                continue

            hr_dims = _get_tif_dimensions(str(tif_files[0]))

            if self.hr_meta_dir is not None:
                csv_path = Path(self.hr_meta_dir) / f"{sample_name}.csv"
            else:
                csv_path = Path(self.hr_data_dir) / "meta" / f"{sample_name}.csv"

            meta = _parse_hr_meta_csv(str(csv_path)) if csv_path.exists() else {}

            tif_info = []
            for tif_path in tif_files:
                imageidx = tif_path.stem
                start_time = meta.get(imageidx)
                tif_info.append((str(tif_path), start_time))

            samples.append({
                "h5_path": h5_path,
                "tif_info": tif_info,
                "hr_dims": hr_dims,
                "year": None,
                "sample_id": sample_name,
            })

        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def get_image_data(self, index: int) -> np.ndarray:
        """Return a randomly cropped HR_4V region as (H, W, C) uint8 array."""
        sample = self._samples[index]
        hr_h, hr_w = sample["hr_dims"][0], sample["hr_dims"][1]
        top, left, crop_h, crop_w = _random_crop_params(hr_h, hr_w, self.hr_crop_scale)
        tif_path, _ = random.choice(sample["tif_info"])
        return _read_tif_crop(tif_path, top, left, crop_h, crop_w)

    def get_target(self, index: int) -> int:
        """Return the sample index as target."""
        return index

    def _process_olmoearth_modalities(
        self, sample_dict: dict, missing_timesteps_masks: dict
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        if self._normalizer_computed is None:
            self._normalizer_computed = Normalizer(Strategy.COMPUTED)
        if self._normalizer_predefined is None:
            self._normalizer_predefined = Normalizer(Strategy.PREDEFINED)

        inference_modalities = self.olmoearth_modalities

        sample_dict, _ = _pad_timestamps(sample_dict, self.max_sequence_length)

        sample_dict, missing_modalities = _fill_sample_with_missing_values(
            sample_dict,
            inference_modalities,
            missing_timesteps_masks,
            self.max_sequence_length,
            np.float32,
            self.missing_value,
        )

        sample_dict = _normalize_sample(
            sample_dict,
            missing_modalities,
            normalizer_computed=self._normalizer_computed,
            normalizer_predefined=self._normalizer_predefined,
            missing_value=self.missing_value,
        )

        modality_tensors = {}
        metadata_tensors = {}

        for key, val in sample_dict.items():
            if val is None:
                continue
            if key == "timestamps":
                metadata_tensors["timestamps"] = torch.tensor(val, dtype=torch.long)
            elif key == "latlon":
                metadata_tensors["latlon"] = torch.tensor(val, dtype=torch.float32)
            else:
                modality_tensors[key] = torch.tensor(val, dtype=torch.float32)

        return modality_tensors, metadata_tensors

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a dict with DINO-augmented HR crops and H5 modality crops.

        All coordinate mapping between HR and H5 uses the configured
        ``hr_h5_resolution_ratio`` for exact integer arithmetic. H5
        modalities are stored at the same spatial resolution, so they
        all receive the same fractional crop and uniform resize.

        Level 1 — Crop in H5 reference space (derived from HR / ratio):
            H5 reference dims = hr_dims / ratio.  A random crop is
            generated in that space (aligned to spatial_align).
            HR coordinates = h5_coord * ratio (exact integer).
            All H5 modalities get the same fractional crop.

        Level 2 — DINO global crop with H5-aligned parameters:
            The augmentation generates crop params in the H5 reference
            space (derived from HR L1 dims / ratio), then maps to HR
            via integer multiplication.  H5 modalities get the same
            fractional crop as each other, then uniform resize to
            aligned(global_crops_size / ratio).

        Returns:
            dict with:
                global_crops: list of (C, gH, gW) normalized tensors
                local_crops: list of (C, lH, lW) normalized tensors
                h5_olmoearth_crops: list of 2 dicts, each with "modalities"
                                    and "metadata" for the corresponding
                                    global crop's H5 region
                hr_data_start_time: str or None
                hr_target_images: list of (H, W, C) uint8 numpy arrays
                hr_target_start_times: list of str or None
        """
        ratio = self.hr_h5_resolution_ratio
        sample = self._samples[index]
        h5_path = sample["h5_path"]
        tif_info = sample["tif_info"]
        hr_h, hr_w = sample["hr_dims"][0], sample["hr_dims"][1]

        # ---- Read H5 file ----
        all_modalities = list(self.olmoearth_modalities) + ["latlon", "timestamps"]
        sample_dict_raw, missing_timesteps_masks = _read_h5_file(h5_path, all_modalities)

        if self.debug_crop_dims:
            logger.info(f"[sample {index}] HR full=({hr_h},{hr_w}), ratio={ratio}")

        # ---- Level 1: Crop in H5 reference space (hr / ratio), map to HR ----
        ref_h, ref_w = hr_h // ratio, hr_w // ratio
        h5_top, h5_left, h5_crop_h, h5_crop_w = _random_crop_params_aligned(
            ref_h, ref_w, self.hr_crop_scale, align=self.spatial_align
        )

        # Fractional coordinates (relative to full H5 reference space)
        top_frac = h5_top / ref_h
        left_frac = h5_left / ref_w
        h_frac = h5_crop_h / ref_h
        w_frac = h5_crop_w / ref_w

        # Map to HR using exact integer multiplication
        hr_top = h5_top * ratio
        hr_left = h5_left * ratio
        hr_crop_h = h5_crop_h * ratio
        hr_crop_w = h5_crop_w * ratio

        # ---- Randomly select data/target TIFF ----
        data_idx = random.randint(0, len(tif_info) - 1)

        data_array = None
        data_start_time = None
        target_arrays: list[np.ndarray] = []
        target_start_times: list[str | None] = []

        # HR: read the Level 1 crop region from TIFF
        for i, (tif_path, start_time) in enumerate(tif_info):
            if i == data_idx:
                arr = _read_tif_crop(tif_path, hr_top, hr_left, hr_crop_h, hr_crop_w)
                data_array = arr
                data_start_time = start_time
            elif self.return_hr_targets:
                arr = _read_tif_crop(tif_path, hr_top, hr_left, hr_crop_h, hr_crop_w)
                target_arrays.append(arr)
                target_start_times.append(start_time)

        # H5: apply Level 1 crop — same fractions for all modalities
        sample_dict = _crop_h5_sample_dict_fractional(
            sample_dict_raw, top_frac, left_frac, h_frac, w_frac,
            align=self.spatial_align,
        )

        if self.debug_crop_dims:
            logger.info(
                f"[L1] scale={self.hr_crop_scale} "
                f"HR=({hr_top},{hr_left},{hr_crop_h},{hr_crop_w}) "
                f"H5=({h5_top},{h5_left},{h5_crop_h},{h5_crop_w})"
            )
            logger.info(f"  HR -> ({data_array.shape[0]},{data_array.shape[1]})")
            for k, v in sample_dict.items():
                if v is not None and isinstance(v, np.ndarray) and k not in _NON_SPATIAL_KEYS and v.ndim >= 2:
                    logger.info(f"  H5/{k} -> ({v.shape[0]},{v.shape[1]})")

        # ---- Level 2: DINO augmentation on HR + H5-aligned crop ----
        hr_tensor = _hr_to_tensor(data_array)  # (C, H', W') float [0, 1]

        # Ensure channel count matches n_channels
        actual_c = hr_tensor.shape[0]
        if actual_c > self.n_channels:
            hr_tensor = hr_tensor[:self.n_channels]
        elif actual_c < self.n_channels:
            pad = torch.zeros(self.n_channels - actual_c, hr_tensor.shape[1], hr_tensor.shape[2])
            hr_tensor = torch.cat([hr_tensor, pad], dim=0)

        # Augmentation uses hr_h5_resolution_ratio internally for Level 2 alignment
        aug_output = self.transform(hr_tensor)

        # Apply Level 2 crop and uniform resize to H5
        h5_olmoearth_crops = []
        global_crops_size = self.transform.global_crops_size

        for crop_i, ((h5_t, h5_l, h5_ch, h5_cw), do_flip) in enumerate(
            zip(aug_output["h5_crop_params"], aug_output["global_crop_flips"])
        ):
            # Convert H5 ref-space crop params to fractions of the L1-cropped space
            hr_L1_h, hr_L1_w = hr_tensor.shape[1], hr_tensor.shape[2]
            h5_L1_h, h5_L1_w = hr_L1_h // ratio, hr_L1_w // ratio
            crop_top_frac = h5_t / h5_L1_h
            crop_left_frac = h5_l / h5_L1_w
            crop_h_frac = h5_ch / h5_L1_h
            crop_w_frac = h5_cw / h5_L1_w

            # Same fractional crop for all H5 modalities
            cropped_h5 = _crop_h5_sample_dict_fractional(
                copy.deepcopy(sample_dict), crop_top_frac, crop_left_frac,
                crop_h_frac, crop_w_frac, align=self.spatial_align,
            )
            if do_flip:
                cropped_h5 = _flip_h5_sample_dict(cropped_h5)

            # Uniform resize: all modalities to the same target size
            cropped_h5 = _resize_h5_sample_dict_uniform(
                cropped_h5, ratio, global_crops_size, align=self.spatial_align,
            )

            if self.debug_crop_dims:
                gc_shape = aug_output["global_crops"][crop_i].shape
                logger.info(
                    f"[L2 global[{crop_i}]] H5=({h5_t},{h5_l},{h5_ch},{h5_cw}) flip={do_flip}"
                )
                logger.info(f"  HR -> ({gc_shape[1]},{gc_shape[2]})")
                for k, v in cropped_h5.items():
                    if v is not None and isinstance(v, np.ndarray) and k not in _NON_SPATIAL_KEYS and v.ndim >= 2:
                        logger.info(f"  H5/{k} -> ({v.shape[0]},{v.shape[1]})")

            modalities, metadata = self._process_olmoearth_modalities(
                cropped_h5, missing_timesteps_masks
            )

            h5_olmoearth_crops.append({
                "modalities": modalities,
                "metadata": metadata,
            })

        return {
            "global_crops": aug_output["global_crops"],
            "local_crops": aug_output["local_crops"],
            "h5_olmoearth_crops": h5_olmoearth_crops,
            "hr_data_start_time": data_start_time,
            **(
                {
                    "hr_target_images": target_arrays,
                    "hr_target_start_times": target_start_times,
                }
                if self.return_hr_targets
                else {}
            ),
        }

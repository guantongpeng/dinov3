# DINOv3 HR + H5 多模态遥感数据支持

## 一、整体架构

DINOv3 在 stage 2 训练中同时处理两类遥感数据：

| 数据类型 | 存储格式 | 分辨率 | 用途 |
|----------|----------|--------|------|
| **HR 图像** | TIFF（3 或 4 通道 RGBNIR） | 高分辨率（米级） | DINO 自监督学习输入 |
| **H5 多模态** | HDF5（sentinel2/sentinel1/landsat 等） | 低分辨率（十米级） | 输入冻结的 OLMoEarth 模型提取多模态嵌入 |

两条数据管线通过 `hr_h5_resolution_ratio`（HR 像素 / H5 像素的整数比）建立空间对应关系，确保 HR 和 H5 始终覆盖同一物理区域。

---

## 二、数据目录结构

```
<dataset_root>/
  2024/
    H5/
      sample_0.h5          # 低分辨率多模态数据
      sample_1.h5
    HR/
      sample_0/
        T0.tif              # 高分辨率多时序图像
        T1.tif
      sample_1/
        T0.tif
      meta/
        sample_0.csv        # TIFF 元数据（imageidx → start_time）
        sample_1.csv
  2025/
    H5/...
    HR/...
```

CSV 元数据格式：

| crs | col | row | tile_time | imageidx | start_time | end_time |
|-----|-----|-----|-----------|----------|------------|----------|
| EPSG:4326 | 0 | 0 | 2024-01-15 | T0 | 2024-01-15T00:00:00 | ... |

`start_time` 被解析为 `YYYY-MM` 格式，用于后续的时间条件重建。

---

## 三、数据集索引

数据集初始化时通过 `_discover_samples()` 扫描目录结构，支持三种模式：

| 模式 | 触发条件 | 说明 |
|------|----------|------|
| **JSON 索引** | `root` 为 `.json` 文件 | 由 `build_dataset_index.py` 预构建，避免每次启动扫描 |
| **年份组织** | `root` 下有 `YYYY/` 子目录 | 自动扫描 `H5/`、`HR/`、`HR/meta/` |
| **文件列表** | `root` 为 `.txt/.lst/.csv` | 每行一个 H5 路径，HR 数据从 `hr_data_dir` 查找 |

每个 `(年份, sample_id)` 对作为一个独立训练样本，使得同一地点的多年数据可独立采样。

**JSON 索引预构建**（`build_dataset_index.py`）：

```bash
python build_dataset_index.py <dataset_root> -o dataset_index.json
```

输出格式：

```json
{
  "dataset_root": "/path/to/root",
  "samples": [{
    "sample_id": "sample_0",
    "years": {
      "2024": {
        "h5_path": "2024/H5/sample_0.h5",
        "tif_info": [
          {"tif_path": "2024/HR/sample_0/T0.tif", "start_time": "2024-01"},
          {"tif_path": "2024/HR/sample_0/T1.tif", "start_time": "2024-03"}
        ],
        "hr_dims": [2048, 2048, 4]
      }
    }
  }]
}
```

---

## 四、HR（TIFF）数据管线

HR 管线负责将高分辨率遥感图像转为 DINO 训练所需的增强 crops。

### 4.1 TIFF 读取：`_read_tif_crop()`

使用 `rasterio` 的 Window 读取机制，只加载裁剪区域而非常规全图加载，大幅降低 IO 开销：

```python
with rasterio.open(tif_path) as src:
    window = Window(col_off=left, row_off=top, width=crop_w, height=crop_h)
    data = src.read(window=window)  # (C, H, W) → transpose → (H, W, C)
```

**数据类型归一化**：
- `uint16/int16` → 除以 256 转为 `uint8`
- `float32/float64` → 若值域 [0,1] 则乘 255 转为 `uint8`

### 4.2 转为 Tensor：`_hr_to_tensor()`

```
(H, W, C) uint8  ──→  (C, H, W) float32 [0, 1]
```

通道数处理：
- 若实际通道 > 配置通道：仅保留前 n_channels
- 若实际通道 < 配置通道：补零填充

### 4.3 两级空间裁剪

为确保 HR 与 H5 始终覆盖同一物理区域，裁剪分两级进行：

#### Level 1（数据集层）：H5 参考空间随机裁剪

裁剪参数先在 **H5 参考空间**（HR 尺寸 / ratio）中生成，对齐到 `spatial_align` 的倍数，再通过整数乘法映射回 HR：

```
h5_ref_h = hr_h // ratio
h5_ref_w = hr_w // ratio
↓
H5 参考空间中随机裁剪 (top, left, crop_h, crop_w)，对齐到 spatial_align
↓
hr_top  = h5_top  * ratio    （精确整数）
hr_left = h5_left * ratio
hr_crop_h = h5_crop_h * ratio
hr_crop_w = h5_crop_w * ratio
```

**为什么在 H5 参考空间采样？** 因为 OLMoEarth 的 patchify 要求 H5 空间尺寸是 `patch_size` 的倍数（通常 2-4），在 H5 参考空间对齐保证了兼容性。

裁剪比例范围由 `olmoearth.hr_crop_scale` 配置，例如 `[0.1, 0.8]` 表示裁剪 10% 到 80% 的区域。

#### Level 2（增强层）：DataAugmentationDINOMultiChannel

Level 1 裁剪后的 HR 图像（`(C, H', W')` float tensor）送入多通道 DINO 增强器进行全局/局部 crop 增强。**关键设计**：增强器的几何裁剪参数先在 H5 参考空间中采样，通过整数乘法映射到 HR——与 Level 1 的思路一致，保证像素级精确对齐。

### 4.4 多通道数据增强：`DataAugmentationDINOMultiChannel`

**与标准 DataAugmentationDINO 的核心差异**：

| 特性 | DataAugmentationDINO | DataAugmentationDINOMultiChannel |
|------|---------------------|----------------------------------|
| 输入 | PIL Image（3 通道） | `(C, H, W)` float tensor（3 或 4 通道） |
| 颜色抖动 | RGB 三通道统一处理 | **RGB 做 color jitter + grayscale + solarize；NIR 仅做 brightness/contrast** |
| 归一化 | ImageNet 3 通道均值/方差 | RGB 用 ImageNet 参数，NIR 用独立参数（默认 mean=0.5, std=0.25） |
| 几何裁剪 | 直接在输入 tensor 上采样 | **在 H5 参考空间中采样，整数乘法映射到 HR** |
| 输出 | crops + offsets | crops + **h5_crop_params**（H5 参考空间坐标）+ **global_crop_flips** |

#### 几何裁剪函数：`_apply_h5_aligned_geometric_crop()`

```
1. 从 HR 尺寸推导 H5 参考尺寸：
   h5_ref_h = hr_h // ratio, h5_ref_w = hr_w // ratio

2. 在 H5 参考空间中采样 RandomResizedCrop 参数：
   h5_top, h5_left, h5_crop_h, h5_crop_w

3. 对齐到 spatial_align 倍数

4. 整数乘法映射到 HR 像素坐标：
   hr_top = h5_top * ratio, hr_crop_h = h5_crop_h * ratio

5. 在 HR tensor 上执行 resized_crop → (C, global_crops_size, global_crops_size)
```

返回裁剪后的 HR tensor 和 `(h5_top, h5_left, h5_crop_h, h5_crop_w)` 供 H5 管线使用。

#### 增强流水线（以 4 通道 RGBNIR 为例）

```
Level 1 裁剪后的 (4, H', W') float tensor
│
├─ 全局 crop 1:
│   ├─ _apply_h5_aligned_geometric_crop → (4, 224, 224) + h5_crop_params
│   ├─ 随机水平翻转（记录 flip 状态）
│   ├─ RGB[:3]: ColorJitter(0.4,0.4,0.2,0.1) + RandomGrayscale(0.2)
│   ├─ NIR[3:4]: ColorJitter(brightness=0.4, contrast=0.4)  # 无饱和度/色调
│   ├─ GaussianBlur(p=1.0) + Normalize
│   └─ → (4, 224, 224) 归一化后的 global crop
│
├─ 全局 crop 2:
│   ├─ 同上几何裁剪（独立参数）
│   ├─ GaussianBlur(p=0.1) + RandomSolarize(0.2) + Normalize
│   └─ → (4, 224, 224) 归一化后的 global crop
│
├─ 局部 crop × 8:
│   ├─ _apply_geometric_crop → (4, 96, 96) + 分数坐标
│   ├─ 随机水平翻转
│   ├─ RGB[:3]: ColorJitter + Grayscale
│   ├─ NIR[3:4]: brightness/contrast jitter
│   ├─ GaussianBlur(p=0.5) + Normalize
│   └─ → (4, 96, 96) 归一化后的 local crop
│
└─ 输出:
    - global_crops: list[2 × (C, gH, gW)]
    - local_crops: list[8 × (C, lH, lW)]
    - global_crop_flips: [bool, bool]
    - h5_crop_params: [(top, left, h, w), ...]  ← H5 参考空间坐标
```

**为什么 NIR 通道不做颜色抖动？** 颜色抖动中的 saturation/hue 操作假设输入是 RGB 色彩空间，对红外波段没有物理意义。NIR 仅做亮度/对比度调整，模拟不同光照条件。

### 4.5 HR Target 图像

当 `olmoearth.return_hr_targets=true` 时，非选中作为 DINO 输入的 TIFF（同一地点、不同时间）也会被读取：

- 所有 target 被 resize 到与 global crop 相同尺寸
- 通过 `hr_target_masks` 布尔掩码标记哪些是有效 target（不同样本 target 数量可能不同）
- `start_time` 信息保留供时间条件重建使用

---

## 五、H5 多模态管线

H5 管线负责读取低分辨率多模态遥感数据，经裁剪、填充、归一化后转为 OLMoEarth 模型的输入。

### 5.1 H5 文件结构

每个 H5 文件包含以下数据集：

| 数据集 | 维度 | 说明 |
|--------|------|------|
| `sentinel2_l2a` | (H, W, T, C) | Sentinel-2 多光谱（空间+时间+波段） |
| `sentinel1` | (H, W, T, C) | Sentinel-1 SAR 数据 |
| `landsat` | (H, W, T, C) | Landsat 多光谱 |
| `timestamps` | (T, 3) | 时间戳（年, 月, 日） |
| `latlon` | (2,) | 经纬度坐标 |
| `missing_timesteps_masks/*` | (T,) | 每个模态的时间步缺失掩码 |

### 5.2 读取与填充：`_read_h5_file()` + 填充管线

```
H5 文件
  │
  ├─ h5py.File(h5_path, "r")
  │
  ├─ 分离数据层和掩码层：
  │   - 普通数据集 → sample_dict
  │   - missing_timesteps_masks/ → missing_timesteps_masks
  │
  ├─ _pad_timestamps()：将时间序列 pad 到 max_sequence_length（edge 模式）
  │
  ├─ _fill_sample_with_missing_values()：
  │   - 完全缺失的模态 → 生成全 MISSING_VALUE（-99999）占位
  │   - 部分缺失的时间步 → 用 MISSING_VALUE 填充
  │
  └─ _normalize_sample()：
      对每个模态调用 OLMoEarth Normalizer：
      - 先尝试 COMPUTED 策略（基于数据统计）
      - 失败则回退到 PREDEFINED 策略（预定义参数）
      - 缺失值位置保持 MISSING_VALUE 不变
```

**归一化策略**（`_normalize_sample`）：

```python
for modality_name in sample_dict:
    mod_spec = Modality.get(modality_name)       # OLMoEarth 模态元信息
    try:
        normalized = normalizer_computed.normalize(mod_spec, data)   # 基于统计
    except Exception:
        normalized = normalizer_predefined.normalize(mod_spec, data)  # 预定义参数
    # 将缺失值位置还原为 MISSING_VALUE
    sample_dict[key] = np.where(missing_mask, MISSING_VALUE, normalized).astype(np.float32)
```

### 5.3 空间裁剪：跟踪 HR 裁剪

H5 空间裁剪严格跟随 HR 裁剪，保证两者覆盖同一物理区域。

#### Level 1 裁剪（数据集层）

使用 HR Level 1 裁剪的分数坐标，按相同比例裁剪所有 H5 空间模态：

```python
# HR Level 1 裁剪参数（已在 H5 参考空间中采样）
top_frac  = h5_top  / ref_h      # 例如 0.2
left_frac = h5_left / ref_w      # 例如 0.1
h_frac    = h5_crop_h / ref_h    # 例如 0.6
w_frac    = h5_crop_w / ref_w    # 例如 0.7

# 对每个 H5 空间模态应用相同分数坐标
for modality in spatial_modalities:
    m_top    = int(top_frac  * mod_h)
    m_left   = int(left_frac * mod_w)
    m_crop_h = align(int(h_frac * mod_h), spatial_align)
    m_crop_w = align(int(w_frac * mod_w), spatial_align)
    cropped = modality[m_top:m_top+m_crop_h, m_left:m_left+m_crop_w]
```

**为什么用分数坐标而不是像素坐标？** 不同 H5 模态可能因网格偏移有微小尺寸差异，分数坐标确保裁剪的物理区域一致。

#### Level 2 裁剪（增强层）

每个 DINO global crop 在 H5 参考空间中生成独立的随机裁剪参数（通过 `_apply_h5_aligned_geometric_crop`），H5 数据据此再做一次空间裁剪：

```
Level 1 裁剪后的 H5 数据
  │
  ├─ 对每个 global crop i:
  │   ├─ 从 aug_output["h5_crop_params"][i] 获取 (h5_t, h5_l, h5_ch, h5_cw)
  │   ├─ 转为 Level 1 裁剪空间的分数坐标
  │   ├─ _crop_h5_sample_dict_fractional() → 裁剪所有空间模态
  │   ├─ 若 global_crop_flips[i] = True → _flip_h5_sample_dict() 水平翻转
  │   └─ _resize_h5_sample_dict_uniform() → 统一尺寸
  │
  └─ 每个 global crop 产生一对 "modalities" + "metadata"
```

**水平翻转变换**（`_flip_h5_sample_dict`）：

对 H5 空间模态沿 W 轴翻转（`np.flip(axis=1)`），与 HR 的翻转保持一致。非空间模态（timestamps, latlon）不变。翻转后的数组使用 `.copy()` 确保内存连续（OLMoEarth 推理需要）。

**统一尺寸变换**（`_resize_h5_sample_dict_uniform`）：

Level 2 裁剪后，不同 H5 模态的尺寸可能略有不同（因为对齐取整）。此函数将所有空间模态 resize 到统一目标尺寸：

```
target = align(global_crops_size / hr_h5_resolution_ratio, spatial_align)

例如: global_crops_size=480, ratio=40 → target=12

4D 张量 (H,W,T,C):
  → permute(2,3,0,1) → reshape(T*C, H, W) → bilinear interpolate → reshape back

3D 张量 (H,W,C):
  → permute(2,0,1) → bilinear interpolate → permute back
```

### 5.4 转为 Tensor：`_process_olmoearth_modalities()`

将 numpy 数组转为 torch tensor：

```python
modality_tensors[key] = torch.tensor(val, dtype=torch.float32)   # 模态数据
metadata_tensors["timestamps"] = torch.tensor(val, dtype=torch.long)  # 时间戳
metadata_tensors["latlon"] = torch.tensor(val, dtype=torch.float32)    # 经纬度
```

---

## 六、完整 `__getitem__` 数据流

```python
def __getitem__(self, index):
    # ═══════════════════════════════════════════════════════════════
    # 1. 读取 H5 文件
    # ═══════════════════════════════════════════════════════════════
    sample_dict_raw, missing_timesteps_masks = _read_h5_file(h5_path, all_modalities)

    # ═══════════════════════════════════════════════════════════════
    # 2. Level 1 裁剪：H5 参考空间采样 → HR + H5 同步裁剪
    # ═══════════════════════════════════════════════════════════════
    ref_h, ref_w = hr_h // ratio, hr_w // ratio
    h5_top, h5_left, h5_crop_h, h5_crop_w = _random_crop_params_aligned(
        ref_h, ref_w, hr_crop_scale, align=spatial_align
    )
    # → 分数坐标
    top_frac, left_frac = h5_top/ref_h, h5_left/ref_w
    h_frac, w_frac = h5_crop_h/ref_h, h5_crop_w/ref_w
    # → HR 像素坐标（精确整数）
    hr_top, hr_left = h5_top*ratio, h5_left*ratio
    hr_crop_h, hr_crop_w = h5_crop_h*ratio, h5_crop_w*ratio

    # ═══════════════════════════════════════════════════════════════
    # 3. 读取 HR TIFF（data + targets）
    # ═══════════════════════════════════════════════════════════════
    for each TIFF:
        arr = _read_tif_crop(tif_path, hr_top, hr_left, hr_crop_h, hr_crop_w)
        # 随机选一张作为 DINO 输入，其余作为 target（可选）

    # ═══════════════════════════════════════════════════════════════
    # 4. H5 Level 1 裁剪
    # ═══════════════════════════════════════════════════════════════
    sample_dict = _crop_h5_sample_dict_fractional(
        sample_dict_raw, top_frac, left_frac, h_frac, w_frac
    )

    # ═══════════════════════════════════════════════════════════════
    # 5. Level 2: DINO 增强 + H5 对齐裁剪
    # ═══════════════════════════════════════════════════════════════
    hr_tensor = _hr_to_tensor(data_array)          # (C,H,W) uint8 → float [0,1]
    aug_output = self.transform(hr_tensor)         # DINO 增强

    for each global crop:
        # 5a. 获取 DINO 增强的 H5 参考空间裁剪参数
        h5_t, h5_l, h5_ch, h5_cw = aug_output["h5_crop_params"][crop_i]
        do_flip = aug_output["global_crop_flips"][crop_i]

        # 5b. H5 Level 2 裁剪（跟踪 HR）
        cropped_h5 = _crop_h5_sample_dict_fractional(
            sample_dict, h5_t/h5_L1_h, h5_l/h5_L1_w, h5_ch/h5_L1_h, h5_cw/h5_L1_w
        )
        if do_flip:
            cropped_h5 = _flip_h5_sample_dict(cropped_h5)

        # 5c. 统一尺寸 + 归一化 + 转 tensor
        cropped_h5 = _resize_h5_sample_dict_uniform(cropped_h5, ratio, global_crops_size)
        modalities, metadata = _process_olmoearth_modalities(
            cropped_h5, missing_timesteps_masks
        )

    # ═══════════════════════════════════════════════════════════════
    # 6. 返回
    # ═══════════════════════════════════════════════════════════════
    return {
        "global_crops": aug_output["global_crops"],           # HR: [2 × (C,224,224)]
        "local_crops": aug_output["local_crops"],             # HR: [8 × (C,96,96)]
        "h5_olmoearth_crops": [                               # H5: per global crop
            {"modalities": {...}, "metadata": {...}},          # crop 0
            {"modalities": {...}, "metadata": {...}},          # crop 1
        ],
        "hr_data_start_time": "2024-01",                       # HR 数据时间
        "hr_target_images": [...],                              # 其他时序 TIFF
        "hr_target_start_times": [...],
    }
```

---

## 七、Collation（Batch 组装）

`collate_h5_olmoearth_and_cast()` 将多个样本合并为一个训练 batch，分四个步骤：

### 步骤 1：合并 DINO crops

```python
collated_global_crops = torch.stack(...)  # [2*B, C, gH, gW]
collated_local_crops = torch.stack(...)   # [8*B, C, lH, lW]
```

所有样本的 HR crops 尺寸一致（由 DINO 增强保证），可直接 stack。

### 步骤 2：生成 iBOT 掩码

与标准 DINOv3 相同的 block-wise masking 逻辑，为每个 global crop 生成随机的 patch 掩码用于 masked image modeling。

### 步骤 3：合并 H5 多模态数据（per global crop）

H5 模态数据的空间尺寸可能不一致（不同样本 Level 1 裁剪后尺寸不同），需要插值对齐：

```python
for key in modality_keys:
    tensors = [s["modalities"][key] for s in samples_list]
    shapes = [t.shape for t in tensors]

    if all same shape:
        stack directly

    elif ndim == 4:  # (H, W, T, C)
        max_h, max_w = max size, aligned to spatial_align
        for each tensor with different size:
            reshape to (T*C, H, W) → bilinear interpolate → reshape back
        stack

    elif ndim == 3:  # (H, W, C)
        max_h, max_w = max size, aligned to spatial_align
        for each tensor with different size:
            permute to (C, H, W) → bilinear interpolate → permute back
        stack
```

### 步骤 4：合并 HR target 图像

不同样本的 target 时序长度不同，使用零填充 + 布尔掩码：

```python
max_targets = max(len(targets) for targets in all_target_tensors)

for each sample's targets:
    if len(targets) < max_targets:
        pad with zero tensors                          # 填充
    mask[:len(targets)] = True                         # 标记有效位

hr_target_images_batch = torch.stack(padded_targets)   # [B, max_T, C, H, W]
hr_target_masks_batch = torch.stack(target_masks)       # [B, max_T]
```

---

## 八、训练集成

`DINOv3WithOLMoEarth.forward_backward()` 中的处理：

```python
def forward_backward(self, data, *, teacher_temp, iteration=0):
    # 1. 提取 H5 数据（SSLMetaArch 不需要）
    olmoearth_modalities_list = data.pop("olmoearth_modalities", None)   # list of dicts
    olmoearth_metadata_list = data.pop("olmoearth_metadata", None)

    # 2. 清理 HR target 字段（SSLMetaArch 不需要）
    data.pop("hr_target_images", None)
    data.pop("hr_target_masks", None)
    data.pop("hr_target_start_times", None)
    data.pop("hr_data_start_time", None)

    # 3. 标准 DINO V3 forward-backward（仅使用 HR 图像）
    total_loss, dino_metrics = self.dino_model.forward_backward(
        data, teacher_temp=teacher_temp, iteration=iteration
    )

    # 4. OLMoEarth 推理（冻结，无梯度）per global crop
    for crop_idx, (modalities, metadata) in enumerate(
        zip(olmoearth_modalities_list, olmoearth_metadata_list)
    ):
        olmoearth_embeddings = run_olmoearth_inference(
            self.olmoearth_model, modalities, metadata,
            patch_size=self.olmoearth_patch_size,
        )
        del olmoearth_embeddings  # 当前阶段仅验证管道，后续接入融合 loss

    return total_loss, metrics_dict
```

**OLMoEarth 推理**（`run_olmoearth_inference()`）：

1. 构建 `batch_dict`（timestamps, latlon, 各模态数据）和 `mask_dict`（per-modal 掩码）
2. 检查每个 batch item：若某模态数据全为 MISSING_VALUE，其掩码标记为 MISSING（OLMoEarth 内部跳过该样本）
3. 构建 `MaskedOlmoEarthSample`，调用 `model.encoder(masked_sample, fast_pass=False, patch_size=patch_size)`
4. 提取 `tokens_and_masks` 中的各模态嵌入

---

## 九、关键配置参数

```yaml
olmoearth:
  enabled: true
  model_id_or_path: "OlmoEarth-v1-Nano"    # OLMoEarth 模型 ID 或本地路径
  max_sequence_length: 12                   # 时序最大长度
  missing_value: -99999                      # 缺失值填充
  patch_size: 2                              # OLMoEarth patchify 步长
  spatial_align: 4                           # H5 空间尺寸对齐倍数
  hr_h5_resolution_ratio: 40                 # HR/H5 像素比例（整数）
  hr_crop_scale: [0.1, 0.8]                  # Level 1 裁剪比例范围
  return_hr_targets: false                   # 是否返回 HR target（用于重建 loss）
  debug_crop_dims: false                     # 是否打印各级裁剪尺寸

crops:
  n_channels: 4                              # 3=RGB, 4=RGBNIR
  nir_mean: 0.5                              # NIR 归一化均值
  nir_std: 0.25                              # NIR 归一化方差
  rgb_mean: [0.485, 0.456, 0.406]           # RGB 归一化（ImageNet）
  rgb_std: [0.229, 0.224, 0.225]
  global_crops_size: 480
  local_crops_size: 200
  global_crops_scale: [0.32, 1.0]
  local_crops_scale: [0.05, 0.32]
  local_crops_number: 8

student:
  in_chans: 4                                # ViT 输入通道数（需匹配 n_channels）

teacher:
  in_chans: 4                                # 教师网络同样 4 通道
```

---

## 十、HR 与 H5 变换总结

```
                        HR (高分辨率 TIFF)                  H5 (低分辨率多模态)
                        ════════════════════                ════════════════════
原始数据                  (H, W, C) uint8                    (H_h5, W_h5, T, C) mixed dtypes
                          3 或 4 通道                         多模态时间序列
                              │                                    │
Level 1 裁剪               H5 参考空间采样                         分数坐标同步裁剪
                          × ratio 映射到 HR                      _crop_h5_sample_dict_fractional()
                          _read_tif_crop()                      对齐到 spatial_align
                              │                                    │
数据类型转换              _hr_to_tensor()                       时间步填充/缺失值填充
                          (C,H,W) float [0,1]                   _normalize_sample()
                              │                                    │
Level 2 裁剪               _apply_h5_aligned_                    分数坐标同步裁剪
(DINO 增强)               geometric_crop()                      _crop_h5_sample_dict_fractional()
                          × ratio 映射到 HR                      水平翻转跟踪 global_crop_flips
                          水平翻转（记录状态）                      统一尺寸 resize
                              │                                    │
颜色/图像增强             RGB: ColorJitter+Saturate+Gray       -
                          NIR: Brightness/Contrast only         -
                          GaussianBlur + Solarize               -
                              │                                    │
归一化                    RGB: ImageNet mean/std                 OLMoEarth Normalizer
                          NIR: 独立 mean/std                     (COMPUTED→PREDEFINED)
                              │                                    │
输出                      [2×(C,224,224)] global crops          [2× dict of tensors]
                          [8×(C,96,96)] local crops              per global crop:
                          [bool,bool] flip states                  - modalities (torch.float32)
                          [(t,l,h,w),...] h5_crop_params          - metadata (timestamps/latlon)
                              │                                    │
Batch 组装                torch.stack                           变尺寸 bilinear 插值 → stack
(collate)                 尺寸一致直接 stack                      4D/3D 分别处理
                              │                                    │
训练                      DINO/iBOT forward-backward            冻结 OLMoEarth encoder 推理
                          (SSLMetaArch)                          → multi-modal embeddings
```

---

## 十一、文件清单

| 文件 | 行数 | 作用 |
|------|------|------|
| `dinov3/data/datasets/h5_olmoearth.py` | 982 | H5 + TIFF 数据集，含读取、裁剪、归一化、增强全流程 |
| `dinov3/data/augmentations.py` | 511 | 多通道 DINO 增强器 `DataAugmentationDINOMultiChannel` |
| `dinov3/data/collate.py` | 326 | H5 OLMoEarth batch 组装 `collate_h5_olmoearth_and_cast` |
| `dinov3/data/loaders.py` | 250 | 数据集注册与 DataLoader 构建 |
| `dinov3/data/build_dataset_index.py` | 188 | JSON 索引预构建脚本 |
| `dinov3/train/dino_with_olmoearth.py` | 166 | `DINOv3WithOLMoEarth` 联合训练模型 |
| `dinov3/train/olmoearth_inference.py` | 130 | OLMoEarth 模型加载 + 推理 |
| `dinov3/train/train.py` | 747 | 训练入口，含 `build_h5_olmoearth_data_loader_from_cfg` |
| `dinov3/configs/train/dinov3_olmoearth.yaml` | 222 | OLMoEarth 训练配置 |
| `dinov3/models/__init__.py` | — | ViT 构建新增 `in_chans` 参数 |
| `tests/h5_data_maker.py` | 162 | 合成 H5 测试数据生成 |
| `tests/test_h5_olmoearth_dataloader.py` | 327 | 数据集单元测试 |
| `tests/test_dinov3witholmoearth.py` | 78 | 端到端管道测试 |

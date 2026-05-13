# DINO V3 + OLMoEarth 数据预处理维度变化详细过程

> 基于 YAML 配置 `dinov3_olmoearth.yaml` 中的参数，以具体数值举例说明。
> 配置值：`hr_h5_resolution_ratio=20`, `global_crops_size=224`, `local_crops_size=96`,
> `n_channels=4`, `spatial_align=2`, `hr_crop_scale=(0.1, 0.2)`, `olmoearth.patch_size=2`

---

## 0. 原始数据

| 数据来源 | 维度 | 说明 |
|---------|------|------|
| HR TIFF | `(H_hr, W_hr, C)`, e.g. `(2000, 2000, 4)` uint8 | 高分辨率影像，C=3(RGB) 或 4(RGBNIR) |
| H5 sentinel2_l2a | `(H_h5, W_h5, T, C)`, e.g. `(100, 100, 12, 12)` float32 | 多时相多光谱，T=时间步，C=波段数 |
| H5 sentinel1 | `(H_h5, W_h5, T, C)`, e.g. `(100, 100, 12, 2)` float32 | SAR 数据 |
| H5 landsat | `(H_h5, W_h5, T, C)`, e.g. `(100, 100, 12, 6)` float32 | Landsat 数据 |
| H5 timestamps | `(T, 2)` int64 | 每个时间步的 year + month |
| H5 latlon | `(2,)` float64 | 经纬度 |

**关键前提**：H5 中各模态的存储分辨率不一定等于真实物理分辨率。所有空间对齐计算均基于 YAML 配置的 `hr_h5_resolution_ratio`，而非从数据 shape 反推。

---

## 1. Level 1 裁剪：H5 参考空间 → HR 整数映射

### 1.1 计算参考空间维度

```
ref_h = H_hr // hr_h5_resolution_ratio
ref_w = W_hr // hr_h5_resolution_ratio

例: 2000 // 20 = 100,  2000 // 20 = 100
```

### 1.2 在 H5 参考空间生成对齐的随机裁剪参数

```
scale = random.uniform(hr_crop_scale[0], hr_crop_scale[1])  # e.g. 0.15
h5_crop_h = max(spatial_align, int(ref_h * scale) // spatial_align * spatial_align)
h5_crop_w = max(spatial_align, int(ref_w * scale) // spatial_align * spatial_align)
h5_top, h5_left = aligned random positions

例: scale=0.15 → h5_crop_h = int(100*0.15)//2*2 = 14
    h5_top = 4, h5_left = 8, h5_crop_h = 14, h5_crop_w = 16
```

### 1.3 分数坐标（用于 H5 模态裁剪）

```
top_frac  = h5_top   / ref_h    # 4/100  = 0.04
left_frac = h5_left  / ref_w    # 8/100  = 0.08
h_frac    = h5_crop_h / ref_h   # 14/100 = 0.14
w_frac    = h5_crop_w / ref_w   # 16/100 = 0.16
```

### 1.4 HR 坐标（精确整数乘法）

```
hr_top    = h5_top    * hr_h5_resolution_ratio    # 4  * 20 = 80
hr_left   = h5_left   * hr_h5_resolution_ratio    # 8  * 20 = 160
hr_crop_h = h5_crop_h * hr_h5_resolution_ratio    # 14 * 20 = 280
hr_crop_w = h5_crop_w * hr_h5_resolution_ratio    # 16 * 20 = 320
```

### 1.5 维度变化

| 数据 | Level 1 前 | Level 1 后 | 变换方式 |
|------|-----------|-----------|---------|
| HR TIFF (data) | `(2000, 2000, 4)` uint8 | `(280, 320, 4)` uint8 | `_read_tif_crop` 窗口读取 |
| HR TIFF → tensor | — | `(4, 280, 320)` float32 [0,1] | `_hr_to_tensor` |
| H5 sentinel2_l2a | `(100, 100, 12, 12)` | `(14, 16, 12, 12)` | 同分数裁剪 |
| H5 sentinel1 | `(100, 100, 12, 2)` | `(14, 16, 12, 2)` | 同分数裁剪 |
| H5 landsat | `(100, 100, 12, 6)` | `(14, 16, 12, 6)` | 同分数裁剪 |
| H5 timestamps | `(12, 2)` | `(12, 2)` | 不变 |
| H5 latlon | `(2,)` | `(2,)` | 不变 |

**对齐保证**：HR 裁剪区域 `(80:360, 160:480)` 和 H5 裁剪区域 `(4:18, 8:24)` 对应同一物理区域。

---

## 2. Level 2 裁剪：DINO augmentation（HR 空间 + H5 参考空间对齐）

### 2.1 Augmentation 输入

```
hr_tensor: (4, 280, 320) float32 [0,1]  — Level 1 裁剪后的 HR 数据
```

### 2.2 `_apply_h5_aligned_geometric_crop` 流程

```
ratio = hr_h5_resolution_ratio = 20
h5_ref_h = hr_tensor.shape[1] // ratio = 280 // 20 = 14
h5_ref_w = hr_tensor.shape[2] // ratio = 320 // 20 = 16

在 (14, 16) 的 H5 参考空间采样 RandomResizedCrop.get_params:
  h5_top=2, h5_left=4, h5_crop_h=8, h5_crop_w=10  (对齐到 spatial_align=2)

映射到 HR（精确整数乘法）:
  hr_top    = 2  * 20 = 40
  hr_left   = 4  * 20 = 80
  hr_crop_h = 8  * 20 = 160
  hr_crop_w = 10 * 20 = 200

HR resized_crop → (4, 224, 224)
```

### 2.3 HR global crop 增强（两个 global crop）

```
Global crop 0:
  RandomResizedCrop → (4, 224, 224)
  hflip (50%) → (4, 224, 224)
  ColorJitter (RGB only) → (4, 224, 224)
  NIR Brightness/Contrast Jitter → (4, 224, 224)
  GaussianBlur(p=1.0) → (4, 224, 224)
  Normalize(RGB: ImageNet, NIR: mean=0.5, std=0.25) → (4, 224, 224)

Global crop 1:
  同上，但 GaussianBlur(p=0.1) + Solarize(RGB+NIR, p=0.2)
```

### 2.4 HR local crop 增强（8 个）

```
Local crop:
  RandomResizedCrop(scale=(0.05, 0.32)) → (4, 96, 96)
  hflip → (4, 96, 96)
  ColorJitter (RGB) + NIR Jitter → (4, 96, 96)
  GaussianBlur(p=0.5) → (4, 96, 96)
  Normalize → (4, 96, 96)
```

### 2.5 Augmentation 输出

```
{
  "global_crops": [(4, 224, 224), (4, 224, 224)],    # 2 个
  "local_crops":  [(4, 96, 96) × 8],                  # 8 个
  "h5_crop_params": [(2, 4, 8, 10), ...],             # H5 参考空间裁剪参数
  "global_crop_flips": [True, False],                  # 是否水平翻转
  "hr_h5_resolution_ratio": 20,
}
```

---

## 3. Level 2 H5 裁剪 + 统一 resize

### 3.1 将 H5 参考空间坐标转为 L1-cropped 空间分数

```
hr_L1_h, hr_L1_w = 280, 320
h5_L1_h, h5_L1_w = 280 // 20 = 14, 320 // 20 = 16

对于 crop_params (h5_t=2, h5_l=4, h5_ch=8, h5_cw=10):
  crop_top_frac  = 2  / 14 = 0.143
  crop_left_frac = 4  / 16 = 0.250
  crop_h_frac    = 8  / 14 = 0.571
  crop_w_frac    = 10 / 16 = 0.625
```

### 3.2 H5 模态裁剪（同一组分数坐标）

| 数据 | Level 1 后 | Level 2 裁剪后 | 说明 |
|------|-----------|--------------|------|
| H5 sentinel2_l2a | `(14, 16, 12, 12)` | `(8, 10, 12, 12)` | `int(0.143*14)=2, int(0.571*14)//2*2=8` 等 |
| H5 sentinel1 | `(14, 16, 12, 2)` | `(8, 10, 12, 2)` | 同上 |
| H5 landsat | `(14, 16, 12, 6)` | `(8, 10, 12, 6)` | 同上 |
| timestamps | `(12, 2)` | `(12, 2)` | 不变 |
| latlon | `(2,)` | `(2,)` | 不变 |

**若 flip=True**：所有空间模态沿 W 轴翻转。

### 3.3 统一 resize（`_resize_h5_sample_dict_uniform`）

```
target_h = max(spatial_align, (global_crops_size // ratio) // spatial_align * spatial_align)
         = max(2, (224 // 20) // 2 * 2)
         = max(2, 11 // 2 * 2)
         = max(2, 10)
         = 10

target_w = 10  (同理)
```

| 数据 | Level 2 裁剪后 | resize 后 | 说明 |
|------|--------------|----------|------|
| sentinel2_l2a | `(8, 10, 12, 12)` | `(10, 10, 12, 12)` | bilinear 插值 (H,W) |
| sentinel1 | `(8, 10, 12, 2)` | `(10, 10, 12, 2)` | bilinear 插值 (H,W) |
| landsat | `(8, 10, 12, 6)` | `(10, 10, 12, 6)` | bilinear 插值 (H,W) |

**所有模态统一为 `(10, 10, ...)`**，保证跨模态 token 级对齐。

---

## 4. H5 模态预处理（`_process_olmoearth_modalities`）

### 4.1 时间戳填充

```
timestamps: (T, 2) → 若 T < max_sequence_length, pad 到 (12, 2)
```

### 4.2 缺失值填充

- **完全缺失的模态**：用 `missing_value=-99999` 填充预期形状
- **缺失时间步**：创建 `(H, W, max_T, C)` 全 missing 数组，按 mask 位置填入已有数据

### 4.3 归一化

- 先尝试 `COMPUTED` 策略（基于数据统计），失败则 fallback 到 `PREDEFINED`
- 缺失值位置保持 `-99999` 不变
- 输出 dtype: float32

### 4.4 转 Tensor

| 数据 | numpy shape | tensor shape | dtype |
|------|------------|-------------|-------|
| sentinel2_l2a | `(10, 10, 12, 12)` float32 | `(10, 10, 12, 12)` | float32 |
| sentinel1 | `(10, 10, 12, 2)` float32 | `(10, 10, 12, 2)` | float32 |
| landsat | `(10, 10, 12, 6)` float32 | `(10, 10, 12, 6)` | float32 |
| timestamps | `(12, 2)` int64 | `(12, 2)` | long |
| latlon | `(2,)` float64 | `(2,)` | float32 |

### 4.5 单 sample 最终输出

```python
{
    "global_crops": [          # HR
        (4, 224, 224),        # global crop 0, 归一化 float32
        (4, 224, 224),        # global crop 1
    ],
    "local_crops": [           # HR
        (4, 96, 96) × 8,      # 8 个 local crop
    ],
    "h5_olmoearth_crops": [    # H5, 2 个（对应 2 个 global crop）
        {
            "modalities": {
                "sentinel2_l2a": (10, 10, 12, 12),
                "sentinel1":      (10, 10, 12, 2),
                "landsat":        (10, 10, 12, 6),
            },
            "metadata": {
                "timestamps": (12, 2),
                "latlon":     (2,),
            },
        },
        { ... },               # 第二个 global crop 对应的 H5 数据
    ],
    "hr_data_start_time": "2023-06",
    "hr_target_images": [(280, 320, 4) uint8, ...],  # 其余 TIFF 的 L1 裁剪
    "hr_target_start_times": ["2023-01", ...],
}
```

---

## 5. Collate 阶段（`collate_h5_olmoearth_and_cast`）

batch_size_per_gpu = 2

### 5.1 DINO crops 拼接

| 数据 | 单 sample | collated (B=2) | 说明 |
|------|----------|---------------|------|
| global_crops | `2 × (4, 224, 224)` | `(4, 4, 224, 224)` | `2 crops × 2 samples` |
| local_crops | `8 × (4, 96, 96)` | `(16, 4, 96, 96)` | `8 crops × 2 samples` |

### 5.2 iBOT mask 生成

```
img_size = 224, patch_size = 16
n_tokens = (224 // 16)² = 196
mask_probability = 0.5
mask_ratio = [0.1, 0.5]

collated_masks: (4, 196)     # B=4 (2 global crops × 2 samples)
mask_indices_list: (N_masked,) # 被遮盖的 patch 索引
```

### 5.3 OLMoEarth 模态拼接

由于所有 H5 模态在 resize 后尺寸统一（`(10, 10, ...)`），无需插值对齐，直接 `torch.stack`。

| 数据 | 单 sample | collated (B=2) | 说明 |
|------|----------|---------------|------|
| sentinel2_l2a | `(10, 10, 12, 12)` | `(2, 10, 10, 12, 12)` | 每个 global crop 一组 |
| sentinel1 | `(10, 10, 12, 2)` | `(2, 10, 10, 12, 2)` | |
| landsat | `(10, 10, 12, 6)` | `(2, 10, 10, 12, 6)` | |
| timestamps | `(12, 2)` | `(2, 12, 2)` | |
| latlon | `(2,)` | `(2, 2)` | |

### 5.4 HR target images 拼接

```
target_arrays: [(280, 320, 4) uint8, ...]
  → float32 / 255.0, resize 到 (224, 224)
  → (C, 224, 224) float32
  → pad 到 max_targets 数量, 生成 target_masks
```

### 5.5 最终 batch dict

```python
{
    "collated_global_crops":  (4, 4, 224, 224),        # bf16/fp32
    "collated_local_crops":   (16, 4, 96, 96),          # bf16/fp32
    "collated_masks":         (4, 196),                  # bool
    "mask_indices_list":      (N_masked,),               # long
    "masks_weight":           (N_masked,),               # float
    "upperbound":             int,
    "n_masked_patches":       (1,),                      # long

    # OLMoEarth — list of 2 dicts (per global crop)
    "olmoearth_modalities": [
        {
            "sentinel2_l2a": (2, 10, 10, 12, 12),     # [B, H, W, T, C]
            "sentinel1":      (2, 10, 10, 12, 2),
            "landsat":        (2, 10, 10, 12, 6),
        },
        { ... },  # 第二个 global crop
    ],
    "olmoearth_metadata": [
        {
            "timestamps": (2, 12, 2),                   # [B, T, 2]
            "latlon":     (2, 2),                        # [B, 2]
        },
        { ... },
    ],

    "hr_target_images":       (2, max_T, 4, 224, 224),  # float32
    "hr_target_masks":        (2, max_T),                # bool
    "hr_target_start_times":  [[str, ...], ...],
    "hr_data_start_time":     [str, str],
}
```

---

## 6. 训练前向（`DINOv3WithOLMoEarth.forward_backward`）

### 6.1 数据拆分

```python
# 从 batch pop 出 OLMoEarth 和 HR target 数据
olmoearth_modalities_list = data.pop("olmoearth_modalities")   # list[2] of dicts
olmoearth_metadata_list  = data.pop("olmoearth_metadata")     # list[2] of dicts
data.pop("hr_target_images")
data.pop("hr_target_masks")
data.pop("hr_target_start_times")
data.pop("hr_data_start_time")

# 剩余 data 送入 SSLMetaArch:
#   collated_global_crops, collated_local_crops, collated_masks, ...
```

### 6.2 DINO V3 forward-backward

```
Student ViT:  输入 (4, 4, 224, 224)  → patchify → (4, 196, dim)
Teacher ViT:  输入 (4, 4, 224, 224)  → patchify → (4, 196, dim)
DINO loss + iBOT loss → total_loss
```

### 6.3 OLMoEarth 推理（frozen, no grad）

```python
for crop_idx, (modalities, metadata) in enumerate(zip(olmoearth_modalities_list, olmoearth_metadata_list)):
    embeddings = run_olmoearth_inference(model, modalities, metadata, patch_size=2)

    # 每个模态输出:
    #   e.g. sentinel2_l2a: [B, P_H, P_W, T, Band_Sets, D]
    #   P_H = H // patch_size = 10 // 2 = 5
    #   P_W = W // patch_size = 10 // 2 = 5
    #   → [2, 5, 5, 12, num_band_sets, D]
```

---

## 7. 全链路维度总览

```
原始数据
  HR TIFF:      (2000, 2000, 4) uint8
  H5 s2:        (100, 100, 12, 12) float32

  │  Level 1: ref_h = 2000//20 = 100, crop (4,8,14,16) in ref space
  │  HR coord = crop * 20 → (80, 160, 280, 320)
  ▼

Level 1 裁剪后
  HR data:      (4, 280, 320) float32 [0,1]
  H5 s2:        (14, 16, 12, 12) float32

  │  Level 2: augmentation 在 H5 ref (14,16) 空间采样
  │  crop_params = (2, 4, 8, 10) → HR = (40, 80, 160, 200)
  ▼

Level 2 裁剪后
  HR global:    (4, 224, 224) float32 normalized  ← resized_crop + augment
  H5 s2:        (8, 10, 12, 12) float32           ← 同分数裁剪

  │  统一 resize: target = aligned(224/20) = 10
  ▼

Resize 后
  H5 s2:        (10, 10, 12, 12) float32

  │  _process_olmoearth_modalities: pad + fill + normalize + to_tensor
  ▼

单 sample 输出
  global_crops:  [(4, 224, 224), (4, 224, 224)]
  local_crops:   [(4, 96, 96) × 8]
  h5 modalities: [(10, 10, 12, 12), (10, 10, 12, 2), (10, 10, 12, 6)]

  │  collate_h5_olmoearth_and_cast (B=2)
  ▼

Batch 输出
  collated_global_crops:  (4, 4, 224, 224)
  collated_local_crops:   (16, 4, 96, 96)
  collated_masks:         (4, 196)
  olmoearth_modalities:   [{s2: (2,10,10,12,12), ...}, ...]
  olmoearth_metadata:     [{timestamps: (2,12,2), ...}, ...]

  │  DINOv3WithOLMoEarth.forward_backward
  ▼

训练
  Student/Teacher ViT: (4, 4, 224, 224) → patchify → (4, 196, dim)
  OLMoEarth encoder:   (2, 10, 10, 12, 12) → patchify(2) → (2, 5, 5, 12, ..., D)
```

# Commit 899c37f 修改报告

**提交**: `899c37f2fbb3` — support olmodataloader and olmoearth inference
**作者**: guantongpeng
**日期**: 2026-05-13 16:06:07 +0800
**变更统计**: 18 个文件，+2403 行 / -18 行

---

## 一、修改背景与目标

本次提交为 DINOv3 框架新增了 OLMoEarth 多模态遥感数据的完整支持，使 DINOv3 能够在 stage 2 训练中同时处理：

1. **HR_4V 高分辨率 TIFF 图像** — 作为 DINO 自监督学习的输入（支持 RGB 或 RGBNIR 4 通道）
2. **H5 格式多模态遥感数据** — 输入到冻结的 OLMoEarth 模型中提取多模态嵌入，为后续融合 loss 预留接口

整体架构为 `DINOv3WithOLMoEarth`：在原有 SSLMetaArch 基础上，每个训练 step 先推理 OLMoEarth 获取多模态嵌入，再执行 DINO forward-backward。

---

## 二、新增文件详解

### 2.1 `dinov3/data/datasets/h5_olmoearth.py`（835 行）

**新增 `H5OlmoEarthDataset` 类**，这是整个 OLMoEarth 数据管道的核心数据集类。

**原因**: OLMoEarth 的数据结构与 DINOv3 原有的 ImageNet 等数据集完全不同——它需要同时读取 H5 文件（低分辨率多模态数据）和 TIFF 文件（高分辨率图像），并保证两者在空间上对齐。

#### 关键设计

**（a）双数据源读取**

- H5 文件：包含 `sentinel2_l2a`、`sentinel1`、`landsat` 等低分辨率多模态数据（时间序列，HWC 格式）
- TIFF 文件：HR_4V 高分辨率图像（3 或 4 通道），存储在独立目录中，元信息（`start_time`）在 CSV 文件中

```python
class H5OlmoEarthDataset(ExtendedVisionDataset):
    def __init__(
        self,
        root: str,
        transform,                           # DataAugmentationDINOMultiChannel
        olmoearth_modalities: list[str],     # 要加载的模态列表
        max_sequence_length: int = 12,       # 时间序列最大长度
        hr_data_dir: str | None = None,      # TIFF 根目录
        hr_meta_dir: str | None = None,      # HR 元数据 CSV 目录
        hr_crop_scale: tuple = (0.5, 1.0),   # 随机裁剪比例范围
        n_channels: int = 3,                 # 3=RGB, 4=RGBNIR
    ):
```

**（b）三级数据发现模式**

`_discover_samples()` 支持三种目录结构，以适应不同规模的数据组织方式：

| 模式 | 触发条件 | 目录结构 |
|------|----------|----------|
| JSON 索引 | `root` 为 `.json` 文件 | 预构建的 JSON 索引文件（由 `build_dataset_index.py` 生成） |
| 年份组织 | `root` 下有 `YYYY/` 子目录 | `<root>/2024/H5/`, `<root>/2024/HR/` |
| 平铺目录 | 其他目录 | `<root>/*.h5`, `<hr_data_dir>/sample_x/T0.tif` |

**（c）`__getitem__` 核心流程**

```python
def __getitem__(self, index):
    # 1. Level 1 随机空间裁剪（HR 像素坐标）
    top, left, crop_h, crop_w = _random_crop_params(hr_h, hr_w, self.hr_crop_scale)

    # 2. 随机选择一张 TIFF 作为 data，其余作为 target
    data_idx = random.randint(0, len(tif_info) - 1)

    # 3. 读取 H5 并按 Level 1 裁剪比例做空间裁剪
    sample_dict = _read_h5_file(h5_path, all_modalities)
    sample_dict = _crop_h5_sample_dict(sample_dict, top, left, crop_h, crop_w, hr_h, hr_w)

    # 4. HR data 转 tensor → DINO 多通道增强
    aug_output = self.transform(hr_tensor)
    # aug_output 包含 global_crop_coords 和 global_crop_flips

    # 5. 根据 DINO 增强的 crop 坐标，对 H5 模态做比例裁剪（Level 2）
    for (tf, lf, hf, wf), do_flip in zip(
        aug_output["global_crop_coords"], aug_output["global_crop_flips"]
    ):
        cropped_h5 = _crop_h5_sample_dict_fractional(copy.deepcopy(sample_dict), tf, lf, hf, wf)
        if do_flip:
            cropped_h5 = _flip_h5_sample_dict(cropped_h5)
        modalities, metadata = self._process_olmoearth_modalities(cropped_h5, ...)

    return {
        "global_crops": aug_output["global_crops"],
        "local_crops": aug_output["local_crops"],
        "h5_olmoearth_crops": h5_olmoearth_crops,  # 每个 global crop 对应一份 H5 模态
        "hr_data_start_time": data_start_time,
        "hr_target_images": target_arrays,
        "hr_target_start_times": target_start_times,
    }
```

**（d）辅助函数**

| 函数 | 作用 |
|------|------|
| `_read_h5_file` | 读取 H5 文件，返回样本字典和缺失时间步掩码 |
| `_pad_timestamps` | 将时间序列 pad 到 `max_sequence_length` |
| `_fill_missing_timesteps` | 用 MISSING_VALUE 填充缺失的时间步 |
| `_fill_sample_with_missing_values` | 为缺失的模态生成全 MISSING_VALUE 的占位数据 |
| `_normalize_sample` | 调用 OLMoEarth 的 Normalizer（COMPUTED → PREDEFINED 回退）归一化各模态 |
| `_crop_h5_sample_dict` | 按 HR 像素坐标比例裁剪 H5 空间模态，尺寸对齐到 4 的倍数 |
| `_crop_h5_sample_dict_fractional` | 同上，但输入为分数坐标（来自 DINO 增强的 crop coords） |
| `_flip_h5_sample_dict` | 水平翻转 H5 空间模态 |
| `_random_crop_params` | 在 HR 坐标系下生成随机裁剪参数 |
| `_read_tif_crop` | 使用 rasterio Window 读取 TIFF 的指定裁剪区域 |
| `_hr_to_tensor` | 将 `(H, W, C)` uint8 数组转为 `(C, H, W)` float [0,1] tensor |

---

### 2.2 `dinov3/data/augmentations.py` — 新增 `DataAugmentationDINOMultiChannel`（190 行）

**原因**: 原有的 `DataAugmentationDINO` 假设输入为 PIL Image（3 通道 RGB），不支持 4 通道 RGBNIR，且不返回裁剪坐标信息（H5 模态需要据此做比例裁剪）。

**与 `DataAugmentationDINO` 的关键区别**:

| 特性 | DataAugmentationDINO | DataAugmentationDINOMultiChannel |
|------|---------------------|----------------------------------|
| 输入格式 | PIL Image | `(C, H, W)` float tensor |
| 通道数 | 固定 3 通道 | 支持 3（RGB）或 4（RGBNIR） |
| 颜色增强 | 作用于全部通道 | 仅作用于前 3 通道（RGB），NIR 通道不变 |
| Solarize | 作用于全部通道 | 仅作用于 RGB 通道 |
| 归一化 | ImageNet 3 通道 | RGB 用 ImageNet 均值/方差，NIR 单独配置 |
| 输出内容 | crops + offsets | crops + **global_crop_coords** + **global_crop_flips** |

```python
class DataAugmentationDINOMultiChannel:
    def __init__(self, ..., n_channels=3, nir_mean=0.5, nir_std=0.25, ...):
        # 构建 4 通道均值/方差：[R, G, B, NIR]
        mean_list = list(rgb_mean)
        std_list = list(rgb_std)
        if n_channels == 4:
            mean_list.append(nir_mean)
            std_list.append(nir_std)

    def _apply_geometric_crop(self, tensor, crop_size, scale):
        """返回 (cropped_tensor, top_frac, left_frac, h_frac, w_frac)"""
        # 分数坐标用于 H5 模态的比例裁剪

    def _apply_augmentation(self, crop, is_global, aug_idx):
        """颜色抖动 → 高斯模糊 → 归一化"""
        # 4 通道时：RGB 做 color jitter，NIR 不做
        if self.n_channels == 4:
            rgb = crop[:3]
            nir = crop[3:4]
            rgb = self.color_jittering(rgb)
            crop = torch.cat([rgb, nir], dim=0)

    def __call__(self, image_tensor):
        # 返回 global_crop_coords 和 global_crop_flips
        return {
            "global_crops": [...],
            "local_crops": [...],
            "global_crop_coords": [(tf, lf, hf, wf), ...],  # 新增
            "global_crop_flips": [bool, bool],                # 新增
        }
```

---

### 2.3 `dinov3/data/collate.py` — 新增 `collate_h5_olmoearth_and_cast`（约 200 行）

**原因**: OLMoEarth 数据的 batch 合并比原有 `collate_data_and_cast` 复杂得多——需要同时处理 DINO crops、多模态 H5 数据（尺寸可能不同）、HR target 图像（时序长度可能不同）。

**四个处理步骤**:

```python
def collate_h5_olmoearth_and_cast(samples_list, ...):
    # ---- 1. 合并 DINO crops (global + local) ----
    collated_global_crops = torch.stack(...)  # [n_global_crops * B, C, gH, gW]
    collated_local_crops = torch.stack(...)   # [n_local_crops * B, C, lH, lW]

    # ---- 2. 生成 iBOT 掩码 ----
    # 与 collate_data_and_cast 逻辑相同

    # ---- 3. 合并 OLMoEarth 模态（按 global crop 分组）----
    # 关键：不同样本的同一模态可能尺寸不同，需要插值对齐
    for crop_idx in range(n_h5_crops):
        for key in modality_keys:
            shapes = [t.shape for t in tensors]
            if len(set(shapes)) == 1:
                crop_modalities[key] = torch.stack(tensors)     # 尺寸一致直接 stack
            else:
                # 双线性插值对齐到最大尺寸，4 倍 patch 对齐
                max_h = (max(t.shape[0] for t in tensors) // 4) * 4
                max_w = (max(t.shape[1] for t in tensors) // 4) * 4
                # ... resize 后 stack

    # ---- 4. 合并 HR target 图像和 start_times ----
    # 不同样本可能有不同数量的 target（不同时序长度），用零填充 + mask 处理
    if has_targets:
        for tensors in all_target_tensors:
            n = len(tensors)
            if n < max_targets:
                pad = torch.zeros(n_channels, target_size, target_size)
                tensors = tensors + [pad] * (max_targets - n)
            mask[:n] = True  # mask 标记哪些是有效 target

    return {
        "collated_global_crops": ...,
        "collated_local_crops": ...,
        "collated_masks": ...,
        "olmoearth_modalities": olmoearth_modalities_by_crop,  # 新增：list of dicts
        "olmoearth_metadata": olmoearth_metadata_by_crop,      # 新增：list of dicts
        "hr_target_images": hr_target_images_batch,             # 新增
        "hr_target_masks": hr_target_masks_batch,               # 新增
        "hr_target_start_times": ...,                           # 新增
        "hr_data_start_time": ...,                              # 新增
    }
```

**变尺寸对齐策略**:

- 4D 张量 `(H, W, T, C)`：先 reshape 为 `(T*C, H, W)` 做 bilinear 插值，再 reshape 回原格式
- 3D 张量 `(H, W, C)`：先 permute 为 `(C, H, W)` 做插值，再 permute 回
- 空间尺寸向下取整到 4 的倍数（`_H5_PATCH_ALIGN = 4`），确保与 OLMoEarth patchify 兼容

---

### 2.4 `dinov3/train/dino_with_olmoearth.py`（165 行）

**新增 `DINOv3WithOLMoEarth` 模型类**，封装 DINO + OLMoEarth 联合训练架构。

**原因**: 需要在 DINO 自监督训练循环中同时运行冻结的 OLMoEarth 模型推理，但不影响原有 DINO 的训练逻辑和 FSDP 分布式策略。

```python
class DINOv3WithOLMoEarth(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dino_model = SSLMetaArch(cfg)      # 原有 DINO 训练模块
        self.olmoearth_model = None              # 在 init_weights 后加载（FSDP 初始化后）

    def init_weights(self):
        self.dino_model.init_weights()
        # FSDP 将 DINO 模型移至 CUDA 后，再加载 OLMoEarth 模型
        self.olmoearth_model = load_olmoearth_model(
            self.olmoearth_model_id_or_path, device=torch.device("cuda")
        )

    def prepare_for_distributed_training(self):
        # 只有 DINO 模型进入 FSDP，OLMoEarth 不参与
        self.dino_model.prepare_for_distributed_training()

    def forward_backward(self, data, *, teacher_temp, iteration=0, **kwargs):
        # 1. 提取 OLMoEarth 数据（从 batch 中 pop 出）
        olmoearth_modalities_list = data.pop("olmoearth_modalities", None)
        olmoearth_metadata_list = data.pop("olmoearth_metadata", None)
        data.pop("hr_target_images", None)    # SSLMetaArch 不需要这些字段
        data.pop("hr_target_masks", None)
        data.pop("hr_target_start_times", None)
        data.pop("hr_data_start_time", None)

        # 2. 标准 DINO V3 forward-backward
        total_loss, dino_metrics = self.dino_model.forward_backward(
            data, teacher_temp=teacher_temp, iteration=iteration
        )

        # 3. OLMoEarth 推理（冻结，无梯度），per global crop
        if olmoearth_modalities_list is not None and self.olmoearth_model is not None:
            for crop_idx, (modalities, metadata) in enumerate(
                zip(olmoearth_modalities_list, olmoearth_metadata_list)
            ):
                olmoearth_embeddings = run_olmoearth_inference(
                    self.olmoearth_model, modalities, metadata,
                    patch_size=self.olmoearth_patch_size,
                )
                # 当前仅记录 shape，后续会接入融合 loss
                for mod_name, emb in olmoearth_embeddings.items():
                    metrics_dict[f"olmoearth/crop{crop_idx}/{mod_name}_shape"] = str(list(emb.shape))

        return total_loss, metrics_dict
```

**委托方法**: `train()`、`update_ema()`、`get_params_groups()`、`build_data_augmentation_dino()` 等均委托给 `self.dino_model`，确保与原有训练流程兼容。

**`build_data_augmentation_dino_h5()`**: 构建 `DataAugmentationDINOMultiChannel`，从配置中读取 `n_channels`、`nir_mean`、`nir_std` 等参数。

---

### 2.5 `dinov3/train/olmoearth_inference.py`（130 行）

**OLMoEarth 推理工具函数**。

**`load_olmoearth_model()`**:

```python
def load_olmoearth_model(model_id_or_path, device):
    # 优先尝试作为 ModelID 从 HuggingFace Hub 加载
    try:
        model_id = ModelID(model_id_or_path)   # e.g. "OlmoEarth-v1-Nano"
        model = load_model_from_id(model_id)
    except ValueError:
        # 回退到本地路径加载
        model = load_model_from_path(model_id_or_path)

    model.eval()
    model.requires_grad_(False)    # 冻结参数
    model.to(device)
    return model
```

**`run_olmoearth_inference()`**:

```python
def run_olmoearth_inference(model, olmoearth_modalities, olmoearth_metadata, patch_size=4, device="cuda"):
    # 1. 构建 batch_dict 和 mask_dict
    #    - timestamps → long tensor
    #    - latlon → float32 tensor + mask
    #    - 各模态数据 → float32 tensor + mask（全 MISSING_VALUE 的样本标记为 MISSING）

    # 2. 构建 MaskedOlmoEarthSample
    masked_sample = MaskedOlmoEarthSample(**batch_dict, **mask_dict)

    # 3. 推理
    with torch.no_grad():
        output = model.encoder(masked_sample, fast_pass=False, patch_size=patch_size)

    # 4. 提取各模态嵌入
    return {"sentinel2_l2a": features, "sentinel1": features, ...}
```

---

### 2.6 `dinov3/data/build_dataset_index.py`（188 行）

**数据集索引构建脚本**，用于预构建 JSON 索引文件，加速数据集初始化。

**原因**: 当数据量很大时，每次训练启动时扫描目录会很慢。预构建索引后，数据集只需读取 JSON 即可。

```bash
python build_dataset_index.py <dataset_root> -o dataset_index.json
```

期望的目录结构：
```
<dataset_root>/
  2024/
    H5/sample_0.h5, sample_1.h5, ...
    HR/sample_0/T0.tif, T1.tif, ...
    HR/meta/sample_0.csv, ...
  2025/
    H5/...
    HR/...
```

输出 JSON 格式：
```json
{
  "dataset_root": "/path/to/root",
  "samples": [
    {
      "sample_id": "sample_0",
      "years": {
        "2024": {"h5_path": "...", "hr_dir": "...", "tif_info": [...], "hr_dims": [H, W, C]},
        "2025": {...}
      }
    }
  ]
}
```

---

### 2.7 `dinov3/configs/train/dinov3_olmoearth.yaml`（213 行）

**OLMoEarth 训练配置文件**，关键配置项：

```yaml
MODEL:
  META_ARCHITECTURE: DINOv3WithOLMoEarth    # 使用新的联合架构
student:
  in_chans: 4                                # 学生网络输入 4 通道
teacher:
  in_chans: 4                                # 教师网络输入 4 通道
crops:
  n_channels: 4                              # RGBNIR 4 通道
  nir_mean: 0.5                              # NIR 归一化均值
  nir_std: 0.25                              # NIR 归一化方差
train:
  dataset_path: H5OlmoEarth:root=<path>:n_channels=4
olmoearth:
  enabled: true
  model_id_or_path: "OlmoEarth-v1-Nano"      # OLMoEarth 模型
  patch_size: 4                               # OLMoEarth patch 大小
```

---

### 2.8 测试文件

| 文件 | 说明 |
|------|------|
| `tests/h5_data_maker.py` | 生成合成 H5 测试数据（3 个 H5 文件：正常、单时间步、无 HR_4V） |
| `tests/test_dinov3witholmoearth.py` | 端到端管道测试：数据集 → collate → OLMoEarth 推理 |
| `tests/test_h5_olmoearth_dataloader.py` | 数据集单元测试：基础加载、DINO 增强、collation、DataLoader |

---

## 三、修改文件详解

### 3.1 `dinov3/train/train.py`

**变更 1：新增 `build_h5_olmoearth_data_loader_from_cfg` 函数**（约 80 行）

**原因**: OLMoEarth 数据加载需要使用 `collate_h5_olmoearth_and_cast` 和 `DataAugmentationDINOMultiChannel`，与原有 `build_data_loader_from_cfg` 不兼容。

```python
def build_h5_olmoearth_data_loader_from_cfg(cfg, model, start_iter):
    # 构建 collate 函数（使用 collate_h5_olmoearth_and_cast）
    # 构建数据集（使用 model.build_data_augmentation_dino_h5(cfg)）
    # 构建 DataLoader
```

**变更 2：`do_train()` 中根据配置选择数据加载器**

```python
# 修改前：固定使用多分辨率数据加载器
data_loader = build_multi_resolution_data_loader_from_cfg(cfg=cfg, model=model, start_iter=start_iter)

# 修改后：根据 olmoearth.enabled 选择
olmoearth_enabled = getattr(cfg, "olmoearth", None) is not None and cfg.olmoearth.get("enabled", False)
if olmoearth_enabled:
    data_loader = build_h5_olmoearth_data_loader_from_cfg(cfg=cfg, model=model, start_iter=start_iter)
else:
    data_loader = build_multi_resolution_data_loader_from_cfg(cfg=cfg, model=model, start_iter=start_iter)
```

**变更 3：注册 `DINOv3WithOLMoEarth` 架构**

```python
meta_arch = {
    "SSLMetaArch": SSLMetaArch,
    "MultiDistillationMetaArch": MultiDistillationMetaArch,
    "DINOv3WithOLMoEarth": DINOv3WithOLMoEarth,   # 新增
}.get(cfg.MODEL.META_ARCHITECTURE, None)
```

**变更 4：非数值 metrics 过滤**

**原因**: `DINOv3WithOLMoEarth.forward_backward` 在 metrics_dict 中添加了字符串类型的 metric（如 `olmoearth/crop0/sentinel2_l2a_shape: "[2, 56, 56, 12, 4, 384]"`），而后续 `torch.stack([torch.as_tensor(v, ...) for v in metrics_dict.values()])` 无法处理非数值类型，会导致训练崩溃。

```python
# debug 代码：检测非数值 metric
for k, v in metrics_dict.items():
    if not isinstance(v, (int, float, torch.Tensor)):
        logger.info(f"Non-numeric metric found: Key='{k}', Value='{v}', Type={type(v)}")

# 过滤非数值 metrics
filtered_metrics_dict = {
    k: v for k, v in metrics_dict.items()
    if isinstance(v, (int, float, torch.Tensor))
}
metrics_values = torch.stack(
    [torch.as_tensor(v, ...) for v in filtered_metrics_dict.values()]
)
metrics_dict = dict(zip(filtered_metrics_dict.keys(), metrics_values))
```

**变更 5：`logger.info(cfg)` → `logger.info(str(cfg))`**

**原因**: 某些配置对象（如 OmegaConf DictConfig）不支持直接传给 logger，需要先转为字符串。

**变更 6：`build_multi_resolution_data_loader_from_cfg` 内部调用替换**

```python
# 修改前
loaders.append(build_data_loader_from_cfg(cfg=cfg_i, model=model, start_iter=start_iter))
# 修改后
loaders.append(build_h5_olmoearth_data_loader_from_cfg(cfg=cfg_i, model=model, start_iter=start_iter))
```

> **注意**: 此处将多分辨率数据加载器内部也改为使用 `build_h5_olmoearth_data_loader_from_cfg`，这可能影响非 OLMoEarth 场景下的多分辨率训练。当 `olmoearth.enabled=false` 且使用多分辨率时，会走到 `build_multi_resolution_data_loader_from_cfg`，其内部现在会调用 H5 版本的 builder，可能导致兼容性问题。

---

### 3.2 `dinov3/data/loaders.py`

**变更：注册 `H5OlmoEarthDataset` + 扩展数据集字符串解析**

**原因**: 需要通过配置字符串（如 `H5OlmoEarth:root=/path:hr_data_dir=/path:n_channels=4`）创建 H5OlmoEarthDataset 实例。

```python
# 修改前：只支持 root, extra, split 三个参数
assert key in ("root", "extra", "split")

# 修改后：新增 hr_data_dir, hr_meta_dir, hr_crop_scale, n_channels
assert key in ("root", "extra", "split", "hr_data_dir", "hr_meta_dir", "hr_crop_scale", "n_channels")

# hr_crop_scale 特殊处理：解析为 (float, float) 元组
if key == "hr_crop_scale":
    parts = value.split(",")
    kwargs[key] = (float(parts[0]), float(parts[1])) if len(parts) == 2 else None
elif key == "n_channels":
    kwargs[key] = int(value)

# 新增数据集类型
elif name == "H5OlmoEarth":
    class_ = H5OlmoEarthDataset
```

---

### 3.3 `dinov3/models/__init__.py`

**变更：ViT 构建时新增 `in_chans` 参数**

**原因**: OLMoEarth 使用 4 通道（RGBNIR）输入，原有 ViT 硬编码 3 通道。

```python
# 修改前
vit_kwargs = dict(
    img_size=img_size,
    patch_size=args.patch_size,
    ...
)

# 修改后
vit_kwargs = dict(
    img_size=img_size,
    in_chans=getattr(args, "in_chans", 3),   # 新增：从配置读取，默认 3
    patch_size=args.patch_size,
    ...
)
```

---

### 3.4 `dinov3/data/__init__.py` / `dinov3/data/datasets/__init__.py`

**变更：导出新模块**

```python
# dinov3/data/__init__.py
from .collate import collate_data_and_cast, collate_h5_olmoearth_and_cast  # 新增导出

# dinov3/data/datasets/__init__.py
from .h5_olmoearth import H5OlmoEarthDataset  # 新增导出
```

---

### 3.5 `dinov3/data/collate.py` — 原有 `collate_data_and_cast` 微调

- 删除两行注释（无实质改动）
- `if random_circular_shift:` 后删除注释 `# apply le random circular shift to`（无实质改动）

---

### 3.6 `.gitignore` / `dinov3/data/.gitignore`

```gitignore
# 根目录 .gitignore
+local_dino

# dinov3/data/.gitignore（新增文件）
+demo_dataset
```

---

## 四、数据流全景图

```
┌─────────────────────────────────────────────────────────────────┐
│                      H5OlmoEarthDataset.__getitem__             │
│                                                                 │
│  TIFF 文件 ──→ _read_tif_crop() ──→ (H,W,C) uint8             │
│       │                              │                          │
│       │                              ↓ _hr_to_tensor()         │
│       │                         (C,H,W) float [0,1]            │
│       │                              │                          │
│       │                              ↓ DataAugmentationDINO     │
│       │                              MultiChannel               │
│       │                              │                          │
│       │                    ┌─────────┴──────────┐              │
│       │                    │                    │               │
│       │             global_crops          local_crops           │
│       │             (2 × (C,224,224))   (8 × (C,96,96))       │
│       │             + crop_coords       (H5 不需要 local crops)│
│       │             + crop_flips                                  │
│       │                                                         │
│  H5 文件 ──→ _read_h5_file() ──→ sample_dict                   │
│       │                              │                          │
│       │                     Level 1 crop (与 TIFF 同区域)       │
│       │                              │                          │
│       │                     Level 2 crop (按 DINO crop_coords)  │
│       │                     + flip (按 DINO crop_flips)         │
│       │                              │                          │
│       │                     _normalize_sample()                 │
│       │                              │                          │
│       │                    modalities dict + metadata dict      │
│       │                    (每个 global crop 一份)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              collate_h5_olmoearth_and_cast()                    │
│                                                                 │
│  DINO crops:     torch.stack → [B*2, C, 224, 224] (global)    │
│                              [B*8, C, 96, 96]  (local)         │
│  iBOT masks:     生成 + 合并                                    │
│  OLMoEarth:      变尺寸插值对齐 → stack (per global crop)       │
│  HR targets:     零填充 + mask → stack                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DINOv3WithOLMoEarth.forward_backward()             │
│                                                                 │
│  1. pop olmoearth_modalities / metadata / hr_target_*          │
│  2. self.dino_model.forward_backward(data) → loss, metrics     │
│  3. run_olmoearth_inference(per global crop) → embeddings       │
│     (当前仅记录 shape，后续接入融合 loss)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、遗留问题与注意事项

1. ~~**`build_multi_resolution_data_loader_from_cfg` 内部调用被替换**~~ — **已修复**: 已恢复为 `build_data_loader_from_cfg`，非 OLMoEarth 场景不受影响

2. ~~**debug 代码未清理**~~ — **已修复**: 随问题 3 一起清理，非数值 metric 的 debug 检测代码和过滤逻辑已移除

3. ~~**metrics_dict 中的字符串 metric**~~ — **已修复**: `DINOv3WithOLMoEarth.forward_backward` 中 OLMoEarth embedding shape 改为 `logger.debug()` 记录，不再混入 metrics_dict

4. ~~**测试文件硬编码路径**~~ — **已修复**: `test_dinov3witholmoearth.py`、`test_h5_olmoearth_dataloader.py`、`h5_data_maker.py` 中的硬编码路径改为基于 `os.path.dirname(__file__)` 动态计算

5. **OLMoEarth 推理结果未接入 loss**：当前 `forward_backward` 中 OLMoEarth embedding 仅记录 shape 后 `del`，融合 loss 尚未实现

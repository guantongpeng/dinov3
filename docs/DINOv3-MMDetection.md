# DINOv3 + MMDetection Integration Guide

## 一、概览

本指南介绍如何使用 DINOv3 ViT 作为 backbone，结合 MMDetection 框架进行目标检测下游任务微调。

### 已支持的检测器

| 检测器 | 配置 | 说明 |
|--------|------|------|
| Faster R-CNN | `configs/faster_rcnn/` | 两阶段检测器，含 RPN + FPN + RoI Head |
| Mask R-CNN | `configs/mask_rcnn/` | 在 Faster R-CNN 基础上增加 Mask Head |

### 已支持的 Backbone

| Model | embed_dim | depth | patch_size | 参数量（约） |
|-------|-----------|-------|------------|-------------|
| `dinov3_vitb16` | 768 | 12 | 16 | 86M |
| `dinov3_vitl16` | 1024 | 24 | 16 | 304M |
| `dinov3_vitl16plus` | 1024 | 24 | 16 | 304M (SwiGLU) |
| `dinov3_vits16` | 384 | 12 | 16 | 22M |
| `dinov3_vith16plus` | 1280 | 32 | 16 | 632M |
| `dinov3_vit7b16` | 4096 | 40 | 16 | 7B |

---

## 二、安装

### 2.1 环境要求

```bash
# 激活项目环境
source /home/guantp/pro/olmoearth_pretrain/.venv/bin/activate

# 安装 MMDetection 生态
pip install openmim
mim install mmdet
```

或者手动指定版本：

```bash
pip install mmengine mmcv mmdet
```

### 2.2 验证安装

```python
import mmdet
import mmengine
import mmcv
print(f"MMDetection: {mmdet.__version__}")
print(f"MMEngine: {mmengine.__version__}")
```

---

## 三、快速开始

### 3.1 数据准备

数据集需要 COCO 格式的标准目录结构：

```
data/coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
    *.jpg
  val2017/
    *.jpg
```

### 3.2 训练 Faster R-CNN (单 GPU)

```bash
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16
```

### 3.3 多 GPU 训练 (torchrun)

```bash
torchrun --nproc_per_node=4 -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitl16_1x_coco.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitl16 \
    --batch-size 8 --amp
```

### 3.4 训练 Mask R-CNN

```bash
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/mask_rcnn/mask_rcnn_dinov3_vitb16_1x_coco.py \
    --work-dir ./work_dirs/mask_rcnn_dinov3_vitb16
```

### 3.5 恢复训练

```bash
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16 \
    --resume ./work_dirs/faster_rcnn_dinov3_vitb16/epoch_8.pth
```

---

## 四、架构设计

### 4.1 特征提取流程

```
Input Image (B, 3, H, W)
    │
    ▼
DinoVisionTransformerBackbone
    │
    ├─ patch embedding (patch_size=16)
    ├─ transformer block 0..n_blocks-1
    │
    ├─ layers_to_use[0] → (B, embed_dim, H/16, W/16)  ← 第1个特征
    ├─ layers_to_use[1] → (B, embed_dim, H/16, W/16)  ← 第2个特征
    ├─ layers_to_use[2] → (B, embed_dim, H/16, W/16)  ← 第3个特征
    └─ layers_to_use[3] → (B, embed_dim, H/16, W/16)  ← 第4个特征
    │
    ▼
MMDetection FPN Neck
    │
    ├─ lateral_conv (1×1): embed_dim → 256
    ├─ top-down pathway + strided convs
    │
    ├─ P2 (stride=16):  (B, 256, H/16,  W/16)
    ├─ P3 (stride=32):  (B, 256, H/32,  W/32)
    ├─ P4 (stride=64):  (B, 256, H/64,  W/64)
    ├─ P5 (stride=128): (B, 256, H/128, W/128)
    └─ P6 (stride=256): (B, 256, H/256, W/256)  (for RPN)
    │
    ▼
RPN → RoI Align → Detection / Mask Head
```

### 4.2 关键设计决策

**为什么 Backbone 输出单分辨率特征？**

DINOv3 是 ViT 架构，所有 transformer block 输出的 patch tokens 都在相同空间分辨率（`H//patch_size × W//patch_size`）。Backbone 从不同深度的 block 提取特征（不同语义层次），FPN 通过 top-down pathway 和 strided convs 构建空间多尺度。

这与 ResNet 等 CNN backbone 不同（CNN 天然产生 stride=4, 8, 16, 32 的多分辨率特征），但对 ViT 是合理的，与 Swin Transformer 的做法类似。

---

## 五、配置说明

### 5.1 Model 配置结构

```python
model = dict(
    type="FasterRCNN",                    # 检测器类型
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],   # ImageNet BGR mean
        std=[58.395, 57.12, 57.375],      # ImageNet BGR std
        bgr_to_rgb=True,
        pad_size_divisor=16,              # 关键：必须能被 patch_size 整除
    ),
    backbone=dict(
        type="DinoVisionTransformerBackbone",  # 自定义 backbone
        model_name="dinov3_vitb16",
        pretrained=True,
        layers_to_use=[2, 5, 8, 11],      # 提取哪几层 ViT block
        out_indices=(0, 1, 2, 3),         # 输出哪些特征图给 FPN
        use_layernorm=True,               # 每层输出后是否加 LayerNorm2D
        frozen_stages=-1,                 # -1 冻结全部 backbone
    ),
    neck=dict(
        type="FPN",
        in_channels=[768, 768, 768, 768],  # embed_dim × num_layers
        out_channels=256,
        num_outs=5,                       # P2-P6 (5 个输出层)
    ),
    # ... rpn_head, roi_head, train_cfg, test_cfg
)
```

### 5.2 `layers_to_use` 选择指南

默认选择每个 quarter 的最后一层，确保覆盖不同语义层次：

| Model | n_blocks | 默认 layers_to_use |
|-------|----------|-------------------|
| ViT-S/B | 12 | `[2, 5, 8, 11]` |
| ViT-L | 24 | `[5, 11, 17, 23]` |
| ViT-H+ | 32 | `[7, 15, 23, 31]` |
| ViT-7B | 40 | `[9, 19, 29, 39]` |

可自定义为更密集的特征提取，例如 6 层：`[0, 3, 6, 9, 12, 15]`（但会增加显存开销）。

### 5.3 FPN 输出 stride

FPN 输出 stride = `patch_size × 2^i`：

| FPN Level | stride (patch_size=16) | 典型用途 |
|-----------|----------------------|----------|
| P2 | 16 | RoI Align（小物体） |
| P3 | 32 | RoI Align（中物体） |
| P4 | 64 | RoI Align（大物体） |
| P5 | 128 | RoI Align（极大物体） |
| P6 | 256 | RPN（仅 anchor 生成） |

### 5.4 Learning Rate 策略

默认使用分层学习率（paramwise_cfg）：

```
backbone:  lr * 0.1   (低 10 倍)
neck:      lr * 1.0
rpn_head:  lr * 1.0
roi_head:  lr * 1.0
```

冻结 backbone（`frozen_stages=-1`）时 backbone lr 不生效。

端到端微调（`--train-backbone`）时，建议使用更低的初始 lr（如 `--lr 5e-5`）。

---

## 六、训练策略建议

### 6.1 两阶段微调（推荐）

**阶段 1：冻结 backbone，训练检测头**

```bash
python -m dinov3.eval.detection.mm_train \
    configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \
    --work-dir ./work_dirs/stage1_frozen
```

默认配置即冻结 backbone（`frozen_stages=-1`），只训练 FPN + RPN + RoI Head。这适应了 DINOv3 预训练特征到检测任务。Backbone 参数不变，显存开销小。

**阶段 2：解冻 backbone，端到端微调**

```bash
python -m dinov3.eval.detection.mm_train \
    configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \
    --work-dir ./work_dirs/stage2_unfrozen \
    --train-backbone \
    --lr 5e-5 \
    --resume ./work_dirs/stage1_frozen/epoch_12.pth
```

继承阶段 1 的检测头权重，以更低学习率微调整个网络。

### 6.2 显存优化

| 策略 | 说明 |
|------|------|
| 冻结 backbone | `frozen_stages=-1`（默认），不缓存 ViT 梯度 |
| 减少 batch size | `--batch-size 1` |
| 减少 `layers_to_use` | 提取 2 层而非 4 层，减少 FPN 输入通道维数 |
| AMP 混合精度 | `--amp`，使用 float16 减少一半显存 |
| Gradient Checkpointing | 在 mmengine 配置中设置 `gradient_checkpointing=True` |

### 6.3 多尺度训练

`coco_detection.py` 默认使用多尺度 resize（480-800 短边），增强尺度鲁棒性：

```python
dict(
    type="RandomResize",
    scale=[(480, 1333), (512, 1333), ..., (800, 1333)],
    keep_ratio=True,
)
```

---

## 七、自定义数据集

### 7.1 COCO 格式

MMDetection 默认支持 COCO 格式。自定义数据集只需提供与 COCO 格式相同的 JSON annotation 文件：

```json
{
  "images": [{"id": 1, "file_name": "img.jpg", "height": 480, "width": 640}],
  "annotations": [{"image_id": 1, "bbox": [x, y, w, h], "category_id": 0}],
  "categories": [{"id": 0, "name": "cat"}]
}
```

### 7.2 修改类别数

在 config 中修改：

```python
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=YOUR_NUM_CLASSES),
    ),
)
# 以及 mask_head（Mask R-CNN 时）
```
### 7.3 使用项目内置 CustomDetectionDataset

项目自带 `CustomDetectionDataset`（`dinov3/eval/detection/dataset.py`）也支持自定义 JSON 格式。但 MMDetection 路径使用 `CocoDataset`，如果你的数据已符合 COCO 格式建议直接用 COCO 路径。

---

## 八、命令行参数参考

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `config` | MMDetection 配置 .py 文件（必须） | — |
| `--work-dir` | 输出目录（必须） | — |
| `--batch-size` | 覆盖 batch size | config 中的值 |
| `--lr` | 覆盖学习率 | config 中的值 |
| `--amp` | 启用 AMP 混合精度 | False |
| `--train-backbone` | 解冻 backbone 端到端训练 | False |
| `--backbone-weights` | 自定义 backbone 权重路径 | None (使用 hub) |
| `--layers-to-use` | 覆盖 layers_to_use | config 中的值 |
| `--resume` | 从 checkpoint 恢复 | None |
| `--seed` | 随机种子 | 42 |

---

## 九、文件清单

| 文件 | 作用 |
|------|------|
| `dinov3/eval/detection/mm_backbone.py` | MMDetection 兼容的 DINOv3 backbone wrapper |
| `dinov3/eval/detection/mm_train.py` | MMDetection 训练入口脚本 |
| `dinov3/eval/detection/configs/_base_/dinov3_faster_rcnn_base.py` | Faster R-CNN 基础配置 |
| `dinov3/eval/detection/configs/_base_/dinov3_mask_rcnn_base.py` | Mask R-CNN 基础配置 |
| `dinov3/eval/detection/configs/_base_/coco_detection.py` | COCO 数据集配置 |
| `dinov3/eval/detection/configs/_base_/schedule_1x.py` | 1x schedule (12 epochs) |
| `dinov3/eval/detection/configs/_base_/schedule_3x.py` | 3x schedule (36 epochs) |
| `dinov3/eval/detection/configs/_base_/default_runtime.py` | 默认运行时配置 |
| `dinov3/eval/detection/configs/faster_rcnn/` | Faster R-CNN 各模型具体配置 |
| `dinov3/eval/detection/configs/mask_rcnn/` | Mask R-CNN 各模型具体配置 |
| `dinov3/models/vision_transformer.py` | DinoVisionTransformer (核心 ViT 模型) |
| `dinov3/hub/backbones.py` | Hub API 加载预训练 backbone |

---

## 十、常见问题

### Q: 报错 "MMDetection ecosystem packages not found"

A: 需要安装 MMDetection 及其依赖：
```bash
pip install openmim && mim install mmdet
```

### Q: 训练时图像尺寸报错 "size must be divisible by patch_size"

A: ViT 的 patch embedding 要求输入 H、W 都能被 `patch_size`(16) 整除。检查：
1. `data_preprocessor.pad_size_divisor` 是否设为 16
2. `RandomResize` 的 scale 是否都是 16 的倍数

### Q: 显存不够 (OOM)

A: 参考「显存优化」表格，优先尝试：
1. 确认 backbone 已冻结 (`frozen_stages=-1`)
2. `--batch-size 1`
3. `--amp`
4. 减少 `layers_to_use` 的层数

### Q: Backbone 权重从哪里加载？

A: 默认从 DINOv3 hub 下载预训练权重（LVD-1689M）。可通过 `--backbone-weights` 指定本地路径，或在 config 中配置 `init_cfg`。
